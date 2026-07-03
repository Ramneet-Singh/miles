#!/usr/bin/env bash
# run-qwen3-30b-a3b-swesmith-agentic-async.sh — disaggregated *async* *agentic*
# RL on Qwen3-30B-A3B (text MoE, 3B active, thinking-capable). Chosen over
# Qwen3.5-35B-A3B because the latter is a multimodal checkpoint whose text-only
# HF->Megatron conversion kept hitting bridge walls; this is a clean text model.
# mini-swe-agent fixes SWE-smith bugs (difficulty-graded Harbor tasks) via the
# Harbor agent server; GRPO trains on the partial (combined) verifier reward.
#
# Same harness/dataset as the Nemotron + qwen35 launchers; the levers:
#   1. Inference-heavy GPU split. Nemotron-120B needed all 32 train GPUs to fit;
#      a 30B/A3B model fits an inference engine on ONE H200 (~60 GB BF16), so we
#      move a train node to rollout: 16 train GPUs + 48 rollout GPUs (was 32/32).
#      The runs were rollout-bound, so this is the highest-leverage change.
#   2. Context. Qwen3-30B-A3B is native ~32K (rotary-base 1e6). We run at 32K for
#      now; >32K (our 64K/200K goal) needs YaRN rope-scaling — deferred until the
#      pipeline is green.
#   3. Anti-drift carryover from the Nemotron swesmith run (0.72->0.39 no-anchor
#      drift): lr 2e-6 (was 3e-6) + a small KL coef 0.001 (was 0.0) to anchor
#      the policy, and we now SAVE checkpoints (cheap at 35B) to keep the peak.
#
# PARALLELISM NOTE (smoke-test the first step before trusting the layout):
#   - GQA has num_query_groups=4, so TP can be up to 4; we use TP2. Training layout
#     TP2*PP1*CP2*EP8 = DP4 on 16 GPUs.
#   - CP2: a single trajectory can't be split without CP. At the 32K seq cap CP2
#     shards a full trajectory to 16K/rank (so max-tokens-per-gpu >= 16384; we set
#     32768 for packing headroom). If it OOMs, raise to CP4 (8K/rank) then CP8.
#   - Recompute is KEPT. The small model is a big *inference* win (1 GPU/engine)
#     but only a modest *training-memory* win: stored activation is dominated by
#     LAYER COUNT (48), not per-token size, so long-context training still needs
#     recompute. Drop to selective/none only if smoke shows real headroom.
#
# PREREQUISITES (separate from this launcher):
#   0. Model weights on disk: /cpfs01/models/Qwen3-30B-A3B
#        hf download Qwen/Qwen3-30B-A3B --local-dir /cpfs01/models/Qwen3-30B-A3B
#      (~60 GB; check `ls /cpfs01/models`.)
#   1. Ray cluster up (ansible/ray.yml), all 8 nodes.
#   2. Agent server up on node0 (ansible/agent-server.yml) with the swe-smith
#      tasks + combined reward:
#        -e harbor_tasks_dir=/cpfs01/swe-smith-tasks/py -e harbor_reward_key=combined
#      (regenerate the 150 tasks with stage-swesmith-tasks.sh if starting fresh).
#      NOTE: until the Modal backend swap lands, AGENT_MAX_CONCURRENT stays 16
#      (node0 docker), so 128/step still runs in ~8 waves — the full throughput
#      win from the inference-heavy split is realized once Modal lifts that cap.
#
# Run INSIDE the node0 miles container (pass the W&B key so it is never committed):
#   docker exec -e WANDB_API_KEY="$WANDB_API_KEY" miles \
#     bash /root/miles/examples/experimental/swe-agent-v2/cluster/nemotron/run-qwen3-30b-a3b-swesmith-agentic-async.sh
set -euo pipefail

MILES_ROOT=${MILES_ROOT:-/root/miles}
MODELS_DIR=${MODELS_DIR:-/cpfs01/models}
HEAD_IP=${HEAD_IP:-10.0.96.128}
# Host the sandbox-side agent uses to reach the session server. Private IP for the
# docker backend; set ROUTER_EXTERNAL_HOST=127.0.0.1 for the Modal SSH-tunnel path
# (the task container forwards its localhost:30000 -> node0 over ssh).
ROUTER_EXTERNAL_HOST=${ROUTER_EXTERNAL_HOST:-$HEAD_IP}
# Rollout width / length — env-overridable for smoke tests; defaults are the real
# 8x16 / 50-step run. Keep GLOBAL_BATCH_SIZE = ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT.
NUM_ROLLOUT=${NUM_ROLLOUT:-50}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-128}
SWE_AGENT_DIR=$MILES_ROOT/examples/experimental/swe-agent-v2
CKPT_DIR=${CKPT_DIR:-/cpfs01/ckpts/qwen3-30b-a3b-swesmith}
cd "$MILES_ROOT"

source "$MILES_ROOT/scripts/models/qwen3-30B-A3B.sh"   # sets MODEL_ARGS (text MoE: 48L, 128 experts topk8, GQA=4)

CKPT_ARGS=(
   # BF16 HF checkpoint via AutoBridge. Unlike the throwaway Nemotron runs we now
   # SAVE (35B checkpoints are cheap) so the reward peak is recoverable.
   --hf-checkpoint $MODELS_DIR/Qwen3-30B-A3B
   --ref-load      $MODELS_DIR/Qwen3-30B-A3B
   --megatron-to-hf-mode bridge
   --save $CKPT_DIR
   --save-interval 10            # /cpfs01 has TBs free; the host-NVME pressure is containers, not ckpts.
   --no-save-optim               # weights-only: this is a 50-step probe we don't resume, so skip the ~3x
                                 # optimizer-state bloat (each save ~70 GB not ~280 GB). Keep-latest-only is
                                 # enforced by a post-save prune of old iter_* dirs in the tracking loop.
)

ROLLOUT_ARGS=(
   # Fully-async rollout: background worker fills a buffer; trainer drains it.
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --prompt-data $SWE_AGENT_DIR/cluster/nemotron/swe-smith-py-150.jsonl
   --input-key prompt --metadata-key metadata   # prompt = task instruction; metadata carries instance_id + agent_name
   --rollout-shuffle
   # No --rm-type / --apply-chat-template: reward comes from the agent server
   # (--custom-rm-path below) and the agent builds its own chat via TITO.
   --num-rollout $NUM_ROLLOUT              # default 50 (8x16 probe); override NUM_ROLLOUT for smoke
   --rollout-batch-size $ROLLOUT_BATCH_SIZE        # distinct prompts/step (default 8)
   --n-samples-per-prompt $N_SAMPLES_PER_PROMPT    # samples/prompt (default 16) -> dense GRPO gradient
   --global-batch-size $GLOBAL_BATCH_SIZE          # = rollout-batch * n-samples (default 128)
   --rollout-max-response-len 16384  # per-TURN cap raised 8k->16k: Qwen3 emits <think> blocks, so a single
                                     # turn (reasoning + one bash command) is longer than Nemotron's.
   --max-seq-len 32768           # Qwen3-30B-A3B is native ~32K (rotary-base 1e6). Stay at 32K for now (no YaRN).
                                 # For >32K (our 64K/200K goal) enable YaRN rope-scaling (MODEL_ARGS_ROTARY_BASE
                                 # + rope-scaling) — deferred until the pipeline is green.
   --rollout-temperature 1
   --balance-data
)

AGENT_ARGS=(
   # Agentic generate: TITO session tracing + dispatch to the Harbor agent server.
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path swe_agent_function.run            # POSTs /run to the agent server
   --custom-rm-path generate.reward_func                          # reads the verifier reward the agent server returned
   --use-session-server --session-server-port 30000              # traces each turn; session_server_ip defaults to the router (node0)
   --tito-model qwen3 --tito-allowed-append-roles tool user       # qwen3 chat template (thinking-capable); mini-swe-agent appends user-role tool outputs
   # All-or-nothing group filter: drop a group if any sample aborted.
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
)

ASYNC_ARGS=(
   # Rare weight updates + high staleness: at the weight-update barrier we abort
   # in-flight generation (the only pause mode that lets flush_cache drain the
   # fully-async queue), so update seldom and tolerate stale rollouts to minimize
   # the trajectory-time that straddles a barrier and poisons a GRPO group.
   --max-weight-staleness 4
   --update-weights-interval 8
   --pause-generation-mode abort
)

PERF_ARGS=(
   --tensor-model-parallel-size 2       # GQA num_query_groups=4 allows up to TP4; we use TP2
   --sequence-parallel                  # shards LayerNorm/dropout activations across TP ranks
   --pipeline-model-parallel-size 1
   --context-parallel-size 2            # CP2: shards a full 32K trajectory to 16K/rank (see PARALLELISM NOTE).
                                        # Starting point; raise to CP4 (8K/rank) then CP8 (=> DP1) if it OOMs.
   --expert-model-parallel-size 8       # 128 experts sharded 8-way; EP spans the group (TP2*PP1*CP2 => DP4 on 16 GPU)
   --expert-tensor-parallel-size 1
   --recompute-granularity full         # activation is layer-count-dominated (48 layers), so still needed for
   --recompute-method uniform           # long context; drop to selective/none only if the smoke step has headroom.
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768           # FLOOR is the per-rank shard: at CP2 a full 32K trajectory is 16K/rank,
                                        # so >=16384 fits it; 32768 packs more. recompute keeps activation bounded.
                                        # (Do NOT set PYTORCH_CUDA_ALLOC_CONF=
                                        # expandable_segments globally: it breaks the SGLang engines' graph capture.)
   --log-probs-chunk-size 128
)

GRPO_ARGS=(
   --advantage-estimator grpo
   # Anti-drift carryover: the Nemotron swesmith run climbed 0.49->0.72 then drifted
   # to ~0.39 with kl_coef 0.0 (no anchor). Add a small KL coef to hold the policy
   # near the reference while it climbs.
   --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl
   --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28
   --use-tis                            # truncated importance sampling: off-policy correction for stale async rollouts
)

OPTIMIZER_ARGS=(
   --optimizer adam --lr 2e-6 --lr-decay-style constant --weight-decay 0.1   # lr 3e-6->2e-6: the drift was a
                                 # too-fast walk past the easy gains; pair the smaller step with the KL anchor.
   --adam-beta1 0.9 --adam-beta2 0.98
   --use-precision-aware-optimizer
   # Offload optimizer states to the training node's host RAM (free in disaggregated mode).
   --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1     # TP1 engines: 35B fits one H200 -> 48 engines on 48 rollout GPUs
                                        # (vs Nemotron's 8-GPU TP8 engines). This is the concurrency unlock.
   # KV-pool / throughput knob. 0.8 leaves headroom for cuda-graphs, weight-update
   # receive buffer, etc. WATCH for engine OOM right after the first weight broadcast.
   --sglang-mem-fraction-static 0.8
   # FOLLOW-UP throughput lever (not enabled yet): FP8 rollout precision on H200
   # (Hopper has no FP4). Verify the miles/SGLang quant flag before enabling; keep
   # training in BF16. Expected ~1.5-2x rollout tok/s.
   --sglang-reasoning-parser qwen3     # parse <think></think> for the thinking-capable qwen3
   --sglang-tool-call-parser qwen3_coder
   # Timeout cascade (router < session < agent), all > the observed legit turn:
   #   router->engine 300s  <  session->router 450s  <  agent litellm 600s.
   --sglang-router-request-timeout-secs 300
   --miles-router-timeout 450           # session-server->router proxy client (session_server.py)
)

MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend auto
   # NB: --moe-token-dispatcher-type is set in the model def (alltoall) — do not duplicate.
   --distributed-timeout-minutes 30    # defensive: FP/CPFS load + first collective can be slow
)

# W&B. Enabled only when WANDB_API_KEY is exported (pass via `docker exec -e
# WANDB_API_KEY=...`) so the key is never committed.
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-team proximal_all
    --wandb-project qwen3-30b-a3b-swesmith
    --wandb-group "qwen3-30b-a3b-swesmith-agentic-async-$(date +%Y%m%d-%H%M%S)"
    --disable-wandb-random-suffix
  )
fi

# Attach to the EXISTING ray cluster (ansible/ray.yml owns its lifecycle).
export MILES_SCRIPT_EXTERNAL_RAY=1
export MASTER_ADDR=$HEAD_IP

# RoCE env must reach the training actors (ray runtime_env does NOT auto-propagate
# IB_HCA/GID). PYTHONPATH resolves --rollout-function-path (examples/fully_async)
# and the agent/reward modules (swe-agent-v2). Agent env vars tell
# swe_agent_function where the agent server is and how it should dial back to the
# session server (MILES_ROUTER_EXTERNAL_HOST).
# WANDB_API_KEY travels in the runtime-env FILE below (never in argv, so not
# visible in `ps`). wandb.init() in the actors reads it from the env. Empty when unset.
WANDB_ENV_KV=""
[ -n "${WANDB_API_KEY:-}" ] && WANDB_ENV_KV="\"WANDB_API_KEY\": \"$WANDB_API_KEY\","
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    $WANDB_ENV_KV
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/miles/examples/fully_async:$SWE_AGENT_DIR:/root/miles\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",

    \"NCCL_NVLS_ENABLE\": \"1\",
    \"NCCL_IB_HCA\": \"mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7\",
    \"NCCL_IB_GID_INDEX\": \"3\",
    \"NCCL_SOCKET_IFNAME\": \"eth0\",
    \"GLOO_SOCKET_IFNAME\": \"eth0\",
    \"WANDB_DIR\": \"/root/wandb\",
    \"AGENT_SERVER_URL\": \"http://$HEAD_IP:11000\",
    \"AGENT_MODEL_NAME\": \"model\",
    \"MILES_ROUTER_EXTERNAL_HOST\": \"$ROUTER_EXTERNAL_HOST\"
  }
}"

# Write the runtime env to a 0600 file and pass --runtime-env (a path), NOT
# --runtime-env-json (inline), so WANDB_API_KEY never lands in the process argv
# (visible via `ps`). Removed on exit.
RUNTIME_ENV_FILE=$(mktemp /tmp/miles_runtime_env.XXXXXX.json)
chmod 600 "$RUNTIME_ENV_FILE"
printf '%s' "$RUNTIME_ENV_JSON" > "$RUNTIME_ENV_FILE"
trap 'rm -f "$RUNTIME_ENV_FILE"' EXIT

# Disaggregated, inference-heavy split: 2 training nodes (16 GPU: TP2*PP1*CP2 =>
# DP4, EP8) + 48 rollout GPUs (48 TP1 engines). No --colocate (async rejects it).
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env "$RUNTIME_ENV_FILE" \
   -- python3 train_async.py \
   --actor-num-nodes 2 --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 48 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${AGENT_ARGS[@]} ${ASYNC_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} ${MISC_ARGS[@]} ${WANDB_ARGS[@]}
