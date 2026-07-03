#!/usr/bin/env bash
# run-qwen35-swesmith-agentic-async.sh — disaggregated *async* *agentic* RL on
# Qwen3.5-35B-A3B (MoE, 3B active, 256K native context, thinking-capable).
# mini-swe-agent fixes SWE-smith bugs (difficulty-graded Harbor tasks) via the
# Harbor agent server; GRPO trains on the partial (combined) verifier reward.
#
# This is the MODEL swap of run-super-swesmith-agentic-async.sh. We keep the
# dataset (swe-smith-py-150.jsonl) and the harness identical — the only variables
# are the model and the knobs that the smaller model unlocks:
#   1. Inference-heavy GPU split. Nemotron-120B needed all 32 train GPUs just to
#      fit (TP4*PP2*CP4*EP8); a 35B/A3B model fits an inference engine on ONE
#      H200 (~70 GB BF16 weights), so we move a train node to rollout: 16 train
#      GPUs + 48 rollout GPUs (was 32/32). The runs were rollout-bound (trainer
#      ~50 s vs rollout ~1575 s), so this is the highest-leverage change.
#   2. Longer context. Native 256K (rotary-base 1e7) means no YaRN below that.
#      We run rollouts at a 64K cap (up from 32K) to exercise thinking-heavy
#      multi-turn trajectories; 200K stays available as headroom for future
#      long-horizon tasks (raise --max-seq-len then — costs KV/concurrency).
#   3. Anti-drift carryover from the Nemotron swesmith run (0.72->0.39 no-anchor
#      drift): lr 2e-6 (was 3e-6) + a small KL coef 0.001 (was 0.0) to anchor
#      the policy, and we now SAVE checkpoints (cheap at 35B) to keep the peak.
#
# PARALLELISM NOTE (smoke-test the first step before trusting the layout):
#   - TP is capped at 2 by GQA: the model has num_query_groups=2, so TP4 can't
#     shard the KV heads. Training uses TP2*PP1*CP2*EP8 = DP4 on 16 GPUs.
#   - CP2 (starting point): a single trajectory can't be split without CP, so at
#     CP1 a full-length 64K trajectory is one 64K-token per-rank microbatch that
#     must fit alone (max-tokens-per-gpu >= 65536) -> OOM. CP2 shards 64K ->
#     32K/rank (so max-tokens-per-gpu = 32768, below). If 32K/rank OOMs, raise to
#     CP4 (16K/rank, max-tokens 16384) then CP8 (8K/rank, => DP1).
#   - Recompute is KEPT. The smaller model is a big *inference* win (1 GPU/engine)
#     but only a modest *training-memory* win: stored activation is dominated by
#     LAYER COUNT (still 40), not per-token size, so long-context training still
#     needs recompute. Drop to selective/none only if smoke shows real headroom.
#
# PREREQUISITES (separate from this launcher):
#   0. Model weights on disk: /cpfs01/models/Qwen3.5-35B-A3B
#        huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir /cpfs01/models/Qwen3.5-35B-A3B
#      (~70 GB; not yet present — check `ls /cpfs01/models`.)
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
#     bash /root/miles/examples/experimental/swe-agent-v2/cluster/nemotron/run-qwen35-swesmith-agentic-async.sh
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
CKPT_DIR=${CKPT_DIR:-/cpfs01/ckpts/qwen35-swesmith}
cd "$MILES_ROOT"

source "$MILES_ROOT/scripts/models/qwen3.5-35B-A3B.sh"   # sets MODEL_ARGS (arch + MoE routing + MTP)

CKPT_ARGS=(
   # BF16 HF checkpoint via AutoBridge. Unlike the throwaway Nemotron runs we now
   # SAVE (35B checkpoints are cheap) so the reward peak is recoverable.
   --hf-checkpoint $MODELS_DIR/Qwen3.5-35B-A3B
   --ref-load      $MODELS_DIR/Qwen3.5-35B-A3B
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
   --rollout-max-response-len 16384  # per-TURN cap raised 8k->16k: Qwen3.5 emits <think> blocks, so a single
                                     # turn (reasoning + one bash command) is longer than Nemotron's.
   --max-seq-len 65536           # full multi-turn trajectory cap, 32k->64k. Model supports 256K natively; we
                                 # cap at runtime to protect KV/concurrency (a 64k seq ~= 10 GB KV at TP1).
                                 # Raise toward 200K only for genuinely long-horizon tasks.
   --rollout-temperature 1
   --balance-data
)

AGENT_ARGS=(
   # Agentic generate: TITO session tracing + dispatch to the Harbor agent server.
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path swe_agent_function.run            # POSTs /run to the agent server
   --custom-rm-path generate.reward_func                          # reads the verifier reward the agent server returned
   --use-session-server --session-server-port 30000              # traces each turn; session_server_ip defaults to the router (node0)
   --tito-model qwen35 --tito-allowed-append-roles tool user      # qwen35 chat template (thinking-capable); mini-swe-agent appends user-role tool outputs
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
   --tensor-model-parallel-size 2       # capped at 2 by GQA (num_query_groups=2)
   --sequence-parallel                  # shards LayerNorm/dropout activations across TP ranks
   --pipeline-model-parallel-size 1
   --context-parallel-size 2            # CP2: shards a full 64K trajectory to 32K/rank (see PARALLELISM NOTE).
                                        # Starting point; raise to CP4 (16K/rank, max-tokens 16384) then CP8
                                        # (8K/rank, => DP1) if 32K/rank OOMs.
   --expert-model-parallel-size 8       # 256 experts sharded 8-way; EP spans the group (TP2*PP1*CP2 => DP4 on 16 GPU)
   --expert-tensor-parallel-size 1
   --recompute-granularity full         # activation is layer-count-dominated (40 layers), so still needed for
   --recompute-method uniform           # 64K context; drop to selective/none only if the smoke step has headroom.
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 32768           # 4x Nemotron's 8192. FLOOR is the per-rank shard: at CP2 a full-length
                                        # trajectory is 32K/rank, so 32768 lets it form one microbatch. The
                                        # ~4x-lighter per-token activation + recompute keep that in memory; it
                                        # tracks layer count (not per-token size), so this is the ceiling of what
                                        # the halving from CP4 buys. (Do NOT set PYTORCH_CUDA_ALLOC_CONF=
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
   --sglang-reasoning-parser qwen3     # parse <think></think> for the thinking-capable qwen3.5 (was nemotron_3)
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
    --wandb-project qwen35-swesmith
    --wandb-group "qwen35-swesmith-agentic-async-$(date +%Y%m%d-%H%M%S)"
    --wandb-key "$WANDB_API_KEY"
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
RUNTIME_ENV_JSON="{
  \"env_vars\": {
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

# Disaggregated, inference-heavy split: 2 training nodes (16 GPU: TP2*PP1*CP2 =>
# DP4, EP8) + 48 rollout GPUs (48 TP1 engines). No --colocate (async rejects it).
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 2 --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 48 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${AGENT_ARGS[@]} ${ASYNC_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} ${MISC_ARGS[@]} ${WANDB_ARGS[@]}
