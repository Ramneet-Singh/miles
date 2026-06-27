#!/usr/bin/env bash
# run-super-tb2-agentic-async.sh — B2: disaggregated *async* *agentic* RL on
# Nemotron-3-Super-120B-A12B-BF16. mini-swe-agent solves Terminal-Bench-2 tasks
# via the Harbor agent server; GRPO trains on the verifier reward.
#
# This is a clean extension of the validated B1 math run (run-super-math-async.sh):
# same disaggregated-async machinery (train_async.py, 4 train / 4 infer, bridge
# weight sync, fully-async rollout, Rust router, abort barrier, TIS), with the
# dapo-math task swapped for an agentic rollout:
#   - prompt-data is the TB2 task list; reward is pre-computed by the agent
#     server (read back via --custom-rm-path), not a math grader.
#   - --custom-generate-function-path opens a TITO session and dispatches each
#     sample to --custom-agent-function-path (swe_agent_function.run), which
#     calls the agent server's POST /run. The agent's model calls flow back
#     through the session server (:30000) -> Rust router -> SGLang engines.
#   - MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1 is required: it gates registration
#     of the custom generate function's args (--max-seq-len,
#     --custom-agent-function-path; arguments.py add_user_provided_function_arguments).
#     It adapts our legacy fully_async rollout fn via the compatibility shim and
#     does not touch the bridge weight-sync — the proven GLM agentic-async combo.
#
# PREREQUISITES (separate from this launcher):
#   - Ray cluster up (ansible/ray.yml), all 8 nodes.
#   - Agent server up on node0 (ansible/agent-server.yml) — POST /run on :11000.
#   - TB2 task dirs at /cpfs01/harbor_tasks/terminal-bench (instance_ids match
#     tb2-tasks.jsonl).
#
# Run INSIDE the node0 miles container (pass the W&B key so it is never committed):
#   docker exec -e WANDB_API_KEY="$WANDB_API_KEY" miles \
#     bash /root/miles/examples/experimental/swe-agent-v2/cluster/nemotron/run-super-tb2-agentic-async.sh
set -euo pipefail

MILES_ROOT=${MILES_ROOT:-/root/miles}
MODELS_DIR=${MODELS_DIR:-/cpfs01/models}
HEAD_IP=${HEAD_IP:-10.0.96.128}
SWE_AGENT_DIR=$MILES_ROOT/examples/experimental/swe-agent-v2
cd "$MILES_ROOT"

source "$MILES_ROOT/scripts/models/nemotron-3-super-120b-a12b.sh"   # sets MODEL_ARGS (incl. MoE routing)

CKPT_ARGS=(
   # BF16 HF checkpoint via AutoBridge; no --save (throwaway validation run).
   --hf-checkpoint $MODELS_DIR/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
   --ref-load      $MODELS_DIR/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   # Fully-async rollout: background worker fills a buffer; trainer drains it.
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --prompt-data $SWE_AGENT_DIR/cluster/nemotron/tb2-tasks.jsonl
   --input-key prompt --metadata-key metadata   # prompt = task instruction; metadata carries instance_id + agent_name
   --rollout-shuffle
   # No --rm-type / --apply-chat-template: reward comes from the agent server
   # (--custom-rm-path below) and the agent builds its own chat via TITO.
   --num-rollout 20              # horizon; watch the FIRST step before committing hours (each step is minutes/trajectory)
   # 32 in-flight trajectories (= rollout-batch-size x n-samples) — the validated
   # envelope. 128-wide saturated the agent/session layer (slow turns, aborts);
   # scale back up once a step lands cleanly.
   --rollout-batch-size 4        # 4 distinct prompts/step
   --n-samples-per-prompt 8      # 8 samples/prompt -> within-group GRPO advantage
   --global-batch-size 32        # = 4 prompts x 8 samples / 1 step
   --rollout-max-response-len 8192   # per-TURN response cap
   --max-seq-len 32768           # full multi-turn trajectory cap; the session server now ENFORCES this
                                 # (context_length_exceeded 400) so the agent ends cleanly instead of running
                                 # to the model's ~256K limit. Sharded across CP2 in training (16k/rank).
   --rollout-temperature 1
   --balance-data
)

AGENT_ARGS=(
   # Agentic generate: TITO session tracing + dispatch to the Harbor agent server.
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path swe_agent_function.run            # POSTs /run to the agent server
   --custom-rm-path generate.reward_func                          # reads the verifier reward the agent server returned
   --use-session-server --session-server-port 30000              # traces each turn; session_server_ip defaults to the router (node0)
   --tito-model nemotron3 --tito-allowed-append-roles tool user   # auto-resolves the chat template; mini-swe-agent appends user-role tool outputs
   # All-or-nothing group filter: drop a group if any sample aborted. Combined
   # with rare weight updates + high staleness below to keep the abort rate low.
   --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_no_aborted
)

ASYNC_ARGS=(
   # Rare weight updates + high staleness: at the weight-update barrier we must
   # abort in-flight generation (the only pause mode that lets flush_cache drain
   # the fully-async queue), which would otherwise kill minutes-long agent
   # trajectories and poison their GRPO groups. So update seldom and tolerate
   # stale rollouts, minimizing the trajectory-time that straddles a barrier.
   --max-weight-staleness 4
   --update-weights-interval 8
   --pause-generation-mode abort
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel                  # shards LayerNorm/dropout activations across TP ranks
   --pipeline-model-parallel-size 2
   --context-parallel-size 2            # shards the long-trajectory SEQUENCE across 2 GPUs -> halves per-GPU
                                        # activation so the MoE forward fits at 32k context. Layout becomes
                                        # TP4*PP2*CP2 = 16 GPU/replica -> DP2. Free of the usual DP/optimizer
                                        # penalty because --optimizer-cpu-offload keeps optimizer states off-GPU.
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384           # ~= max_seq_len // cp_size (CP shards the sequence): one full 32k
                                        # trajectory -> 16k tokens/rank, ~1.8x B1's proven 9216. WATCH for OOM.
   --log-probs-chunk-size 128
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl
   --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28
   --use-tis                            # truncated importance sampling: off-policy correction for the stale async rollouts
)

OPTIMIZER_ARGS=(
   --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
   --adam-beta1 0.9 --adam-beta2 0.98
   --use-precision-aware-optimizer
   # Long trajectories make activations large; offload the optimizer states to
   # the training node's 2TB host RAM (free in disaggregated mode) for headroom.
   --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8     # TP8 engines; 32 rollout GPUs -> 4 engines
   # KV-pool / throughput knob. 0.8 (up from B1's 0.7) gives more concurrent
   # long-context generation while leaving 1-frac headroom for cuda-graphs, R3
   # capture, and the weight-update receive buffer. WATCH for engine OOM right
   # after the first weight broadcast (the symptom of too little headroom).
   --sglang-mem-fraction-static 0.8
   # Default sglang_router (Rust), not --use-miles-router (the Python router
   # churned under fully-async load and timed out flush_cache).
   --use-rollout-routing-replay        # sigmoid-MoE logprob alignment
   --sglang-reasoning-parser nemotron_3   # matches Nemotron3TITOTokenizer
   --sglang-tool-call-parser qwen3_coder
)

MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend auto
   --moe-token-dispatcher-type alltoall  # allgather doesn't support variable seqlen (--use-dynamic-batch-size)
   --distributed-timeout-minutes 30    # defensive: FP/CPFS load + first collective can be slow
)

# W&B (project nemotron-3-super-rl). Enabled only when WANDB_API_KEY is exported
# (pass via `docker exec -e WANDB_API_KEY=... `) so the key is never committed.
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-team proximal_all
    --wandb-project nemotron-3-super-rl
    --wandb-group "super-tb2-agentic-async-$(date +%Y%m%d-%H%M%S)"
    --wandb-key "$WANDB_API_KEY"
    --disable-wandb-random-suffix
  )
fi

# Attach to the EXISTING ray cluster (ansible/ray.yml owns its lifecycle).
export MILES_SCRIPT_EXTERNAL_RAY=1
export MASTER_ADDR=$HEAD_IP

# RoCE env must reach the training actors (ray runtime_env does NOT auto-propagate
# IB_HCA/GID). PYTHONPATH resolves --rollout-function-path (examples/fully_async)
# and the agent/reward modules (swe-agent-v2: swe_agent_function, generate).
# Agent env vars tell swe_agent_function where the agent server is and how the
# agent server should dial back to the session server (MILES_ROUTER_EXTERNAL_HOST).
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
    \"MILES_ROUTER_EXTERNAL_HOST\": \"$HEAD_IP\"
  }
}"

# Disaggregated split: 4 training nodes (32 GPU, 4 DP replicas of TP4*PP2) +
# 32 rollout GPUs (4 TP8 engines). No --colocate (async rejects it).
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 4 --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 32 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${AGENT_ARGS[@]} ${ASYNC_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} ${MISC_ARGS[@]} ${WANDB_ARGS[@]}
