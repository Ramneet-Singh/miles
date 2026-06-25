#!/usr/bin/env bash
# run-super-math-async.sh — B1: disaggregated *async* RL on Nemotron-3-Super-120B-A12B-BF16.
# Same model + dapo-math task as run-super-math-smoke.sh, but DISAGGREGATED and ASYNC:
#   - train_async.py (not train.py); --colocate is rejected by async.
#   - GPUs split into a training pool and a separate rollout pool (no time-sharing).
#   - fully-async rollout: a background worker generates continuously into a buffer;
#     groups older than --max-weight-staleness weight versions are recycled.
#   - --use-tis corrects the resulting off-policyness.
# Disaggregation also removes the colocate host-RAM stack (sglang + train + save on one
# node) that OOM'd the colocate smoke, so --optimizer-cpu-offload is no longer needed.
#
# Runs INSIDE the node0 miles container; attaches to the Ray cluster that
# ansible/ray.yml already started (does NOT manage Ray):
#   docker exec miles bash /root/miles/examples/experimental/swe-agent-v2/cluster/nemotron/run-super-math-async.sh
set -euo pipefail

MILES_ROOT=${MILES_ROOT:-/root/miles}
MODELS_DIR=${MODELS_DIR:-/cpfs01/models}
DATASETS_DIR=${DATASETS_DIR:-/cpfs01/datasets}
HEAD_IP=${HEAD_IP:-10.0.96.128}
cd "$MILES_ROOT"

source "$MILES_ROOT/scripts/models/nemotron-3-super-120b-a12b.sh"   # sets MODEL_ARGS (incl. MoE routing)

CKPT_ARGS=(
   # BF16 HF checkpoint loaded directly via AutoBridge (no torch_dist).
   # No --save / --save-interval: this is a throwaway async-validation run. With any
   # save-interval set, miles force-saves on the final step regardless of the interval
   # (should_run_periodic_action), which is the heavy 120B distcp write we don't need here.
   --hf-checkpoint $MODELS_DIR/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
   --ref-load      $MODELS_DIR/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   # Fully-async rollout: background worker fills a buffer; trainer drains it.
   # Module lives in examples/fully_async (added to PYTHONPATH in RUNTIME_ENV_JSON).
   --rollout-function-path fully_async_rollout.generate_rollout_fully_async
   --prompt-data $DATASETS_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt --label-key label
   --apply-chat-template --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 20              # longer horizon to look for reward hill-climb
   --rollout-batch-size 16       # 16 distinct prompts/step -> lower per-step reward variance
   --n-samples-per-prompt 16     # 16 samples/prompt -> stronger GRPO within-group advantage
   --rollout-max-response-len 8192   # let the thinking model finish + reach \boxed{} (1024 truncated ~92%)
   --rollout-temperature 1
   --global-batch-size 256       # = 16 prompts x 16 samples / 1 step
   --balance-data
)

ASYNC_ARGS=(
   # Bound how stale a rollout may be: groups whose oldest weight version is >2 behind
   # the engine's current version are recycled back to the buffer instead of trained on.
   --max-weight-staleness 2
   # Broadcast fresh weights to the rollout engines every 4 training steps. Less
   # frequent pause+flush (the weight-update barrier), more genuine async overlap,
   # and makes the staleness bound above actually engage. (interval 1 = barely async.)
   --update-weights-interval 4
   # Abort in-flight + queued generations at the weight-update barrier. In fully-async
   # the background worker keeps the engine queue full, so the default "retract" leaves
   # requests pending and flush_cache times out (400: pending requests). "abort" clears
   # them so the cache flushes; aborted prompts are re-generated with fresh weights.
   --pause-generation-mode abort
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216    # cover one full ~8.7k-token sequence (prompt + 8192 response) per microbatch
   --log-probs-chunk-size 128
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl
   --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28
   --use-tis                            # truncated importance sampling: off-policy correction
)

OPTIMIZER_ARGS=(
   --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
   --adam-beta1 0.9 --adam-beta2 0.98
   --use-precision-aware-optimizer
   # Offload optimizer states to host RAM. Fine without it at 1024-token responses,
   # but 8192-token sequences make activations ~9x larger and train() GPU-OOM'd
   # (123/140GB). Offloading the optimizer (~tens of GB/GPU) frees that headroom;
   # the training node has 2TB host RAM so the cost is trivial in disaggregated mode.
   --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8     # TP8 engines; 32 rollout GPUs -> 4 engines
   --sglang-mem-fraction-static 0.7
   # Default sglang_router (Rust) — not --use-miles-router. The Python miles router
   # threw 500/ReadError churn under the continuous fully-async load, leaving
   # requests pending so the weight-update flush_cache timed out. The Rust router is
   # what the proven fully-async example uses and is independent of R3 below.
   --use-rollout-routing-replay        # sigmoid-MoE logprob alignment (doc 5.3)
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
    --wandb-group "super-math-async-$(date +%Y%m%d-%H%M%S)"
    --wandb-key "$WANDB_API_KEY"
    --disable-wandb-random-suffix
  )
fi

# Attach to the EXISTING ray cluster (ansible/ray.yml owns its lifecycle).
export MILES_SCRIPT_EXTERNAL_RAY=1
export MASTER_ADDR=$HEAD_IP

# RoCE env must reach the training actors — ray runtime_env does NOT auto-propagate
# IB_HCA/GID, which caused the first heavy collective to fail (QP mismatch) on DSV4.
# PYTHONPATH includes examples/fully_async so --rollout-function-path resolves.
# NVLS left on (stock behavior for our NVLink H200s).
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/:/root/miles/examples/fully_async\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"1\",
    \"NCCL_IB_HCA\": \"mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7\",
    \"NCCL_IB_GID_INDEX\": \"3\",
    \"NCCL_SOCKET_IFNAME\": \"eth0\",
    \"GLOO_SOCKET_IFNAME\": \"eth0\",
    \"WANDB_DIR\": \"/root/wandb\"
  }
}"

# Disaggregated split: 4 training nodes (32 GPU, 4 DP replicas of TP4*PP2) +
# 32 rollout GPUs (4 TP8 engines). No --colocate (async rejects it).
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes 4 --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 32 \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} ${ASYNC_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} ${MISC_ARGS[@]} ${WANDB_ARGS[@]}
