#!/usr/bin/env bash
# run-super-math-smoke.sh — Phase-A smoke: validate Nemotron-3-Super-120B-A12B-BF16
# loads + trains (GRPO on dapo-math) on OUR live 64-GPU cluster. Non-agentic.
#
# Runs INSIDE the node0 miles container; attaches to the Ray cluster that
# ansible/ray.yml already started (does NOT manage Ray):
#   docker exec miles bash /root/miles/examples/experimental/swe-agent-v2/cluster/nemotron/run-super-math-smoke.sh
set -euo pipefail

MILES_ROOT=${MILES_ROOT:-/root/miles}
MODELS_DIR=${MODELS_DIR:-/cpfs01/models}
DATASETS_DIR=${DATASETS_DIR:-/cpfs01/datasets}
HEAD_IP=${HEAD_IP:-10.0.96.128}
cd "$MILES_ROOT"

source "$MILES_ROOT/scripts/models/nemotron-3-super-120b-a12b.sh"   # sets MODEL_ARGS (incl. MoE routing)

CKPT_ARGS=(
   # BF16 HF checkpoint loaded directly via AutoBridge (no torch_dist).
   --hf-checkpoint $MODELS_DIR/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
   --ref-load      $MODELS_DIR/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
   --save          $MODELS_DIR/nemotron-3-super-120b-a12b_miles
   --save-interval 20
   --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
   --prompt-data $DATASETS_DIR/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt --label-key label
   --apply-chat-template --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 5               # smoke: a handful of steps
   --rollout-batch-size 32
   --n-samples-per-prompt 4
   --rollout-max-response-len 1024
   --rollout-temperature 1
   --global-batch-size 128       # = 32 prompts x 4 samples / 1 step
   --balance-data
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
   --max-tokens-per-gpu 1024
   --log-probs-chunk-size 128
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl
   --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1
   --adam-beta1 0.9 --adam-beta2 0.98
   --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8     # 8 TP8 engines across 64 GPU
   --sglang-mem-fraction-static 0.7
   --use-miles-router
   --use-rollout-routing-replay        # sigmoid-MoE logprob alignment (doc 5.3)
)

MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend auto
   --moe-token-dispatcher-type alltoall  # allgather doesn't support variable seqlen (--use-dynamic-batch-size)
   --distributed-timeout-minutes 30    # defensive: FP/CPFS load + first 64-GPU collective can be slow
)

# W&B (project nemotron-3-super-rl). Enabled only when WANDB_API_KEY is exported
# (pass via `docker exec -e WANDB_API_KEY=... `) so the key is never committed.
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-team proximal_all
    --wandb-project nemotron-3-super-rl
    --wandb-group "super-math-smoke-$(date +%Y%m%d-%H%M%S)"
    --wandb-key "$WANDB_API_KEY"
    --disable-wandb-random-suffix
  )
fi

# Attach to the EXISTING ray cluster (ansible/ray.yml owns its lifecycle).
export MILES_SCRIPT_EXTERNAL_RAY=1
export MASTER_ADDR=$HEAD_IP

# RoCE env must reach the training actors — ray runtime_env does NOT auto-propagate
# IB_HCA/GID, which caused the first heavy collective to fail (QP mismatch) on DSV4.
# NVLS left on (stock behavior for our NVLink H200s).
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"1\",
    \"NCCL_IB_HCA\": \"mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7\",
    \"NCCL_IB_GID_INDEX\": \"3\",
    \"NCCL_SOCKET_IFNAME\": \"eth0\",
    \"GLOO_SOCKET_IFNAME\": \"eth0\",
    \"WANDB_DIR\": \"/root/wandb\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 8 --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 64 --colocate \
   ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} ${GRPO_ARGS[@]} ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} ${MISC_ARGS[@]} ${WANDB_ARGS[@]}
