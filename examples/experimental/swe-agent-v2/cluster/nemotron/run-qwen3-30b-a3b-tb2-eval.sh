#!/usr/bin/env bash
# run-qwen3-30b-a3b-tb2-eval.sh — OFFLINE held-out eval of the E3 checkpoints on
# Terminal-Bench-2 (89 tasks, decontaminated from the swe-smith train repos).
#
# Uses the built-in EVAL-ONLY path (train.py:39-40): --num-rollout 0 +
# --eval-interval 1 loads a saved Megatron checkpoint, broadcasts it to the
# SGLang engines, runs ONE agentic eval over TB2, and exits — no training, no
# optimizer, no weight updates beyond the initial load->engine sync.
#
# The agentic eval reuses the exact training data path (agentic_tool_call.generate
# -> swe_agent_function.run -> agent server /run -> verifier reward), via
# --eval-function-path generate.RolloutFn (InferenceRolloutFn._call_eval). NOTE:
# the fully-async rollout fn RAISES on eval, so we run train.py (not train_async)
# and set the eval fn explicitly.
#
# Per checkpoint the mean reward is logged as `eval/tb2` (metrics.py) to stdout
# and W&B. Loops over CKPT_STEPS (default the 5 E3 saves 4/9/14/19/24); each is
# an independent Ray job that reloads+rebroadcasts that iteration's weights.
#
# PREREQUISITES:
#   0. Checkpoints at /cpfs01/ckpts/qwen3-30b-a3b-swesmith/iter_000000{4,9,14,19,24}
#      and base model /cpfs01/models/Qwen3-30B-A3B (tokenizer/config + bridge).
#   1. Ray cluster up (ansible/ray.yml).
#   2. Agent server up on node0 pointed at TB2 with the modal backend:
#        ansible-playbook agent-server.yml -e recreate=true -e harbor_env_type=modal
#      (defaults: harbor_tasks_dir=/cpfs01/harbor_tasks/terminal-bench,
#       harbor_reward_key=reward — TB2 verifiers emit "reward", NOT "combined").
#
# Run INSIDE the node0 miles container (W&B key via -e so it's never committed):
#   docker exec -e WANDB_API_KEY="$WANDB_API_KEY" miles \
#     bash /root/miles/examples/experimental/swe-agent-v2/cluster/nemotron/run-qwen3-30b-a3b-tb2-eval.sh
set -euo pipefail

MILES_ROOT=${MILES_ROOT:-/root/miles}
MODELS_DIR=${MODELS_DIR:-/cpfs01/models}
HEAD_IP=${HEAD_IP:-10.0.96.128}
# The in-sandbox agent dials the session server via the Modal proxy's allowlisted
# node0 public ingress (same as the training run).
ROUTER_EXTERNAL_HOST=${ROUTER_EXTERNAL_HOST:-47.74.85.155}
CKPT_DIR=${CKPT_DIR:-/cpfs01/ckpts/qwen3-30b-a3b-swesmith}
CKPT_STEPS=${CKPT_STEPS:-"4 9 14 19 24"}         # override e.g. CKPT_STEPS="4" to validate one first
N_SAMPLES_PER_EVAL=${N_SAMPLES_PER_EVAL:-3}       # attempts/task averaged out (1 = single 89-task pass)
SWE_AGENT_DIR=$MILES_ROOT/examples/experimental/swe-agent-v2
cd "$MILES_ROOT"

source "$MILES_ROOT/scripts/models/qwen3-30B-A3B.sh"   # MODEL_ARGS (text MoE: 48L, 128 experts topk8, GQA=4)

CKPT_ARGS=(
   # --hf-checkpoint supplies tokenizer/config + the AutoBridge; --load points at
   # the SAVE dir and --ckpt-step selects the iteration (native Megatron torch_dist
   # load, parallelism-agnostic reshard). Weights-only saves => --no-load-optim.
   --hf-checkpoint $MODELS_DIR/Qwen3-30B-A3B
   --load $CKPT_DIR
   --no-load-optim
   --megatron-to-hf-mode bridge
   # train.py's startup update_weights() broadcasts from the weights_backuper,
   # which is only populated when _enable_weight_backup is True. That gate is
   # `with_ref or keep_old_actor or colocate`, and with_ref = (kl_coef != 0 or
   # use_kl_loss) (placement_group.py:147). So --ref-load ALONE is NOT enough —
   # we also need --use-kl-loss to flip with_ref. Without the gate the backup is
   # empty -> "not in new_weight_dict, list=[]". Both are INERT at num-rollout 0
   # (no KL is computed; the ref model is loaded but unused) — they only enable
   # the backup so the loaded checkpoint can be broadcast to the engines.
   --ref-load $MODELS_DIR/Qwen3-30B-A3B
   --use-kl-loss --kl-loss-coef 0.001 --kl-loss-type low_var_kl
   # (ckpt-step is appended per-iteration in the loop below.)
)

EVAL_ARGS=(
   # Eval-only: num-rollout 0 -> train.py runs exactly one eval and exits.
   --num-rollout 0
   --eval-interval 1
   # Required by train.py's argparse even for eval-only; INERT at num-rollout 0
   # (no training rollouts/steps). Present only so arg parsing + train_iters math
   # (num_rollout * rollout_batch * n_samples / global_batch = 0) don't choke.
   --rollout-batch-size 8 --global-batch-size 128 --n-samples-per-prompt 16
   --eval-function-path generate.RolloutFn            # agentic eval; NOT the fully-async fn (it raises on eval)
   --rollout-function-path generate.RolloutFn         # unused at num-rollout 0, but keeps the fallback sane
   --eval-prompt-data tb2 $SWE_AGENT_DIR/cluster/nemotron/tb2-tasks.jsonl
   # --prompt-data (training set) is loaded by the RolloutManager's data_source at
   # init even in eval-only mode; omitting it -> None -> TypeError. Point it at the
   # same TB2 file — INERT at num-rollout 0 (never consumed for training rollouts).
   --prompt-data $SWE_AGENT_DIR/cluster/nemotron/tb2-tasks.jsonl
   --input-key prompt --metadata-key metadata         # inherited by eval; carries instance_id + agent_name to the agent
   --n-samples-per-eval-prompt $N_SAMPLES_PER_EVAL
   --eval-temperature 1                               # match training sampling
   --eval-max-response-len 16384                      # per-turn cap, same as training
   --max-seq-len 40960                                # Qwen3-30B-A3B native context
   # NB: do NOT set --reward-key/--eval-reward-key — the agent server returns a
   # plain float reward; indexing it by key would crash (inference_rollout_eval).
)

AGENT_ARGS=(
   --custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate
   --custom-agent-function-path swe_agent_function.run
   --custom-rm-path generate.reward_func
   --use-session-server --session-server-port 30000
   --tito-model qwen3 --tito-allowed-append-roles tool user
)

PERF_ARGS=(
   # Same proven layout as training; the trainer only loads+broadcasts here (eval
   # forward passes run on the SGLang engines), and torch_dist reshards, so this
   # is about holding the 30B checkpoint, not training throughput.
   --tensor-model-parallel-size 2 --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 4
   --expert-model-parallel-size 8 --expert-tensor-parallel-size 1
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
   --use-dynamic-batch-size --max-tokens-per-gpu 16384 --log-probs-chunk-size 128
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
   --sglang-reasoning-parser qwen3
   --sglang-tool-call-parser qwen3_coder
   --sglang-router-request-timeout-secs 300
   --miles-router-timeout 450
)

OPTIMIZER_ARGS=(
   # Eval-only STILL fully initializes a training actor (optimizer + LR scheduler).
   # The scheduler asserts lr_decay_steps = lr_decay_iters * global_batch_size > 0
   # (model.py:69), and num-rollout 0 leaves lr_decay_iters 0 -> crash. Set a dummy
   # positive --lr-decay-iters; the optimizer is constructed but NEVER stepped here.
   --optimizer adam --lr 2e-6 --lr-decay-style constant --lr-decay-iters 1
   --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98
   --use-precision-aware-optimizer
   --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d
)

MISC_ARGS=(
   --attention-dropout 0.0 --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend auto
   --distributed-timeout-minutes 30
)

# Attach to the existing Ray cluster.
export MILES_SCRIPT_EXTERNAL_RAY=1
export MASTER_ADDR=$HEAD_IP

# WANDB_API_KEY rides a 0600 runtime-env FILE (never in argv). Empty when unset.
WANDB_ENV_KV=""
[ -n "${WANDB_API_KEY:-}" ] && WANDB_ENV_KV="\"WANDB_API_KEY\": \"$WANDB_API_KEY\","
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    $WANDB_ENV_KV
    \"PYTHONPATH\": \"/root/Megatron-LM/:$SWE_AGENT_DIR:/root/miles\",
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
RUNTIME_ENV_FILE=$(mktemp /tmp/miles_runtime_env.XXXXXX.json)
chmod 600 "$RUNTIME_ENV_FILE"
printf '%s' "$RUNTIME_ENV_JSON" > "$RUNTIME_ENV_FILE"
trap 'rm -f "$RUNTIME_ENV_FILE"' EXIT

for N in $CKPT_STEPS; do
  echo "===== TB2 EVAL: checkpoint iter_$(printf '%07d' "$N") (ckpt-step $N) ====="
  WANDB_ARGS=()
  if [ -n "${WANDB_API_KEY:-}" ]; then
    WANDB_ARGS=(
      --use-wandb --wandb-team proximal_all --wandb-project qwen3-30b-a3b-swesmith
      --wandb-group "tb2-eval-iter${N}" --disable-wandb-random-suffix
    )
  fi
  ray job submit --address="http://127.0.0.1:8265" \
     --runtime-env "$RUNTIME_ENV_FILE" \
     -- python3 train.py \
     --actor-num-nodes 2 --actor-num-gpus-per-node 8 \
     --rollout-num-gpus 48 \
     --ckpt-step "$N" \
     ${MODEL_ARGS[@]} ${CKPT_ARGS[@]} ${EVAL_ARGS[@]} ${AGENT_ARGS[@]} \
     ${OPTIMIZER_ARGS[@]} ${PERF_ARGS[@]} ${SGLANG_ARGS[@]} ${MISC_ARGS[@]} ${WANDB_ARGS[@]}
  echo "===== done iter_$(printf '%07d' "$N") ====="
done
