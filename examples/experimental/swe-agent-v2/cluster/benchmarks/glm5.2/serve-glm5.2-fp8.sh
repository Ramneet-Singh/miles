#!/usr/bin/env bash
# serve-glm5.2-fp8.sh — stand up a single-node SGLang server for GLM-5.2-FP8 on
# 8x H200, following the official SGLang cookbook recipes
# (docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.2, hw=h200, quant=fp8,
# nodes=single). Three strategies, selected by MODE:
#
#   MODE=high-throughput (default) — DP-attention + DeepEP MoE all-to-all, NO
#                      speculative decoding (spec wastes compute once batches are
#                      full), max-running 256. Maximizes aggregate tok/s.
#   MODE=balanced      — same DP-attn + DeepEP but with light EAGLE MTP (1-1-2)
#                      and chunked prefill; trades a little peak throughput for
#                      better latency.
#   MODE=low-latency   — plain TP8, heavy EAGLE MTP (5-1-6) to minimize per-token
#                      latency at low concurrency.
#
# GLM-5.2 (arch GlmMoeDsaForCausalLM, 744B-A40B, DeepSeek-Sparse-Attention +
# built-in MTP/EAGLE head) is NOT supported by the in-cluster miles SGLang
# (0.5.14). We therefore run it in a SEPARATE, self-contained
# `lmsysorg/sglang:latest` container — no interaction with the miles/Ray stack.
#
# The container takes all 8 GPUs of the host it runs on (default: node0, whose
# GPUs are idle when no training job is running). Weights are mounted read-only
# from CPFS so the 755 GB FP8 checkpoint is never copied to host NVMe.
#
# Usage:
#   bash serve-glm5.2-fp8.sh                       # balanced, wait for /health
#   MODE=low-latency bash serve-glm5.2-fp8.sh
#   CONTEXT_LEN=65536 bash serve-glm5.2-fp8.sh
#   bash serve-glm5.2-fp8.sh stop                  # tear the server down
set -euo pipefail

CONTAINER=${CONTAINER:-glm52-sglang}
IMAGE=${IMAGE:-lmsysorg/sglang:latest}
MODEL_DIR=${MODEL_DIR:-/cpfs01/models/GLM-5.2-FP8}
PORT=${PORT:-30000}
MODE=${MODE:-high-throughput}          # high-throughput | balanced | low-latency
# GLM-5.2 is natively 1M-context; that would reserve an enormous KV cache and
# starve concurrency. Bound it to a benchmark-appropriate window (>= max
# input+output of the sweep). Override for long-context tests.
CONTEXT_LEN=${CONTEXT_LEN:-32768}
LOG=${LOG:-/cpfs01/logs/glm5.2-serve.log}

if [ "${1:-}" = "stop" ]; then
  docker rm -f "$CONTAINER" 2>/dev/null && echo "stopped $CONTAINER" || echo "no $CONTAINER running"
  exit 0
fi

# Preconditions: image present, weights fully staged (config + all shards).
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "ERROR: image $IMAGE not pulled yet"; exit 1; }
[ -f "$MODEL_DIR/config.json" ] || { echo "ERROR: $MODEL_DIR/config.json missing (download incomplete)"; exit 1; }
n_shards=$(ls "$MODEL_DIR"/model-*.safetensors 2>/dev/null | wc -l)
[ "$n_shards" -ge 141 ] || { echo "ERROR: only $n_shards/141 safetensors shards present (download incomplete)"; exit 1; }

docker rm -f "$CONTAINER" 2>/dev/null || true

# Per-MODE server args (verbatim from the SGLang GLM-5.2 cookbook variants).
# The EAGLE draft head ships inside the FP8 repo, so no --speculative-draft-model.
case "$MODE" in
  high-throughput)
    # Peak aggregate throughput: DP-attention (one attention replica per GPU) +
    # DeepEP MoE all-to-all, NO speculative decoding, explicit running-request cap.
    # CHUNKED_PREFILL bounds the per-step prefill chunk (needed for very long
    # inputs, e.g. 128K, so prefill activations don't OOM); unset = SGLang default.
    MODE_ARGS=(
      --tp 8 --dp 8 --enable-dp-attention
      --moe-a2a-backend deepep
      --mem-fraction-static 0.85
      --max-running-requests "${MAX_RUNNING:-256}"
    )
    [ -n "${CHUNKED_PREFILL:-}" ] && MODE_ARGS+=(--chunked-prefill-size "$CHUNKED_PREFILL")
    ;;
  balanced)
    # Throughput: DP-attention (one attention replica per GPU) + DeepEP MoE
    # all-to-all, light MTP (1-1-2 helps little at high concurrency),
    # chunked prefill, and an explicit running-request cap.
    MODE_ARGS=(
      --tp 8 --dp 8 --enable-dp-attention
      --moe-a2a-backend deepep
      --speculative-algorithm EAGLE
      --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2
      --mem-fraction-static 0.85
      --chunked-prefill-size 32768
      --max-running-requests 256
    )
    ;;
  low-latency)
    # Minimize per-token latency at low concurrency: plain TP8 + heavy MTP (5-1-6).
    MODE_ARGS=(
      --tp 8
      --speculative-algorithm EAGLE
      --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6
      --mem-fraction-static 0.8
    )
    ;;
  *) echo "ERROR: unknown MODE=$MODE (want: high-throughput | balanced | low-latency)"; exit 1;;
esac

# --network host so bench_serving (and any client) reaches it on localhost:$PORT.
# --shm-size / --ipc host are required for SGLang's multi-GPU NCCL shared memory.
docker run -d --name "$CONTAINER" \
  --gpus all --network host --ipc host --shm-size 32g \
  -v "$MODEL_DIR":/model:ro \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path /model \
    "${MODE_ARGS[@]}" \
    --context-length "$CONTEXT_LEN" \
    --host 0.0.0.0 --port "$PORT"

echo "launched $CONTAINER (MODE=$MODE, ctx $CONTEXT_LEN, port $PORT); streaming to $LOG"
docker logs -f "$CONTAINER" > "$LOG" 2>&1 &

# Block until the server reports healthy (weight load of 755 GB from CPFS can
# take several minutes) or the container dies.
echo -n "waiting for /health "
until curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; do
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo ; echo "ERROR: $CONTAINER exited during startup. Last log lines:"; tail -n 40 "$LOG"; exit 1
  fi
  echo -n "."; sleep 5
done
echo ; echo "GLM-5.2-FP8 server is HEALTHY on localhost:$PORT"
