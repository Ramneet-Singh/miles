#!/usr/bin/env bash
# bench-cluster-glm5.2-fp8.sh — measure AGGREGATE serving throughput of the
# 8-replica GLM-5.2-FP8 deployment through the router (serve-cluster-*.sh).
#
# This answers "how many GLM-5.2 tokens/s can this 8-node cluster serve people".
# We push a concurrency sweep at the router endpoint and read total output tok/s.
# Client runs in a throwaway image container (--network host, tokenizer mounted).
#
# Usage:
#   bash bench-cluster-glm5.2-fp8.sh
#   CONCURRENCY="64 512 2048" ISL=4096 OSL=512 bash bench-cluster-glm5.2-fp8.sh
set -euo pipefail

IMAGE=${IMAGE:-lmsysorg/sglang:latest}
MODEL_DIR=${MODEL_DIR:-/cpfs01/models/GLM-5.2-FP8}
HEAD=${HEAD:-10.0.96.128}
ROUTER_PORT=${ROUTER_PORT:-40000}
ISL=${ISL:-8192}
OSL=${OSL:-1024}
# 8 replicas x 256 max-running = 2048 aggregate ceiling; sweep up to it.
CONCURRENCY=${CONCURRENCY:-"8 64 256 512 1024 2048"}
PROMPTS_PER_CONC=${PROMPTS_PER_CONC:-4}
WARMUP=${WARMUP:-64}
STAMP=${STAMP:-cluster}
OUT_JSONL=${OUT_JSONL:-/cpfs01/logs/glm5.2-bench-${STAMP}.jsonl}

curl -sf "http://$HEAD:$ROUTER_PORT/health" >/dev/null || { echo "ERROR: router not healthy at $HEAD:$ROUTER_PORT"; exit 1; }
: > "$OUT_JSONL"
LOGDIR=$(dirname "$OUT_JSONL"); BASE=$(basename "$OUT_JSONL")

for CONC in $CONCURRENCY; do
  NP=$(( CONC * PROMPTS_PER_CONC )); [ "$NP" -lt 64 ] && NP=64
  echo "===== [cluster] conc=$CONC num-prompts=$NP isl=$ISL osl=$OSL ====="
  # Client in a throwaway container: tokenizer at /model, host log dir at /out.
  docker run --rm --network host -v "$MODEL_DIR":/model:ro -v "$LOGDIR":/out "$IMAGE" \
    python3 -m sglang.bench_serving \
      --backend sglang \
      --host "$HEAD" --port "$ROUTER_PORT" \
      --model /model --tokenizer /model \
      --dataset-name random \
      --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1 \
      --num-prompts "$NP" --max-concurrency "$CONC" \
      --warmup-requests "$WARMUP" --flush-cache \
      --output-file /out/"$BASE" || echo "  [warn] conc=$CONC failed, continuing"
done

echo ; echo "===== AGGREGATE SUMMARY (8 replicas, isl=$ISL osl=$OSL) ====="
python3 - "$OUT_JSONL" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
def g(r,*ks):
    for k in ks:
        if k in r: return r[k]
    return None
hdr = ("conc","done","in_tok/s","out_tok/s","tot_tok/s","TTFT_p50","TTFT_p99","ITL_p50","e2e_p50(s)")
print("{:>5} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>8} {:>10}".format(*hdr))
for r in rows:
    print("{:>5} {:>5} {:>9.0f} {:>9.0f} {:>9.0f} {:>9.0f} {:>9.0f} {:>8.1f} {:>10.2f}".format(
        g(r,"max_concurrency","concurrency") or 0,
        g(r,"completed") or 0,
        g(r,"input_throughput","total_input_throughput") or 0,
        g(r,"output_throughput") or 0,
        (g(r,"total_throughput") or ((g(r,"input_throughput") or 0)+(g(r,"output_throughput") or 0))),
        g(r,"median_ttft_ms") or 0,
        g(r,"p99_ttft_ms") or 0,
        g(r,"median_itl_ms") or 0,
        (g(r,"median_e2e_latency_ms") or 0)/1000.0,
    ))
PY
echo "(raw: $OUT_JSONL)"
