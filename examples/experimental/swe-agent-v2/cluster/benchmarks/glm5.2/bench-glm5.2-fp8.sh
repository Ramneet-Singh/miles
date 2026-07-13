#!/usr/bin/env bash
# bench-glm5.2-fp8.sh — drive the running GLM-5.2-FP8 SGLang server with
# `sglang.bench_serving` and sweep concurrency to map the latency/throughput
# curve. Assumes serve-glm5.2-fp8.sh has the server HEALTHY on localhost:$PORT.
#
# We run the client INSIDE the server container (it already has sglang + the
# tokenizer at /model, and reaches the server on localhost via --network host).
#
# Two things are measured:
#   1. The cookbook's low-latency point verbatim (isl 8192 / osl 1024, conc 1)
#      — reproduces the published operating point.
#   2. A concurrency sweep at the same token profile — finds peak serving
#      throughput (output tok/s) and shows where latency (TTFT/ITL) degrades.
#
# Results: one JSON line per run -> $OUT_JSONL, summarized to a table at the end.
#
# Usage:
#   bash bench-glm5.2-fp8.sh
#   CONCURRENCY="1 8 64 256" ISL=4096 OSL=512 bash bench-glm5.2-fp8.sh
set -euo pipefail

CONTAINER=${CONTAINER:-glm52-sglang}
IMAGE=${IMAGE:-lmsysorg/sglang:latest}
MODEL_DIR=${MODEL_DIR:-/cpfs01/models/GLM-5.2-FP8}   # host path -> mounted for the tokenizer
PORT=${PORT:-30000}
ISL=${ISL:-8192}                       # random input length (cookbook: 8192)
OSL=${OSL:-1024}                       # random output length (cookbook: 1024)
CONCURRENCY=${CONCURRENCY:-"1 4 16 64 128 256"}
PROMPTS_PER_CONC=${PROMPTS_PER_CONC:-6}  # num-prompts = conc * this (steady state), floored at 32
WARMUP=${WARMUP:-16}
STAMP=${STAMP:-run}
OUT_JSONL=${OUT_JSONL:-/cpfs01/logs/glm5.2-bench-${STAMP}.jsonl}

docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$" || { echo "ERROR: $CONTAINER not running"; exit 1; }
curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "ERROR: server not healthy on :$PORT"; exit 1; }

# Fresh results file on the HOST. The client runs in a throwaway container with
# the log dir mounted at /out and the tokenizer at /model, hitting the server on
# localhost via --network host. (Running the client *inside* the server
# container fails: it has no host log mount, so --output-file is lost.)
: > "$OUT_JSONL"
LOGDIR=$(dirname "$OUT_JSONL"); BASE=$(basename "$OUT_JSONL")

for CONC in $CONCURRENCY; do
  NP=$(( CONC * PROMPTS_PER_CONC )); [ "$NP" -lt 32 ] && NP=32
  echo "===== conc=$CONC  num-prompts=$NP  isl=$ISL osl=$OSL ====="
  docker run --rm --network host -v "$MODEL_DIR":/model:ro -v "$LOGDIR":/out "$IMAGE" \
    python3 -m sglang.bench_serving \
      --backend sglang \
      --host localhost --port "$PORT" \
      --model /model --tokenizer /model \
      --dataset-name random \
      --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1 \
      --num-prompts "$NP" --max-concurrency "$CONC" \
      --warmup-requests "$WARMUP" --flush-cache \
      --output-file /out/"$BASE" || echo "  [warn] conc=$CONC failed, continuing"
done

echo ; echo "===== SUMMARY (isl=$ISL osl=$OSL) ====="
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
