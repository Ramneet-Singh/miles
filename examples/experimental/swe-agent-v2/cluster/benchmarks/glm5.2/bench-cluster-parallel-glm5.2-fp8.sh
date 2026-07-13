#!/usr/bin/env bash
# bench-cluster-parallel-glm5.2-fp8.sh — measure the 8-node cluster's AGGREGATE
# serving ceiling by driving each replica with its OWN local client in parallel
# and summing. This avoids the single-router / single-client connection
# bottleneck (a lone bench client saturates its socket pool past ~512 concurrent
# connections through one router, producing spurious ConnectionResetErrors).
#
# Because the replicas are fully independent (no cross-node comm), the true
# cluster capacity IS the sum of per-replica throughput at saturation. Each node
# runs bench_serving against its own localhost:$PORT replica.
#
# Usage:
#   bash bench-cluster-parallel-glm5.2-fp8.sh
#   CONC=64 NP=256 bash bench-cluster-parallel-glm5.2-fp8.sh
set -euo pipefail

IMAGE=${IMAGE:-lmsysorg/sglang:latest}
MODEL_DIR=${MODEL_DIR:-/cpfs01/models/GLM-5.2-FP8}
PORT=${PORT:-30000}
ISL=${ISL:-8192}
OSL=${OSL:-1024}
CONC=${CONC:-256}                 # per-replica concurrency (256 = single-node saturation)
NP=${NP:-512}                     # per-replica num-prompts
WARMUP=${WARMUP:-16}
STAMP=${STAMP:-parallel}
# REPLICA_NODES overrides the endpoint list (space-separated IPs) — e.g. the 4
# rank0 nodes of a 2-node-replica cluster. Defaults to all 8 (1-node replicas).
if [ -n "${REPLICA_NODES:-}" ]; then read -ra NODES <<< "$REPLICA_NODES"; else
  NODES=(10.0.96.128 10.0.96.129 10.0.96.130 10.0.96.131 10.0.96.132 10.0.96.133 10.0.96.134 10.0.96.135); fi
HEAD=${NODES[0]}
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"

echo "===== driving all ${#NODES[@]} replicas in parallel (conc=$CONC, np=$NP/node, $ISL/$OSL) ====="
for ip in "${NODES[@]}"; do
  OUT=/cpfs01/logs/glm5.2-${STAMP}-${ip}.jsonl
  CMD="docker run --rm --network host -v $MODEL_DIR:/model:ro -v /cpfs01/logs:/out $IMAGE \
    python3 -m sglang.bench_serving --backend sglang --host localhost --port $PORT \
    --model /model --tokenizer /model --dataset-name random \
    --random-input-len $ISL --random-output-len $OSL --random-range-ratio 1 \
    --num-prompts $NP --max-concurrency $CONC --warmup-requests $WARMUP --flush-cache \
    --output-file /out/$(basename "$OUT")"
  if [ "$ip" = "$HEAD" ]; then bash -lc "$CMD" >/dev/null 2>&1 & else $SSH "$ip" "$CMD" >/dev/null 2>&1 & fi
  echo "  launched client -> $ip"
done
echo "waiting for all clients..."; wait; echo "all clients done"

echo ; echo "===== PER-REPLICA + AGGREGATE (conc=$CONC/replica, $ISL/$OSL) ====="
python3 - "$STAMP" "$ISL" "$OSL" "${NODES[@]}" <<'PY'
import json,os,sys
stamp=sys.argv[1]; isl=sys.argv[2]; osl=sys.argv[3]; nodes=sys.argv[4:]
tot_out=tot_all=0; n_done=0
print(f"{'replica':>15} {'done':>6} {'out_tok/s':>10} {'tot_tok/s':>10} {'TTFT_p50':>9}")
for ip in nodes:
    p=f"/cpfs01/logs/glm5.2-{stamp}-{ip}.jsonl"
    if not os.path.exists(p): print(f"{ip:>15}   MISSING"); continue
    r=[json.loads(l) for l in open(p) if l.strip()][-1]
    o=r.get('output_throughput',0); t=r.get('total_throughput',(r.get('input_throughput',0)+o))
    tot_out+=o; tot_all+=t; n_done+=r.get('completed',0)
    print(f"{ip:>15} {r.get('completed',0):>6} {o:>10.0f} {t:>10.0f} {r.get('median_ttft_ms',0)/1000:>8.1f}s")
print("-"*56)
print(f"{'AGGREGATE':>15} {n_done:>6} {tot_out:>10.0f} {tot_all:>10.0f}")
print(f"\n>>> Cluster serves ~{tot_out:,.0f} output tok/s  (~{tot_all:,.0f} total tok/s incl. prefill) at {isl}/{osl} <<<")
PY
