#!/usr/bin/env bash
# serve-cluster-glm5.2-fp8.sh — serve GLM-5.2-FP8 from the WHOLE 8-node cluster
# as 8 independent single-node replicas behind one SGLang router.
#
# Why replicas, not one 64-GPU instance: the 705 GB FP8 model fits on a single
# 8x H200 node, so the throughput-optimal cluster deployment is data-parallel
# replicas (one per node) load-balanced by a router. Each replica keeps its MoE
# all-to-all (DeepEP) intra-node over NVLink; there is NO cross-node expert
# communication, so aggregate throughput scales ~linearly (~8x single node).
# A single model sharded across all 64 GPUs would add an inter-node all-to-all
# tax and yield LOWER aggregate tok/s (it only helps single-request latency /
# very long contexts).
#
# Topology:
#   node0..node7 : one `sglang.launch_server` replica each, TP8/DP8 high-tput,
#                  :$WORKER_PORT, weights mounted RO from shared CPFS.
#   node0        : one `sglang_router` fronting all 8, :$ROUTER_PORT — this is
#                  the single endpoint you point load at.
#
# Prereqs: passwordless SSH node0->workers (ansible user), weights staged on
# CPFS (shared mount), image pullable on every node.
#
# Usage:
#   bash serve-cluster-glm5.2-fp8.sh          # launch all replicas + router
#   bash serve-cluster-glm5.2-fp8.sh stop     # tear everything down
set -euo pipefail

IMAGE=${IMAGE:-lmsysorg/sglang:latest}
MODEL_DIR=${MODEL_DIR:-/cpfs01/models/GLM-5.2-FP8}
MODE=${MODE:-high-throughput}
CONTEXT_LEN=${CONTEXT_LEN:-32768}
WORKER_PORT=${WORKER_PORT:-30000}
ROUTER_PORT=${ROUTER_PORT:-40000}
ROUTER_POLICY=${ROUTER_POLICY:-round_robin}    # round_robin | cache_aware | power_of_two
WORKER_CONTAINER=glm52-sglang
ROUTER_CONTAINER=glm52-router
NODES=(10.0.96.128 10.0.96.129 10.0.96.130 10.0.96.131 10.0.96.132 10.0.96.133 10.0.96.134 10.0.96.135)
HEAD=${NODES[0]}
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"

# Run a command on a node — locally on the head, over SSH on workers.
on_node() { local ip=$1; shift; if [ "$ip" = "$HEAD" ]; then bash -lc "$*"; else $SSH "$ip" "$*"; fi; }

if [ "${1:-}" = "stop" ]; then
  docker rm -f "$ROUTER_CONTAINER" 2>/dev/null || true
  for ip in "${NODES[@]}"; do echo "stopping replica on $ip"; on_node "$ip" "docker rm -f $WORKER_CONTAINER 2>/dev/null || true"; done
  echo "cluster torn down"; exit 0
fi

# high-throughput server args (matches serve-glm5.2-fp8.sh MODE=high-throughput).
case "$MODE" in
  high-throughput) MODE_ARGS="--tp 8 --dp 8 --enable-dp-attention --moe-a2a-backend deepep --mem-fraction-static 0.85 --max-running-requests ${MAX_RUNNING:-256} ${CHUNKED_PREFILL:+--chunked-prefill-size $CHUNKED_PREFILL}";;
  balanced)        MODE_ARGS="--tp 8 --dp 8 --enable-dp-attention --moe-a2a-backend deepep --speculative-algorithm EAGLE --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 --mem-fraction-static 0.85 --chunked-prefill-size 32768 --max-running-requests 256";;
  low-latency)     MODE_ARGS="--tp 8 --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --mem-fraction-static 0.8";;
  *) echo "ERROR: unknown MODE=$MODE"; exit 1;;
esac

RUN_CMD="docker run -d --name $WORKER_CONTAINER --gpus all --network host --ipc host --shm-size 32g \
  -v $MODEL_DIR:/model:ro $IMAGE \
  python3 -m sglang.launch_server --model-path /model $MODE_ARGS \
  --context-length $CONTEXT_LEN --host 0.0.0.0 --port $WORKER_PORT"

# 1. Pre-pull the image on all nodes in parallel (workers likely don't have it).
echo "===== pulling $IMAGE on all nodes (parallel) ====="
for ip in "${NODES[@]}"; do on_node "$ip" "docker image inspect $IMAGE >/dev/null 2>&1 || docker pull $IMAGE" & done
wait

# 2. Launch one replica per node.
echo "===== launching replicas ====="
for ip in "${NODES[@]}"; do
  on_node "$ip" "docker rm -f $WORKER_CONTAINER 2>/dev/null || true; $RUN_CMD" >/dev/null
  echo "  launched replica on $ip:$WORKER_PORT"
done

# 3. Wait for every replica's /health (parallel weight load from CPFS).
echo "===== waiting for all replicas healthy ====="
for ip in "${NODES[@]}"; do
  echo -n "  $ip "
  until curl -sf "http://$ip:$WORKER_PORT/health" >/dev/null 2>&1; do
    on_node "$ip" "docker ps --format '{{.Names}}'" | grep -q "^${WORKER_CONTAINER}$" \
      || { echo "DIED — last log:"; on_node "$ip" "docker logs --tail 30 $WORKER_CONTAINER"; exit 1; }
    echo -n "."; sleep 5
  done
  echo "OK"
done

# 4. Router on the head, fronting all 8 replicas.
echo "===== launching router ($ROUTER_POLICY) on $HEAD:$ROUTER_PORT ====="
WORKER_URLS=""; for ip in "${NODES[@]}"; do WORKER_URLS="$WORKER_URLS http://$ip:$WORKER_PORT"; done
docker rm -f "$ROUTER_CONTAINER" 2>/dev/null || true
docker run -d --name "$ROUTER_CONTAINER" --network host "$IMAGE" \
  python3 -m sglang_router.launch_router \
    --worker-urls $WORKER_URLS \
    --host 0.0.0.0 --port "$ROUTER_PORT" --policy "$ROUTER_POLICY" >/dev/null
echo -n "  waiting for router "
until curl -sf "http://$HEAD:$ROUTER_PORT/health" >/dev/null 2>&1; do
  docker ps --format '{{.Names}}' | grep -q "^${ROUTER_CONTAINER}$" || { echo "router DIED:"; docker logs --tail 30 "$ROUTER_CONTAINER"; exit 1; }
  echo -n "."; sleep 3
done
echo ; echo "CLUSTER READY — 8 replicas behind router at http://$HEAD:$ROUTER_PORT"
