#!/usr/bin/env bash
# serve-cluster-2node-glm5.2-fp8.sh — serve GLM-5.2-FP8 from the 8-node cluster
# as 4 replicas, each spanning a PAIR of nodes (16 GPUs, TP16), behind one router.
#
# Why 2-node replicas for long context: at 256K context the FP8 weights (88 GB/GPU
# on a single node) leave only ~12 GB/GPU for KV -> ~130K-token pool, which is
# SMALLER than one 128K-in/16K-out request (~144K tokens). Spreading a replica
# over 2 nodes halves weight memory to 44 GB/GPU, freeing ~60 GB/GPU for KV
# (~680K tokens) so the workload fits with concurrency headroom.
#
# Interconnect is RoCE. We use the NCCL data path (proven by training's inter-node
# expert parallelism), NOT DeepEP/NVSHMEM (IBGDA-over-RoCE is unproven here), so
# this config is plain TP16 with EP MoE over NCCL — no --enable-dp-attention,
# no --moe-a2a-backend deepep. Each container gets the cluster's proven RoCE
# setup: --privileged, /dev/infiniband, memlock unlimited, mlx5_bond HCAs, GID 3.
#
# Topology (pairs; rank0 of each hosts the HTTP endpoint on :$WORKER_PORT):
#   replica0: 10.0.96.128(r0) + 10.0.96.129(r1)
#   replica1: 10.0.96.130(r0) + 10.0.96.131(r1)
#   replica2: 10.0.96.132(r0) + 10.0.96.133(r1)
#   replica3: 10.0.96.134(r0) + 10.0.96.135(r1)
#   router on 10.0.96.128:$ROUTER_PORT fronts the 4 rank0 endpoints.
#
# Usage:
#   bash serve-cluster-2node-glm5.2-fp8.sh          # launch 4x2-node replicas + router
#   bash serve-cluster-2node-glm5.2-fp8.sh stop
set -euo pipefail

IMAGE=${IMAGE:-lmsysorg/sglang:latest}
MODEL_DIR=${MODEL_DIR:-/cpfs01/models/GLM-5.2-FP8}
CONTEXT_LEN=${CONTEXT_LEN:-262144}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-16384}
MAX_RUNNING=${MAX_RUNNING:-8}
MEM_FRAC=${MEM_FRAC:-0.85}
WORKER_PORT=${WORKER_PORT:-30000}
DIST_PORT=${DIST_PORT:-20000}
ROUTER_PORT=${ROUTER_PORT:-40000}
ROUTER_POLICY=${ROUTER_POLICY:-round_robin}
WORKER_CONTAINER=glm52-sglang
ROUTER_CONTAINER=glm52-router
# Each pair: "rank0_ip rank1_ip". rank0 exposes the API.
PAIRS=("10.0.96.128 10.0.96.129" "10.0.96.130 10.0.96.131" "10.0.96.132 10.0.96.133" "10.0.96.134 10.0.96.135")
ROUTER_HOST=10.0.96.128
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no"
on_node() { local ip=$1; shift; if [ "$ip" = "$ROUTER_HOST" ]; then bash -lc "$*"; else $SSH "$ip" "$*"; fi; }

ALL_NODES=(10.0.96.128 10.0.96.129 10.0.96.130 10.0.96.131 10.0.96.132 10.0.96.133 10.0.96.134 10.0.96.135)

if [ "${1:-}" = "stop" ]; then
  docker rm -f "$ROUTER_CONTAINER" 2>/dev/null || true
  for ip in "${ALL_NODES[@]}"; do echo "stopping $ip"; on_node "$ip" "docker rm -f $WORKER_CONTAINER 2>/dev/null || true"; done
  echo "2-node cluster torn down"; exit 0
fi

# Proven RoCE container setup + NCCL env (mirrors the miles training container).
# --gpus all attaches the NVIDIA runtime (CUDA); --privileged + /dev/infiniband
# (via privileged) give RDMA access for RoCE. Both are required.
ROCE_FLAGS="--gpus all --privileged --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  -v $MODEL_DIR:/model:ro \
  -e NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7 \
  -e NCCL_IB_GID_INDEX=3 -e NCCL_SOCKET_IFNAME=eth0 -e GLOO_SOCKET_IFNAME=eth0"

# TP16 across the pair, EP MoE over NCCL (no dp-attention / no deepep).
SGLANG_ARGS="--model-path /model --tp 16 --mem-fraction-static $MEM_FRAC \
  --context-length $CONTEXT_LEN --chunked-prefill-size $CHUNKED_PREFILL \
  --max-running-requests $MAX_RUNNING --host 0.0.0.0 --port $WORKER_PORT"

echo "===== launching 4 x 2-node replicas (TP16, ctx $CONTEXT_LEN, RoCE/NCCL) ====="
for pair in "${PAIRS[@]}"; do
  set -- $pair; R0=$1; R1=$2
  DIST="$R0:$DIST_PORT"
  on_node "$R0" "docker rm -f $WORKER_CONTAINER 2>/dev/null || true; docker run -d --name $WORKER_CONTAINER $ROCE_FLAGS $IMAGE python3 -m sglang.launch_server $SGLANG_ARGS --nnodes 2 --node-rank 0 --dist-init-addr $DIST" >/dev/null
  on_node "$R1" "docker rm -f $WORKER_CONTAINER 2>/dev/null || true; docker run -d --name $WORKER_CONTAINER $ROCE_FLAGS $IMAGE python3 -m sglang.launch_server $SGLANG_ARGS --nnodes 2 --node-rank 1 --dist-init-addr $DIST" >/dev/null
  echo "  replica $R0(r0)+$R1(r1) launching, endpoint $R0:$WORKER_PORT"
done

echo "===== waiting for replica endpoints healthy ====="
for pair in "${PAIRS[@]}"; do
  set -- $pair; R0=$1; R1=$2
  echo -n "  $R0 "
  until curl -sf "http://$R0:$WORKER_PORT/health" >/dev/null 2>&1; do
    on_node "$R0" "docker ps --format '{{.Names}}'" | grep -q "^${WORKER_CONTAINER}$" \
      || { echo "r0 DIED:"; on_node "$R0" "docker logs --tail 40 $WORKER_CONTAINER"; exit 1; }
    on_node "$R1" "docker ps --format '{{.Names}}'" | grep -q "^${WORKER_CONTAINER}$" \
      || { echo "r1 DIED:"; on_node "$R1" "docker logs --tail 40 $WORKER_CONTAINER"; exit 1; }
    echo -n "."; sleep 5
  done
  echo "OK"
done

echo "===== launching router ($ROUTER_POLICY) on $ROUTER_HOST:$ROUTER_PORT ====="
WORKER_URLS=""; for pair in "${PAIRS[@]}"; do set -- $pair; WORKER_URLS="$WORKER_URLS http://$1:$WORKER_PORT"; done
docker rm -f "$ROUTER_CONTAINER" 2>/dev/null || true
docker run -d --name "$ROUTER_CONTAINER" --network host "$IMAGE" \
  python3 -m sglang_router.launch_router --worker-urls $WORKER_URLS \
    --host 0.0.0.0 --port "$ROUTER_PORT" --policy "$ROUTER_POLICY" >/dev/null
echo -n "  waiting for router "
until curl -sf "http://$ROUTER_HOST:$ROUTER_PORT/health" >/dev/null 2>&1; do
  docker ps --format '{{.Names}}' | grep -q "^${ROUTER_CONTAINER}$" || { echo "router DIED:"; docker logs --tail 30 "$ROUTER_CONTAINER"; exit 1; }
  echo -n "."; sleep 3
done
echo ; echo "CLUSTER READY — 4 x 2-node replicas behind router at http://$ROUTER_HOST:$ROUTER_PORT"
