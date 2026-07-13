# GLM-5.2-FP8 serving-throughput benchmarks (8×H200/node cluster)

Standalone SGLang serving benchmarks for **GLM-5.2-FP8** (`zai-org/GLM-5.2-FP8`,
744B-A40B MoE, `GlmMoeDsaForCausalLM`, DeepSeek-Sparse-Attention). Runs in a
separate `lmsysorg/sglang:latest` container — no interaction with the miles/Ray
training stack. Weights (~705 GB) mount read-only from CPFS.

## Scripts
| script | what |
|---|---|
| `serve-glm5.2-fp8.sh` | single-node server; `MODE=high-throughput\|balanced\|low-latency` |
| `serve-cluster-glm5.2-fp8.sh` | 8 × **1-node** replicas + router (short-context throughput) |
| `serve-cluster-2node-glm5.2-fp8.sh` | 4 × **2-node** replicas (TP16, RoCE/NCCL) + router (long-context) |
| `bench-glm5.2-fp8.sh` | single-server concurrency sweep |
| `bench-cluster-glm5.2-fp8.sh` | aggregate sweep via the router |
| `bench-cluster-parallel-glm5.2-fp8.sh` | aggregate via independent per-replica clients (no router bottleneck) |

## Results

### Short context — 8192-in / 1024-out, high-throughput mode (DP-attn + DeepEP, no spec)
Per node (8×H200) saturates ~1.3 K out tok/s. **8 × 1-node replicas scale linearly:**

| conc/replica | total conc | aggregate out tok/s | aggregate total tok/s | TTFT p50 | ITL p50 |
|---|---|---|---|---|---|
| 16 | 128 | 4,121 | 37,085 | 3.8 s | 26.4 ms |
| 64 (knee) | 512 | 9,804 | 88,238 | 11.0 s | 34.5 ms |
| 256 (sat) | 2,048 | **11,041** | **99,366** | 99 s | 37.8 ms |

**Cluster peak ≈ 11 K generated tok/s (~99 K total incl. prefill); ~9.8 K at usable latency (conc 64).**

### Long context — 256 K limit, 128 K-in / 16 K-out, 4 × 2-node replicas (TP16)
1-node can't serve this (KV pool 130 K tokens < 144 K/request); 2-node gives 764 K
tokens/replica. KV-bound to ~5 concurrent/replica.

| topology | conc/replica | aggregate out tok/s | aggregate total tok/s | TTFT p50 | E2E p50 |
|---|---|---|---|---|---|
| 4 × 2-node | 4 | **655** | 5,893 | ~30 s | ~400 s |

~17× lower output than short-context: long context is KV-bound (few concurrent) **and**
each decode step attends over 128 K+ context. Memory-bound, GPUs underutilized.

## Gotchas (each cost real time)
- `lmsysorg/sglang:latest` reports version `0.5.14` but **does** register `GlmMoeDsaForCausalLM` (DSA via the shared `deepseek_common` path).
- Run the bench **client** in a throwaway container with the log dir mounted (`-v .../logs:/out`); running it inside the server container loses `--output-file` (no host mount) and trips `set -e`.
- A single client→router path resets connections past ~512 concurrent — measure aggregate via **independent per-replica clients** instead.
- Multi-node replica containers need **both** `--gpus all` (CUDA) **and** `--privileged` + `/dev/infiniband` + `memlock=-1` (RoCE RDMA), plus NCCL env (`NCCL_IB_HCA=mlx5_bond_0..7`, `NCCL_IB_GID_INDEX=3`, `*_SOCKET_IFNAME=eth0`). Interconnect is **RoCE**; use the NCCL path (no DeepEP/NVSHMEM).
- MLA KV can't be TP-sharded (replicated), so per-request context capacity scales only with **total memory (more nodes)**, not TP width.
