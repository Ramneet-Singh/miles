# Research note — Nemotron-3-Super-120B agentic RL on Terminal-Bench-2

**2-line summary:** Reward stayed **flat at ~0.15–0.19 across every config** (more steps, 3× lr, 2× group size) — **no hill-climb** — because the 89 TB2 tasks are the *hard eval benchmark* and the model gets ~zero learnable signal (all_one≈0.015, all_zero≈0.55 → >half of each GRPO group is zero-advantage). Best throughput config was **8×16 / CP4** at **~2065 s/step (~34 min), ~42 tok/gpu/s** (the earlier 8×8 was ~900 s/step); the bottleneck was *signal density / task difficulty*, not the RL setup.

---

## Setup

- **Model:** NVIDIA-Nemotron-3-Super-120B-A12B-BF16 (hybrid Mamba+Attention MoE, 88 layers, 512 experts top-k 22).
- **Task:** mini-swe-agent (text-based bash actions) solving **Terminal-Bench-2** (89 curated *hard eval* tasks) via the Harbor agent server; GRPO on the verifier reward.
- **Framework:** miles disaggregated fully-async RL (`train_async.py`) — 4 train nodes / 4 infer nodes, 64× H200. Rust SGLang router, `--pause-generation-mode abort`, TIS, bridge weight-sync.
- **Reward:** binary Harbor verifier (`reward` field): task solved = 1 else 0.

## Config (final 8×16 run; earlier runs were 8×8)

| Knob | Value |
|---|---|
| num-rollout (steps) | 50 |
| rollout-batch-size (prompts/step) | 8 |
| group size (n-samples-per-prompt) | 16 |
| global-batch-size | 128 |
| agent concurrency (AGENT_MAX_CONCURRENT) | 16 |
| context limit (max-seq-len) | 32768 (32K) |
| response max tokens (per turn, rollout-max-response-len) | 8192 (8K) |
| parallelism (training, 32 GPU) | TP4 · PP2 · CP4 · EP8 · ETP1 |
| inference | rollout-num-gpus 32, 8 GPU/engine (4 engines), sglang-mem-fraction 0.8 |
| max-tokens-per-gpu | 8192 |
| lr / estimator | 3e-6 (constant) / GRPO, TIS, KL-coef 0.0 |
| async | max-weight-staleness 4, update-weights-interval 8 |

Configs swept: **8×8 lr 1e-6 (50 steps)**, **8×8 lr 3e-6 (~25 steps)**, **8×16 lr 3e-6 (50 steps)**.

## Throughput achieved

| Config | rollout_time/step | tokens/gpu/s |
|---|---|---|
| **8×16 / CP4** (128 in-flight) | **~2065 s (~34 min)** | **~42** |
| 8×8 / CP4 (64 in-flight) | ~900 s (~15 min) | ~40 |

- Rollout is the ~10× bottleneck (trainer computes ~50 s, waits ~600–2000 s on rollout).
- 8×16 ≈ 2× the 8×8 wall-clock (agent concurrency capped at 16 → more waves), engine-bound.
- CP4 (16k tokens sharded to 8k/rank) required to fit the 32K-context MoE training step without OOM; CP2 OOM'd on long trajectories.

## Reward finding

**Flat across all three configs** — smoothed raw_reward ~0.15–0.19, first→second-half delta ≈ 0 every time:
- 8×8 lr 1e-6, 50 steps → flat ~0.20
- 8×8 lr 3e-6, ~25 steps → Δ +0.02 (indistinguishable from 1e-6)
- 8×16 lr 3e-6, 50 steps → Δ −0.01 (flat)

`grad_norm` was healthy throughout (0.1–0.6, no NaN) — the loop, infra, and weight-sync are sound. The problem was upstream of the optimizer.

## Conclusion

**The bottleneck is signal density / task difficulty, not the RL hyperparameters.** Three independent levers (more steps, 3× lr, 2× group size) all landed on the same flat ~0.17. The mechanism was consistent:
- `all_one ≈ 0.015` — the model essentially never fully solves a TB2 task.
- `all_zero ≈ 0.55` — >half of every GRPO group returns all-same reward → zero advantage → no gradient.

**We were RL-training on the eval benchmark itself** — 89 uniformly-hard tasks with no difficulty gradient. GRPO cannot climb a hill the rollouts never expose. This directly motivated the pivot to a *difficulty-graded training corpus* (SWE-smith) — see `research-note-swesmith.md`, where the same config produces a real reward climb.
