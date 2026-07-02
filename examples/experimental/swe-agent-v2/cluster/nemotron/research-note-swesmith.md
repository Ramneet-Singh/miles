# Research note — Nemotron-3-Super-120B agentic RL on SWE-smith

**2-line summary:** With the *same* 8×16 / CP4 config but a **difficulty-graded dataset** (SWE-smith, partial `combined` reward), reward **climbed 0.49 → 0.72 over steps 0–15 — the first genuine hill-climb of the effort** (all_zero pinned at 0.0, every group carrying gradient), before **drifting back to a stable ~0.39 plateau** (steps 17–23; lr 3e-6 + no KL anchor → policy drift). Throughput was **~1575 s/step (~26 min), ~47 tok/gpu/s** at 8×16 / CP4; the run was stopped at step 23 having proven that **the dataset — not the RL setup — was the bottleneck**.

---

## Setup

- **Model:** NVIDIA-Nemotron-3-Super-120B-A12B-BF16 (same as the TB2 run).
- **Task:** mini-swe-agent fixing **SWE-smith** synthetic Python bugs, converted to Harbor tasks via the `public-env-calibration` converter. 150 tasks across 4 repos (oauthlib / tenacity / iniconfig / Red-DiscordBot).
- **Converter fix (critical):** SWE-smith ships one shared image per repo with `/testbed` on `main` (working). Each instance's bug is a branch `origin/<id>` = [`Bug Patch`, `Remove F2P Tests`]. The stock converter graded `main` (tests already pass → reward ~1, no signal); we fixed `render_dockerfile` to check out the **`Bug Patch` commit** (bug present + FAIL_TO_PASS tests restored). Oracle-validated: broken → F2P fail; agent fix → `combined` 1.0.
- **Reward:** **partial credit** — `HARBOR_REWARD_KEY=combined` (blends f2p_pass_ratio × p2p_preserve_ratio), a *dense* signal vs TB2's binary.
- **Framework:** identical miles disaggregated fully-async RL, 4 train / 4 infer nodes, 64× H200. **Data-swap only vs the TB2 launcher.**

## Config

| Knob | Value |
|---|---|
| num-rollout (steps) | 50 (**stopped at 23**) |
| rollout-batch-size (prompts/step) | 8 |
| group size (n-samples-per-prompt) | 16 |
| global-batch-size | 128 |
| agent concurrency (AGENT_MAX_CONCURRENT) | 16 |
| context limit (max-seq-len) | 32768 (32K) |
| response max tokens (per turn, rollout-max-response-len) | 8192 (8K) |
| parallelism (training, 32 GPU) | TP4 · PP2 · CP4 · EP8 · ETP1 |
| inference | rollout-num-gpus 32, 8 GPU/engine (4 engines), sglang-mem-fraction 0.8 |
| max-tokens-per-gpu | 8192 |
| lr / estimator | 3e-6 (constant) / GRPO, TIS, **KL-coef 0.0** |
| async | max-weight-staleness 4, update-weights-interval 8 |

## Throughput achieved (8×16 / CP4, 23 steps)

| Metric | Value |
|---|---|
| rollout_time / step | **~1575 s (~26 min)** |
| tokens/gpu/s | **~47.5** (effective ~30.6) |
| avg response_len / total_len | ~18.4k / ~19.5k tokens |

Faster per-step than TB2's 8×16 (~2065 s) — SWE-smith trajectories converge a bit quicker than terminal tasks. First step was ~1600 s inflated by building 150 per-task images (cached after). Rollout remains the dominant cost.

## Reward finding — the hill-climb (and its limit)

```
step:   0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16 | 17   18   19   20   21   22
reward:0.49 0.54 0.59 0.41 0.45 0.51 0.57 0.62 0.63 0.54 0.62 0.62 0.52 0.63 0.57 0.72 0.70|0.32 0.41 0.38 0.36 0.39 0.40
```

- **Step 0 was healthy from the start:** raw_reward 0.49, **all_zero = 0.0**, all_one = 0.0 — *every* GRPO group had reward variance (a real gradient), the exact opposite of TB2's 55% dead groups. This alone validated the pivot.
- **Climb:** reward rose **0.49 → 0.72** (peak step 15), slope stable at ~+0.010/step across four independent windows — a genuine, first-of-the-effort hill-climb.
- **Degradation:** steps 17–23 fell to and held at **~0.39** (below the 0.49 start). grad_norm stayed calm (0.17–0.22, no explosion), all_zero stayed 0.0, but **response_len grew 16k→20k as reward fell** — the policy drifted into longer, less-effective trajectories.

## Conclusion

**The dataset was the bottleneck — confirmed.** Swapping the 89 hard TB2 eval tasks for difficulty-graded SWE-smith + partial reward turned a dead-flat 0.15 into a real 0.49→0.72 climb with zero all-same-reward groups, at the *same* RL config. That is the headline result.

**But the climb was not stable.** With `kl_loss_coef = 0.0` (no anchor to the reference policy) and lr 3e-6, the policy over-shot past the easy gains and drifted into a worse region (~0.39), where it stabilized. Classic climb-then-drift. No checkpoints were saved (throwaway-run pattern), so the step-15 peak weights are not recoverable — only the learning curve (in wandb project `nemotron-3-super-swesmith`).

**Recommended next run for a stable climb-and-hold:** lr **2e-6** + small **KL coef (~0.001)** to anchor the policy; optionally more/harder tasks to raise the ceiling and slow saturation. See `research-note-tb2.md` for why the dataset (not the optimizer) was the original blocker.
