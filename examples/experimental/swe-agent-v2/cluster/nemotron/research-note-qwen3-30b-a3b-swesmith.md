# Research note — Qwen3-30B-A3B agentic RL on SWE-smith (Modal, inference-heavy, 512-concurrency)

**2-line summary:** Swapping Nemotron-120B for the much smaller **Qwen3-30B-A3B** (30B total / 3B active MoE, thinking-capable) unlocked three levers the 120B couldn't afford — an **inference-heavy GPU split** (2 train / 6 infer nodes), **off-host Modal task containers**, and **512-way agentic concurrency** — yielding a clean **in-distribution reward climb 0.17 → 0.34 with no drift** (lr 2e-6 + KL 0.001) at **~2.4× the throughput** of the 120B runs. **But the held-out eval is the headline: the swe-smith reward climb did NOT transfer to Terminal-Bench-2** — TB2 stayed flat at the ~2–4% noise floor across all 5 checkpoints, and the best-swe-smith checkpoint was the *worst* on TB2. Training reward alone would have looked like a win.

---

## Setup

- **Model:** Qwen/Qwen3-30B-A3B (text MoE, 48 layers, 128 experts topk-8, GQA=4, native ctx 40960). Chosen after Qwen3.5-35B-A3B turned out to be a multimodal checkpoint that fought the text-only Megatron bridge (see [[deploy-to-live-mount]] history / prior notes).
- **Task:** mini-swe-agent fixing **SWE-smith** synthetic Python bugs (Harbor tasks via the `public-env-calibration` converter, `Bug Patch` checkout — same converter fix as `research-note-swesmith.md`). **500 tasks balanced across 12 repos** (~43/repo: oauthlib, pygments, pdfminer, cog-creators, agronholm, suor, jd, cknd, pudo, pytest-dev, madzak, seatgeek) — broadened from the 150/4-repo Nemotron set to reduce overfit and raise the ceiling.
- **Reward:** partial credit, `HARBOR_REWARD_KEY=combined` (dense).
- **Framework:** miles disaggregated fully-async RL, **2 train nodes (16 GPU) / 6 infer nodes (48 GPU)**, 64× H200. Task containers run **off-host on Modal** (not node0 docker), reached back to the node0 session server via a **Modal Proxy** (`rl-training-sandbox`) with allowlisted static egress IPs — the SSH-tunnel approach was built, validated, then retired once the proxy proved out.

## Config

| Knob | Value |
|---|---|
| num-rollout | 25 rollout iters → **100 optimizer steps** (see decoupling below) |
| rollout-batch-size (prompts/iter) | **32** → 32×16 = 512 trajectories/iter (matched to Modal concurrency) |
| group size (n-samples-per-prompt) | 16 |
| global-batch-size | 128 → **4 optimizer steps per rollout iteration** (512/128) |
| agent concurrency (AGENT_MAX_CONCURRENT) | **512** |
| context limit (max-seq-len) | 40960 (native 40K, no YaRN) |
| response max tokens (per turn) | 16384 (16K; Qwen3 emits `<think>`) |
| parallelism (training, 16 GPU) | TP2 · PP1 · CP4 · EP8 · ETP1 → DP2 |
| inference | rollout-num-gpus 48, **1 GPU/engine (48 TP1 engines)**, sglang-mem-fraction 0.8 |
| max-tokens-per-gpu | 16384 |
| lr / estimator | **2e-6 (constant)** / GRPO, TIS, **KL-coef 0.001** (anti-drift) |
| async | max-weight-staleness 4, update-weights-interval 8, pause-mode abort |
| save-interval | **5 rollout iters** → checkpoints at opt-steps 20/40/60/80/100 (weights-only) |

**Batch decoupling (new vs Nemotron):** fully-async lets `rollout_batch × n_samples` (512) exceed `global_batch` (128), so each rollout iteration feeds **4** optimizer steps. `num-rollout 25` therefore = **100** gradient updates. `save-interval` counts **rollout iterations**, not opt-steps (train_async keys the save on `rollout_id`), so 5 → the 5-point checkpoint curve.

## Throughput achieved (~2.4× Nemotron)

| Metric | Value |
|---|---|
| rollout_time / iter | **~370 s** for **512** trajectories (Nemotron: ~250 s for 128) → ~4× trajectory rate |
| tokens/gpu/s | **~260 avg (peak 324)** vs Nemotron ~47–110 |
| wait_time_ratio | **~0.59** (trainer idle ~60%) vs Nemotron/E2 ~0.80 — still mildly rollout-bound |
| wall-clock | **~2h25m** for 100 optimizer steps |

The inference-heavy 2/6 split + 1-GPU TP1 engines + 512 concurrency is the throughput story. At the earlier 128-concurrency cap the 48 engines were *starved* (~110 tok/gpu/s); 512 filled them (~260). We deliberately **did not** shift more GPUs to inference: at ~0.59 wait-ratio the trainer was ~60% utilized, so halving it (DP2→DP1) would have flipped the pipeline to trainer-bound.

## Reward finding — in-distribution climb, no drift

Per-rollout raw_reward (mean over the 512 trajectories/iter; epoch = ~16 rollouts):

```
rollout: 0    1    2    3    4    5    6    7    8    9  |10   11   12   13   14   15 |16   17   18   19   20   21   22   23   24
reward: 0.18 0.15 0.22 0.15 0.16 0.18 0.18 0.19 0.13 0.15|0.28 0.22 0.23 0.22 0.21 0.21|0.19 0.23 0.33 0.30 0.39 0.37 0.33 0.34 0.29
```

- **Flat ~0.17 for rollouts 0–9, then a clean climb to ~0.22 (rollouts 10–15) and ~0.33–0.39 (rollouts 16–24), peak 0.394.** Base ~0.17 → held ~0.34 (roughly doubled), **no drift** — the anti-drift lr 2e-6 + KL 0.001 held (vs the Nemotron climb-then-drift with lr 3e-6 / KL 0.0).
- **Read learning in EPOCHS, not rollouts or opt-steps.** raw_reward is the mean over 512 trajectories = 32 prompts, so across-rollout noise is dominated by *which* 32-prompt slice (of 500, 12 heterogeneous repos) was sampled. The flat phase lasted until ~1 epoch (rollout ~16), exactly mirroring Nemotron's climb-at-~1-epoch (its 150/8 set → epoch ~19 rollouts, climbed ~rollout 18). Bigger batch → longer epoch → longer flat phase; this is not stalling.
- KL rose 0.005 → 0.034 (small, oscillating). Early on it rose while reward was flat (looked worrying at <1 epoch) but became **productive** once the reward climbed — rising KL + rising reward = healthy learning, not drift.

## Held-out eval — SWE-smith gains did NOT transfer to TB2

Offline agentic eval of all 5 checkpoints on the **89 held-out Terminal-Bench-2 tasks** (`/cpfs01/harbor_tasks/terminal-bench`, decontaminated from the swe-smith train repos):

| checkpoint | opt-step | swe-smith train reward | **held-out TB2** |
|---|---|---|---|
| iter_4  | 20  | ~0.17 | 0.0225 |
| iter_9  | 40  | ~0.22 | **0.0412** (peak) |
| iter_14 | 60  | ~0.28 | 0.0262 |
| iter_19 | 80  | ~0.33 | 0.0337 |
| iter_24 | 100 | ~0.34 | 0.0187 (final, lowest) |

- **Flat within noise (~2–4% = ~2–4 of 89 tasks), no upward trend**, and the *final* checkpoint (best on swe-smith) is the *worst* on TB2 — a mild specialization/overfit signal. Point-to-point diffs are 1–2 tasks = noise; iter_24 also had 13/267 aborted trajectories (counted as 0).
- **The in-distribution reward doubling did not buy held-out capability in 100 steps.** TB2 (terminal tasks) is a very different distribution from swe-smith (Python bug-fix), so limited transfer isn't shocking — but it's the whole reason to hold out an eval: training reward alone would have declared victory.

## Infra lessons (each cost real time; all fixed + committed)

1. **FD exhaustion at 512 concurrency.** The agent-server container's default `nofile` soft limit (1024) is blown by ~512 concurrent agents (each holds several sockets) → `Errno 24 Too many open files` → agents make **zero** model calls → the pipeline **jams while looking alive** (ran 3h stalled at rollout 7, churning 117k empty sessions). Fix: `--ulimit nofile=1048576`. The FD ceiling was *also* silently throttling effective concurrency (~220 → ~379 post-fix). **Monitor rollout-completion cadence, not just `pgrep`.**
2. **Dedicated Modal app.** Harbor's default `__harbor__` app is *shared* across users — our sandboxes were indistinguishable from another team's live run (nearly `modal app stop`'d theirs). A dedicated app (`qwen-rl-swe-smith`, via `EnvironmentConfig.kwargs["app_name"]`) isolates + identifies + safely terminates our containers, and makes per-run billing possible.
3. **`save-interval` counts rollout iterations, not opt-steps** — set it relative to `num-rollout`, not the gradient-update count.
4. **Offline eval via `train.py` needs the full training scaffold** even though none of it runs: `--rollout-batch-size`, `--lr-decay-iters>0`, `--prompt-data`, and `--ref-load` + `--use-kl-loss` together (the startup `update_weights()` reads a `weights_backuper` gated on `with_ref = kl_coef≠0 or use_kl_loss`). Plus a framework fix: `log_eval_rollout_data` crashed on a `None` reward from an aborted trajectory (now coerced to 0). See `run-qwen3-30b-a3b-tb2-eval.sh`.
5. **Modal proxy replaces the SSH tunnel** for sandbox→node0 reachability (validated direct + DinD-host-net egress before switching).

## Cost

**One training run ≈ $158 of Modal** (this run: 500 tasks, 512 concurrency, 25 rollouts / 12,800 trajectories, ~2.5h). Modal bills only the agentic sandboxes' CPU+memory — the H200s are on-prem, so **$0 Modal-GPU**. **Memory dominated (~80%: ~$130 vs ~$29 CPU)** because each sandbox reserves 8 GB (`HARBOR_OVERRIDE_MEMORY_MB=8192`) × up to 512 concurrent — **the #1 cost lever if tasks don't truly need 8 GB.** The 5-checkpoint TB2 eval added ~$32 (~$6.4 each). (Isolated via `modal billing report --show-resources` filtered to the `qwen-rl-swe-smith` app — only possible because of the dedicated-app rename.)

## Conclusion

The redesigned stack works end-to-end and is reproducible from committed launchers (branch `nemotron-async-cluster`): smaller model → inference-heavy split + Modal + 512 concurrency → **~2.4× throughput and a clean, drift-free in-distribution reward climb (0.17→0.34)**. The anti-drift config (lr 2e-6 + KL 0.001) carried over correctly from the Nemotron findings.

**The scientific result is a negative one worth remembering: 100 steps of 500-task SWE-smith RL improved SWE-smith but produced no measurable Terminal-Bench-2 transfer** — and the best-training checkpoint was the weakest held-out. In-distribution reward is not a proxy for held-out capability across distributions this different.

**Recommended next runs:** (a) train materially longer (100 opt-steps is short) to test whether transfer emerges with more steps; (b) mix TB2-like (terminal) tasks into training to close the distribution gap; or (c) if SWE-smith is the target, evaluate on held-out SWE-smith rather than TB2. Cost lever: drop the 8 GB sandbox memory reservation. See `research-note-swesmith.md` (Nemotron/SWE-smith climb-then-drift) and `research-note-tb2.md` (why the dataset, not the optimizer, was the original blocker).
