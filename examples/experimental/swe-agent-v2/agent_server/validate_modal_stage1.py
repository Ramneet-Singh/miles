"""Stage-1 Modal backend validation — no model / no session server.

Drives ONE SWE-smith task through harbor's *modal* environment with the native
`nop` agent (which does nothing), then lets the verifier grade the untouched
state. This exercises the whole Modal path — auth, `Image.from_dockerfile`
build (with our baked bug-checkout RUN layer), Docker Hub pull, DinD/compose
start, exec, and the verifier — without needing the model loop.

Expected PASS signal: the trial COMPLETES on Modal and the verifier reports
`f2p_pass_ratio ~= 0` (the FAIL_TO_PASS tests fail because the bug is present).
If f2p_pass_ratio were ~1, Modal skipped our bug-checkout layer (grading `main`)
— the silent regrade-main regression we must catch here.

Run inside the modal-enabled agent_env image with ~/.modal.toml mounted:
  docker run --rm -v ~/.modal.toml:/root/.modal.toml:ro -v /cpfs01:/cpfs01 \
    -e HARBOR_TASKS_DIR=/cpfs01/swe-smith-tasks/py -e HARBOR_REWARD_KEY=combined \
    agent_env:latest python /cpfs01/validate_modal_stage1.py [instance_id]
"""

import asyncio
import os
import sys
import time
from pathlib import Path


async def main() -> int:
    from harbor.models.trial.config import (
        AgentConfig,
        EnvironmentConfig,
        TaskConfig,
        TrialConfig,
    )
    from harbor.trial.trial import Trial

    tasks_dir = Path(os.getenv("HARBOR_TASKS_DIR", "/cpfs01/swe-smith-tasks/py")).resolve()
    reward_key = os.getenv("HARBOR_REWARD_KEY", "combined")

    # Pick the instance from argv, else the first task dir.
    if len(sys.argv) > 1:
        instance_id = sys.argv[1]
    else:
        instance_id = sorted(p.name for p in tasks_dir.iterdir() if p.is_dir())[0]
    task_path = tasks_dir / instance_id
    print(f"[stage1] task={instance_id}\n[stage1] path={task_path}", flush=True)
    if not task_path.exists():
        print(f"[stage1] FAIL: task path does not exist", flush=True)
        return 1

    config = TrialConfig(
        trials_dir=Path(os.getenv("HARBOR_TRIALS_DIR", "/cpfs01/harbor-trials")),
        task=TaskConfig(path=task_path),
        agent=AgentConfig(name="nop"),
        environment=EnvironmentConfig(
            type="modal",
            delete=True,
            override_memory_mb=8192,
            suppress_override_warnings=True,
        ),
    )

    print("[stage1] creating trial on Modal (first build pulls the swesmith image + applies bug-checkout)...", flush=True)
    t0 = time.monotonic()
    trial = await Trial.create(config)
    result = await trial.run()
    dt = time.monotonic() - t0

    exc = getattr(result, "exception_info", None)
    vr = getattr(result, "verifier_result", None)
    rewards = dict(getattr(vr, "rewards", None) or {}) if vr is not None else {}

    print(f"\n[stage1] ===== RESULT ({dt:.0f}s) =====", flush=True)
    print(f"[stage1] exception_info : {getattr(exc, 'exception_type', None) if exc else None}", flush=True)
    print(f"[stage1] verifier ran   : {vr is not None}", flush=True)
    print(f"[stage1] rewards        : {rewards}", flush=True)

    f2p = rewards.get("f2p_pass_ratio")
    combined = rewards.get(reward_key)

    ok_completed = vr is not None and exc is None
    ok_broken = f2p is not None and float(f2p) < 0.5  # bug present => F2P should FAIL
    verdict = "PASS" if (ok_completed and ok_broken) else "CHECK"
    print(f"\n[stage1] verifier ran + no infra error : {ok_completed}", flush=True)
    print(f"[stage1] f2p_pass_ratio ~0 (broken baked): {ok_broken} (f2p={f2p})", flush=True)
    print(f"[stage1] combined ({reward_key})          : {combined}", flush=True)
    print(f"[stage1] >>> {verdict} <<<", flush=True)
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
