"""Harbor agent server: the rollout backend for agentic RL.

A thin FastAPI service exposing a single ``POST /run`` endpoint. Each call
runs one Harbor trial — agent (e.g. mini-swe-agent) + task environment +
verifier, all in a per-task Docker container — and returns the verifier
reward plus exit status and metrics. This is the server side of the contract
that ``swe_agent_function.run`` (the ``--custom-agent-function-path``) calls
into; the model the agent talks to is reached purely via the ``base_url`` /
``model`` in each request, so this server is model- and task-agnostic.

It runs as its own process (it drives the host Docker daemon and needs only
harbor + fastapi, not miles or a GPU), built into the ``agent_env`` image via
``Dockerfile.agent_env``.

Request  (POST /run):  the fields ``swe_agent_function.run`` sends —
    base_url, model, sampling_params, max_seq_len, plus the task's own
    metadata (instance_id, agent_name) spread in from the prompt dataset.
Response:  reward, exit_status, eval_report, agent_metrics.

Usage:
    python server.py --port 11000 --max-concurrent 8
"""

import argparse
import asyncio
import logging
import os
import re
import shutil
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, Response
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# A bounded concurrency gate so the server never spawns more simultaneous task
# containers than the host can handle; created in the lifespan so /health can
# report not-ready until it exists.
_semaphore: asyncio.Semaphore | None = None


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _semaphore
    max_concurrent = int(os.getenv("AGENT_MAX_CONCURRENT", "8"))
    _semaphore = asyncio.Semaphore(max_concurrent)
    logger.info("Agent server ready (max_concurrent=%d)", max_concurrent)
    yield


app = FastAPI(title="Harbor agent server", lifespan=_lifespan)


class RunRequest(BaseModel):
    # How the agent reaches the model: base_url is the session server's /v1
    # endpoint, model is "openai/<name>" so litellm routes it.
    base_url: str
    model: str
    sampling_params: dict[str, Any] = {}
    api_key: str = "dummy"

    # Task identity + agent selection arrive spread in from the dataset's
    # per-sample metadata.
    instance_id: str = ""
    agent_name: str = "mini-swe-agent"
    max_seq_len: int | None = None

    # The caller also spreads in session-affinity fields (session_server_id,
    # session_server_instance_id); we don't need them here (base_url already
    # routes to the right session), so accept and ignore unknown fields.
    model_config = {"extra": "allow"}


class RunResponse(BaseModel):
    reward: float = 0.0
    exit_status: str = ""
    eval_report: dict[str, Any] = {}
    agent_metrics: dict[str, Any] = {}


def _get_semaphore() -> asyncio.Semaphore:
    assert _semaphore is not None, "Semaphore not initialized — server not started?"
    return _semaphore


# Harbor exception types mapped to the exit statuses the trainer's filter
# understands. Anything else with an exception is an infra failure (AgentError).
_TIMEOUT_EXCEPTIONS = {"AgentTimeoutError", "VerifierTimeoutError", "EnvironmentStartTimeoutError"}
_OUTPUT_LIMIT_EXCEPTIONS = {"MaxSeqLenExceededError"}

# instance_id indexes a directory under HARBOR_TASKS_DIR; forbid path separators
# so it can't escape that root.
_SAFE_INSTANCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _exit_status(result) -> str:
    """Map a Harbor TrialResult to a trainer-facing exit status."""
    exc = getattr(result, "exception_info", None)
    if exc is not None:
        exc_type = getattr(exc, "exception_type", "")
        if exc_type in _TIMEOUT_EXCEPTIONS:
            return "TimeLimitExceeded"
        if exc_type in _OUTPUT_LIMIT_EXCEPTIONS:
            return "SequenceLengthLimitExceeded"
        return "AgentError"
    if getattr(result, "verifier_result", None) is not None:
        return "Submitted"
    return "Unknown"


def _duration_sec(timing) -> float | None:
    started = getattr(timing, "started_at", None)
    finished = getattr(timing, "finished_at", None)
    if started and finished:
        return (finished - started).total_seconds()
    return None


def _reward(result) -> tuple[float, dict[str, Any]]:
    """Scalar reward + full rewards dict from a Harbor TrialResult.

    A malformed/non-numeric reward is coerced to 0.0 rather than raised: a
    recoverable 0-reward sample must not become a lost (crashing) trajectory.
    """
    vr = getattr(result, "verifier_result", None)
    if vr is None:
        return 0.0, {}
    rewards = getattr(vr, "rewards", None) or {}
    # Which reward field becomes the RL scalar. TB2 verifiers emit "reward";
    # SWE-smith (converted) emits partial fields (f2p_pass_ratio, combined) —
    # set HARBOR_REWARD_KEY=combined for a denser gradient. Falls back to
    # "reward" then the first value so any task type still yields a scalar.
    key = os.getenv("HARBOR_REWARD_KEY", "reward")
    raw = rewards.get(key, rewards.get("reward", next(iter(rewards.values()), 0.0)))
    try:
        reward = float(raw)
    except (TypeError, ValueError):
        logger.warning("Non-numeric reward %r; treating as 0.0", raw)
        reward = 0.0
    return reward, dict(rewards)


def _metrics(result) -> dict[str, Any]:
    """Best-effort agent timing/token metrics; never fails the rollout."""
    metrics: dict[str, Any] = {}
    try:
        ar = getattr(result, "agent_result", None)
        if ar is not None:
            for field in ("n_input_tokens", "n_output_tokens", "cost_usd"):
                val = getattr(ar, field, None)
                if val is not None:
                    metrics[field] = val
            meta = getattr(ar, "metadata", None)
            if isinstance(meta, dict):
                metrics.update(meta)
        for name, key in (("agent_execution", "agent_run_time"), ("verifier", "eval_time")):
            dur = _duration_sec(getattr(result, name, None))
            if dur is not None:
                metrics[key] = dur
    except Exception as e:
        logger.warning("Failed to extract metrics: %s", e, exc_info=True)
    return metrics


def _error(exit_status: str) -> dict[str, Any]:
    return {"reward": 0.0, "exit_status": exit_status, "eval_report": {}, "agent_metrics": {}}


def _resolve_task_path(instance_id: str) -> Path | None:
    """Resolve instance_id to a task dir under HARBOR_TASKS_DIR, or None if invalid."""
    if not instance_id or not _SAFE_INSTANCE_ID.match(instance_id):
        logger.error("Invalid instance_id: %r", instance_id)
        return None
    tasks_dir = Path(os.getenv("HARBOR_TASKS_DIR", "/root/harbor_tasks")).resolve()
    # commonpath (not startswith) so a sibling like "<base>_evil" can't pass,
    # and resolve() so a symlink escaping the root is caught too.
    task_path = (tasks_dir / instance_id).resolve()
    if os.path.commonpath([str(task_path), str(tasks_dir)]) != str(tasks_dir):
        logger.error("Path traversal blocked: %r", instance_id)
        return None
    if not task_path.exists():
        logger.error("Task not found: %s", task_path)
        return None
    return task_path


def _trials_dir() -> Path:
    """The dir Harbor writes per-trial artifacts to.

    Harbor runs inside agent_env but drives the HOST docker daemon over the
    shared socket, so any dir it bind-mounts into a task container must resolve
    to the SAME path on the host. /cpfs01 is mounted identically on both sides,
    so a /cpfs01 path satisfies that (and keeps local NVMe free for the docker
    image/overlay2 store, which can't live on CPFS). Resolve + create eagerly so
    a missing/unmounted volume fails loudly here, not opaquely mid-rollout.
    """
    d = Path(os.getenv("HARBOR_TRIALS_DIR", "/cpfs01/harbor-trials")).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cleanup_trial_dir(trial) -> None:
    """Remove a finished trial's on-disk dir; trajectory tokens live in the
    session server, not here, so this is safe and stops per-rollout accumulation.
    Gated by HARBOR_DELETE_TRIALS so a trial can be kept for debugging."""
    if os.getenv("HARBOR_DELETE_TRIALS", "true").lower() not in ("true", "1", "t"):
        return
    try:
        trial_dir = getattr(getattr(trial, "paths", None), "trial_dir", None)
        if trial_dir and os.path.isdir(trial_dir):
            shutil.rmtree(trial_dir, ignore_errors=True)
    except Exception as e:  # cleanup must never mask the rollout result
        logger.warning("Failed to clean trial dir: %s", e)


def _build_agent(request: RunRequest):
    """Construct the AgentConfig kwargs/env for this request.

    Always advertises a model_info token budget (derived from max_seq_len) — this
    is what bounds the agent's own context. Harbor's max_seq_len kwarg is inert
    and model_info is the only honored input-token limit, so skipping it (as the
    stock path does for a model literally named "model") leaves the agent
    unbounded. We never skip it.

    NOTE (open item, verify at smoke): request.sampling_params carries the
    trainer's rollout sampling config (notably temperature) and is NOT forwarded
    here — the agent/litellm currently controls its own sampling. If smoke shows
    the GRPO samples lack diversity (e.g. mini-swe-agent defaulting to a
    near-deterministic temperature collapses a group to identical trajectories),
    wire sampling_params into the agent's per-call model params once harbor's
    exact knob is confirmed against the installed version.
    """
    from harbor.models.trial.config import AgentConfig

    max_out = int(os.getenv("AGENT_MAX_OUTPUT_TOKENS", "8192"))
    if request.max_seq_len is not None:
        max_in = max(1024, int(request.max_seq_len) - max_out)
    else:
        max_in = int(os.getenv("AGENT_MAX_INPUT_TOKENS", "32768"))
    kwargs: dict[str, Any] = {
        "model_info": {
            "max_input_tokens": max_in,
            "max_output_tokens": max_out,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        }
    }
    if request.max_seq_len is not None:
        kwargs["max_seq_len"] = request.max_seq_len

    # Text-based action mode (bash in ```mswea_bash_command``` blocks) instead of
    # OpenAI tool-calls: Nemotron-3-Super returns a free-form planning JSON rather
    # than a tool call, so tool-call mode FormatErrors every turn (-> TITO desync
    # -> AgentError, 0 trajectories submit). The config is baked into the agent_env
    # image; set AGENT_CONFIG_FILE="" to fall back to default tool-call mode.
    config_file = os.getenv("AGENT_CONFIG_FILE", "/app/mini-textbased.yaml")
    if config_file:
        kwargs["config_file"] = config_file

    # The agent talks to the model via these OpenAI/vLLM-style endpoint vars; the
    # session server behind base_url proxies to the engine and traces each turn.
    env = {
        "OPENAI_API_BASE": request.base_url,
        "OPENAI_API_KEY": request.api_key,
        "HOSTED_VLLM_API_BASE": request.base_url,
        "HOSTED_VLLM_API_KEY": request.api_key,
        "MSWEA_COST_TRACKING": "ignore_errors",
    }
    # Use our output-capped mini-swe-agent subclass (bounds per-command captured
    # output so trajectory.json can't balloon to GBs and corrupt -> AgentError /
    # empty records; see capped_mini_swe_agent.py). Harbor resolves an import_path
    # via AgentFactory.create_agent_from_import_path. Set AGENT_IMPORT_PATH="" to
    # fall back to the trainer-provided agent name (stock, uncapped mini-swe-agent).
    import_path = os.getenv("AGENT_IMPORT_PATH", "capped_mini_swe_agent:CappedMiniSweAgent")
    agent_id = {"import_path": import_path} if import_path else {"name": request.agent_name}
    return AgentConfig(
        **agent_id,
        model_name=request.model,
        env=env,
        kwargs=kwargs,
        # apt + uv tool install at the start of every trial is a network
        # download; under concurrency the slowest setups blow Harbor's 360s
        # default. Raise it so contended setups finish instead of AgentError-ing.
        override_setup_timeout_sec=float(os.getenv("HARBOR_AGENT_SETUP_TIMEOUT_SEC", "900")),
    )


async def _run_trial(request: RunRequest) -> dict[str, Any]:
    try:
        from harbor.models.trial.config import EnvironmentConfig, TaskConfig, TrialConfig
        from harbor.trial.trial import Trial
    except ImportError:
        logger.error("harbor import failed (expected harbor==0.15.0)", exc_info=True)
        return _error("ImportError")

    task_path = _resolve_task_path(request.instance_id)
    if task_path is None:
        return _error("InvalidInstanceId")

    try:
        env_type = os.getenv("HARBOR_ENV_TYPE", "docker")
        # Route our Modal sandboxes into a DEDICATED, descriptively-named app
        # (default qwen-rl-swe-smith) instead of harbor's shared default
        # "__harbor__": the shared app mixes multiple users' runs, so ours can't
        # be told apart or safely bulk-terminated. Own app => isolated, findable
        # on the dashboard, and stoppable without touching anyone else. app_name
        # is a ModalEnvironment ctor param; harbor spreads EnvironmentConfig.kwargs
        # into it (factory **config.kwargs). Modal-only; docker ignores it.
        env_kwargs: dict[str, Any] = {}
        if env_type == "modal":
            env_kwargs["app_name"] = os.getenv("HARBOR_MODAL_APP", "qwen-rl-swe-smith")
        config = TrialConfig(
            trials_dir=_trials_dir(),
            # Cap each agent at a fraction of its task's native timeout: tasks
            # declare long native budgets whose timeout long-tails otherwise
            # dominate wall-clock and trajectory-length stats. Env-overridable.
            agent_timeout_multiplier=float(os.getenv("HARBOR_AGENT_TIMEOUT_MULT", "0.7")),
            task=TaskConfig(path=task_path),
            agent=_build_agent(request),
            environment=EnvironmentConfig(
                # Backend is env-selectable so the same server drives local Docker
                # (default) or a remote pool (HARBOR_ENV_TYPE=modal) to lift the
                # node0 RAM/disk concurrency cap. Any harbor EnvironmentType value
                # works. No force_build needed: our converted tasks set no
                # [environment].docker_image, so harbor takes the from-Dockerfile
                # build path on every backend -> the baked bug-checkout layer is
                # applied (modal would otherwise silently regrade `main`).
                type=env_type,
                kwargs=env_kwargs,
                delete=os.getenv("HARBOR_DELETE_CONTAINERS", "true").lower() in ("true", "1", "t"),
                # Raise the per-container memory cap above the task's native
                # value (TB2 tasks set 2G): the mini-swe-agent setup step peaks
                # over 2G and gets cgroup OOM-killed (exit 137) otherwise. Host
                # has TBs free. Set HARBOR_OVERRIDE_MEMORY_MB=0 for native limits.
                override_memory_mb=int(os.getenv("HARBOR_OVERRIDE_MEMORY_MB", "8192")) or None,
                suppress_override_warnings=True,
            ),
        )

        # harbor 0.15.0: Trial is abstract; the async factory resolves the task
        # and returns the concrete SingleStep/MultiStepTrial. Direct
        # Trial(config=config) raises TypeError.
        trial = await Trial.create(config)
        try:
            result = await trial.run()
            reward, eval_report = _reward(result)
            return {
                "reward": reward,
                "exit_status": _exit_status(result),
                "eval_report": eval_report,
                "agent_metrics": _metrics(result),
            }
        finally:
            _cleanup_trial_dir(trial)
    except Exception as e:
        logger.error("Harbor trial failed: %s\n%s", e, traceback.format_exc())
        return _error(f"Error: {type(e).__name__}")


@app.post("/run")
async def run_instance(request: RunRequest) -> RunResponse:
    logger.info("Running instance: %s", request.instance_id)
    async with _get_semaphore():
        result = await _run_trial(request)
    logger.info(
        "Instance %s finished: exit_status=%s reward=%s",
        request.instance_id, result["exit_status"], result["reward"],
    )
    return RunResponse(**result)


@app.get("/health")
async def health(response: Response):
    # Not-ready until lifespan created the semaphore, so a readiness probe
    # doesn't route /run traffic to a server that can't serve it yet.
    if _semaphore is None:
        response.status_code = 503
        return {"status": "starting"}
    return {"status": "ok"}


def main():
    parser = argparse.ArgumentParser(description="Harbor agent server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=11000)
    parser.add_argument("--max-concurrent", type=int, default=8)
    args = parser.parse_args()

    # Env is the source of truth for a containerized service; the CLI flag is
    # only a fallback when it isn't already set.
    os.environ.setdefault("AGENT_MAX_CONCURRENT", str(args.max_concurrent))
    os.environ.setdefault("MSWEA_API_KEY", "dummy")
    os.environ.setdefault("HOSTED_VLLM_API_KEY", "dummy")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
