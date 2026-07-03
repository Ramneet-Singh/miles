"""Output-capped mini-swe-agent.

mini-swe-agent stores each bash command's FULL stdout in its trajectory; the
observation template only truncates the model's *view* (~10k chars). On
data-heavy Terminal-Bench tasks the agent cats large files/datasets, so
``trajectory.json`` balloons to multiple GB and corrupts ("Unterminated string
starting at ... char 2.2e9") -> harbor can't load it -> AgentError / empty
session records (and ~3.6 GB trial dirs). Empirically this was the dominant
cause of the ~18% per-trajectory abort rate that drove ~5x GRPO oversampling,
plus a chunk of the disk bloat and the ApiRateLimitError misclassification
(harbor regex-matching a "rate limit" substring inside a giant output).

mini-swe-agent v2.4.2 exposes no output-cap config, so we subclass harbor's
installed agent and, right after ``uv tool install``, append a tiny monkeypatch
to the environment's ``execute()`` that caps the returned output to head+tail.

Why this is safe: submission detection (``_check_finished``) runs on the FULL
output inside the original ``execute()`` BEFORE our wrapper sees it, so final
answers are never truncated — only intermediate observations are bounded, and
the model already saw only ~10k of them via the observation template.

Selected via ``AgentConfig(import_path="capped_mini_swe_agent:CappedMiniSweAgent")``
in server.py. Cap is ``MSWEA_MAX_OUTPUT_CHARS`` (default 100000).
"""

import base64
import os

from harbor.agents.installed.mini_swe_agent import MiniSweAgent

# Python appended verbatim to the installed environments' source. Kept as a raw
# string: the ``\n`` sequences must reach the file as backslash-n (valid Python
# string escapes when local.py later runs), not as literal newlines here.
_PATCH = r'''

# --- miles output cap (injected by CappedMiniSweAgent) ---
import os as _miles_os
_MILES_MAX_OUT = int(_miles_os.environ.get("MSWEA_MAX_OUTPUT_CHARS", "100000"))


def _miles_cap_output(s):
    if isinstance(s, str) and len(s) > _MILES_MAX_OUT:
        h = _MILES_MAX_OUT // 2
        return s[:h] + ("\n...[%d chars truncated by miles output cap]...\n" % (len(s) - _MILES_MAX_OUT)) + s[-h:]
    return s


try:
    _MilesEnv = LocalEnvironment
except NameError:
    _MilesEnv = DockerEnvironment

_miles_orig_execute = _MilesEnv.execute


def _miles_capped_execute(self, action, cwd="", *, timeout=None):
    out = _miles_orig_execute(self, action, cwd, timeout=timeout)
    if isinstance(out, dict) and "output" in out:
        out["output"] = _miles_cap_output(out["output"])
    return out


_MilesEnv.execute = _miles_capped_execute
'''

# Append _PATCH to the freshly-installed tool's local.py and docker.py. The
# single-quoted heredoc terminator means the shell performs no expansion, so the
# Python (incl. % and \n) lands verbatim. uv installs under
# $HOME/.local/share/uv/tools/mini-swe-agent/.
_PATCH_CMD = (
    '. "$HOME/.local/bin/env"; '
    'for f in $(find "$HOME/.local/share/uv/tools/mini-swe-agent" '
    "-path '*minisweagent/environments/local.py' -o "
    "-path '*minisweagent/environments/docker.py' 2>/dev/null); do "
    'cat >> "$f" <<\'MILES_PATCH_EOF\'\n'
    + _PATCH
    + '\nMILES_PATCH_EOF\n'
    "done"
)


# --- optional SSH reverse-reachability tunnel (Modal backend) ---
# When the task container runs off-host (HARBOR_ENV_TYPE=modal), the agent's
# model calls must reach the session server on node0, which is NOT publicly
# exposed. If MILES_TUNNEL_* is configured on the agent server, install() opens
# an SSH local-forward FROM the task container so localhost:30000 -> node0's
# session server over port 22 (a restricted, port-forwarding-only key). The
# agent then targets 127.0.0.1:30000 (set MILES_ROUTER_EXTERNAL_HOST=127.0.0.1
# on the trainer). No public :30000 is opened. Inert unless MILES_TUNNEL_* set,
# so the docker backend is unaffected.
_TUNNEL_LOCAL_PORT = 30000


def _tunnel_setup() -> tuple[str, dict[str, str]] | None:
    """(command, env) to open the tunnel inside the task container, or None.

    Reads config from the agent server's own environment (install() runs in the
    agent-server process). The key is passed via the exec env (not baked into the
    command) so it isn't the command string; it's a restricted forward-only key.
    """
    key_file = os.getenv("MILES_TUNNEL_KEY_FILE")
    host = os.getenv("MILES_TUNNEL_HOST")
    user = os.getenv("MILES_TUNNEL_USER", "mtunnel")
    target = os.getenv("MILES_TUNNEL_TARGET", "10.0.96.128:30000")
    if not (key_file and host and os.path.isfile(key_file)):
        return None
    with open(key_file, "rb") as f:
        key_b64 = base64.b64encode(f.read()).decode()
    cmd = (
        "set -e; "
        "command -v ssh >/dev/null 2>&1 || { apt-get update -qq && "
        "apt-get install -y -qq openssh-client >/dev/null; }; "
        "umask 077; printf %s \"$MILES_TK\" | base64 -d > /tmp/miles_tunnel_key; "
        "chmod 600 /tmp/miles_tunnel_key; "
        "ssh -f -N -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null "
        "-o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 "
        f"-i /tmp/miles_tunnel_key -L {_TUNNEL_LOCAL_PORT}:{target} {user}@{host}; "
        # block until the forwarded port answers, so the agent never races the tunnel
        f"for i in $(seq 1 20); do (exec 3<>/dev/tcp/127.0.0.1/{_TUNNEL_LOCAL_PORT}) 2>/dev/null "
        "&& { echo 'miles: tunnel up'; exit 0; }; sleep 0.5; done; "
        "echo 'miles: tunnel FAILED to come up' >&2; exit 1"
    )
    return cmd, {"MILES_TK": key_b64}


class CappedMiniSweAgent(MiniSweAgent):
    """MiniSweAgent that bounds per-command captured output (see module docstring)."""

    async def install(self, environment) -> None:
        await super().install(environment)
        await self.exec_as_agent(environment, command=_PATCH_CMD)
        # Bring up the session-server tunnel first if configured (Modal backend),
        # so it's ready before the agent makes any model call in run().
        tunnel = _tunnel_setup()
        if tunnel is not None:
            cmd, env = tunnel
            await self.exec_as_root(environment, command=cmd, env=env)
