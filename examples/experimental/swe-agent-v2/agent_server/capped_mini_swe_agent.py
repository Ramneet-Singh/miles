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


class CappedMiniSweAgent(MiniSweAgent):
    """MiniSweAgent that bounds per-command captured output (see module docstring)."""

    async def install(self, environment) -> None:
        await super().install(environment)
        await self.exec_as_agent(environment, command=_PATCH_CMD)
