#!/usr/bin/env python3
"""Apply the Nemotron-H R3 fix to the in-container SGLang.

Upstream: sglang PR sgl-project/sglang#27110 (OPEN, not yet in our image's sglang).
The image's sglang builds `self.topk = TopK(...)` in models/nemotron_h.py WITHOUT
`layer_id`, so it defaults to None. With `--use-rollout-routing-replay`, the
routed-experts state-capturer does `buffer[:batch, layer_id, :] = topk_ids` and the
None becomes np.newaxis -> 4-D LHS [tokens,1,layers,topk] -> broadcast RuntimeError
during the SGLang engine warmup. Adding `layer_id=layer_idx` (already in scope in
NemotronHMoE.__init__) is exactly the one line PR #27110 adds.

Applied as an idempotent post-create step by cluster/ansible/containers.yml because
sglang is baked into the image (not our fork). REMOVE this script + the containers.yml
task once #27110 lands in the image's sglang.

Idempotent and fail-loud: re-running is a no-op; if the TopK call can't be located
(sglang layout changed) it errors so we notice instead of silently mis-patching.
"""
import pathlib
import sys

TARGET = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/models/nemotron_h.py"
)
ANCHOR = "self.topk = TopK("
INSERT_LINE = "            layer_id=layer_idx,\n"


def _topk_call_span(src: str) -> tuple[int, int]:
    """Return (open_paren_idx, close_paren_idx) of the `self.topk = TopK(...)` call."""
    start = src.index(ANCHOR) + len(ANCHOR) - 1  # index of '('
    depth = 0
    for k in range(start, len(src)):
        if src[k] == "(":
            depth += 1
        elif src[k] == ")":
            depth -= 1
            if depth == 0:
                return start, k
    raise ValueError("unbalanced parentheses in TopK call")


def main() -> int:
    if not TARGET.exists():
        print(f"[sglang-fix] {TARGET} not found", file=sys.stderr)
        return 1
    src = TARGET.read_text()
    if ANCHOR not in src:
        print("[sglang-fix] 'self.topk = TopK(' not found — sglang layout changed; "
              "re-check whether this fix is still needed", file=sys.stderr)
        return 1
    open_i, close_i = _topk_call_span(src)
    if "layer_id" in src[open_i:close_i]:
        print("[sglang-fix] layer_id already present in TopK(...) — no-op")
        return 0
    anchor_eol = src.index("\n", src.index(ANCHOR)) + 1
    src = src[:anchor_eol] + INSERT_LINE + src[anchor_eol:]
    TARGET.write_text(src)
    print("[sglang-fix] inserted layer_id=layer_idx into TopK(...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
