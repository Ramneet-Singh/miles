#!/usr/bin/env bash
# stage-swesmith-tasks.sh — regenerate the EXACT 150 SWE-smith Harbor tasks that
# run-super-swesmith-agentic-async.sh trains on, on a fresh cluster, from the repo.
#
# The task set is pinned by swe-smith-py-150.jsonl (committed next to this script):
# each row's metadata.instance_id is a Harbor task-dir basename. This script runs
# the converter over the 4 source repos, then keeps only those 150 dirs.
#
# Requirements:
#   - public-env-calibration checked out at >= commit 96f65a4 (the Bug-Patch
#     broken-state fix — without it, tasks grade `main` and reward is ~1/no signal).
#     https://github.com/Proximal-Labs/public-env-calibration
#   - python3 with `datasets` (the miles image has it; run this inside that env).
#   - Docker access on the rollout host (Harbor pulls the swebench/swesmith.* base
#     images and builds one thin per-task image on first use).
#
# Usage:
#   ./stage-swesmith-tasks.sh <path-to-public-env-calibration> [OUT_DIR=/cpfs01/swe-smith-tasks]
set -euo pipefail

CALIB="${1:?path to a public-env-calibration checkout}"
OUT="${2:-/cpfs01/swe-smith-tasks}"
HERE="$(cd "$(dirname "$0")" && pwd)"
JSONL="$HERE/swe-smith-py-150.jsonl"
[ -f "$JSONL" ] || { echo "missing $JSONL"; exit 1; }

# 1) Convert enough python rows to span the 4 repos in the pinned set (they are
#    contiguous in dataset order within the first ~1200 rows: oauthlib, tenacity,
#    iniconfig, Red-DiscordBot).
python3 "$CALIB/scripts/data/convert_swe_smith_to_harbor.py" \
  --output-dir "$OUT" --languages py --num-per-lang 1200

# 2) Keep ONLY the 150 pinned instance_ids; drop the rest; rewrite the manifest.
python3 - "$OUT/py" "$JSONL" <<'PY'
import json, os, sys, shutil
d, jsonl = sys.argv[1], sys.argv[2]
keep = {json.loads(l)["metadata"]["instance_id"] for l in open(jsonl)}
for name in list(os.listdir(d)):
    p = os.path.join(d, name)
    if os.path.isdir(p) and name not in keep:
        shutil.rmtree(p)
kept = {n for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))}
missing = keep - kept
assert not missing, f"{len(missing)} pinned tasks not produced by the converter: {sorted(missing)[:3]}..."
man = os.path.join(d, "manifest.jsonl")
if os.path.exists(man):
    rows = [json.loads(l) for l in open(man)]
    open(man, "w").write("".join(json.dumps(r) + "\n" for r in rows
                                 if r["task_path"].split("/")[-1] in keep))
print(f"staged {len(kept)} tasks under {d}")
PY

cat <<EOF

Done. Next steps to run the RL job:
  1) Point the agent server at these tasks with the partial (combined) reward:
       ansible-playbook agent-server.yml -e recreate=true \\
         -e harbor_tasks_dir=$OUT/py -e harbor_reward_key=combined
     (or: docker run ... -e HARBOR_TASKS_DIR=$OUT/py -e HARBOR_REWARD_KEY=combined agent_env:latest)
  2) Launch training:
       bash $HERE/run-super-swesmith-agentic-async.sh
EOF
