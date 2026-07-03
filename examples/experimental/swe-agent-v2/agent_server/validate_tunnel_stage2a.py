"""Stage-2a: validate the SSH tunnel from a real Modal sandbox to node0.

Creates a PLAIN Modal sandbox (no harbor, no GPU), opens the same SSH local-
forward the agent will use (`ssh -N -L 30000:<target> mtunnel@<node0>` with the
restricted key), and curls 127.0.0.1:30000 through it. Expects to reach a temp
HTTP listener you run on node0:30000. Proves: Modal egress can reach node0:22,
the restricted key authenticates + forwards, and node0:30000 answers over the
tunnel — the one new risk in the SSH approach, isolated from the model stack.

Run from the ops venv (has modal + ~/.modal.toml):
  # on node0, first:  python3 -m http.server 30000 --bind 0.0.0.0 &
  python validate_tunnel_stage2a.py
"""

import base64
import sys

import modal

KEY_FILE = "/home/user/miles-tunnel-key/id_tunnel"
HOST = "47.74.85.155"
USER = "mtunnel"
TARGET = "10.0.96.128:30000"

key_b64 = base64.b64encode(open(KEY_FILE, "rb").read()).decode()

app = modal.App.lookup("miles-tunnel-probe", create_if_missing=True)
img = modal.Image.debian_slim().apt_install("openssh-client", "curl")

script = (
    "set -e; umask 077; "
    f"printf %s {key_b64} | base64 -d > /tmp/k; chmod 600 /tmp/k; "
    "ssh -f -N -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null "
    "-o ExitOnForwardFailure=yes -o ConnectTimeout=10 "
    f"-i /tmp/k -L 30000:{TARGET} {USER}@{HOST}; "
    "for i in $(seq 1 20); do (exec 3<>/dev/tcp/127.0.0.1/30000) 2>/dev/null && break; sleep 0.5; done; "
    'curl -s -o /dev/null -w "TUNNEL_HTTP=%{http_code}\\n" --max-time 5 http://127.0.0.1:30000/ '
    "|| echo TUNNEL_CURL_FAIL"
)

print(f"[stage2a] creating plain Modal sandbox; tunneling to {USER}@{HOST} -> {TARGET}", flush=True)
sb = modal.Sandbox.create(app=app, image=img, timeout=180)
try:
    p = sb.exec("bash", "-lc", script)
    out = p.stdout.read()
    err = p.stderr.read()
    rc = p.wait()
    print("[stage2a] stdout:\n" + out, flush=True)
    if err.strip():
        print("[stage2a] stderr:\n" + err, file=sys.stderr, flush=True)
    ok = "TUNNEL_HTTP=200" in out
    print(f"[stage2a] exit={rc}  >>> {'PASS' if ok else 'CHECK'} <<<", flush=True)
finally:
    sb.terminate()
