"""Stage-2: validate the Modal Proxy egress path to node0:30000 (replaces the SSH tunnel).

Two escalating checks, both from a sandbox with the rl-training-sandbox proxy attached:
  1. DIRECT  — sandbox egress -> HOST:PORT, expects TOKEN back from a temp listener on
     node0. Proves: proxy static-egress -> firewall allowlist -> node0:30000.
  2. DIND    — same request, but issued from a `--network host` container inside a DinD
     sandbox (Modal enable_docker + docker:28.3.3-dind + {"bridge":"none"}), mirroring
     harbor's real task-container path (compose services run network_mode: host, so the
     container shares the sandbox netns and its egress rides the proxy).

Run from the ops venv (has modal + ~/.modal.toml). Start the matching token listener on
node0:30000 first (validate_proxy_stage2 does NOT start it):
  python3 -c 'from http.server import BaseHTTPRequestHandler as B,HTTPServer as S;\
T=b"miles-proxy-probe-OK";\
H=type("H",(B,),{"do_GET":lambda s:(s.send_response(200),s.end_headers(),s.wfile.write(T)),\
"log_message":lambda *a:None});S(("0.0.0.0",30000),H).serve_forever()'
  python validate_proxy_stage2.py
"""

import sys

import modal

PROXY = "rl-training-sandbox"
HOST = "47.74.85.155"
PORT = 30000
TOKEN = "miles-proxy-probe-OK"

proxy = modal.Proxy.from_name(PROXY)
app = modal.App.lookup("miles-proxy-probe", create_if_missing=True)


def _run(label: str, sb: modal.Sandbox, script: str) -> bool:
    try:
        p = sb.exec("sh", "-lc", script)
        out, err = p.stdout.read(), p.stderr.read()
        rc = p.wait()
        print(f"[{label}] stdout:\n{out}", flush=True)
        if err.strip():
            print(f"[{label}] stderr:\n{err}", file=sys.stderr, flush=True)
        ok = TOKEN in out
        print(f"[{label}] exit={rc}  >>> {'PASS' if ok else 'CHECK'} <<<", flush=True)
        return ok
    finally:
        sb.terminate()


# ── Test 1: direct sandbox egress via the proxy ──────────────────────────────
print(f"[direct] sandbox (proxy={PROXY}) -> http://{HOST}:{PORT}/", flush=True)
sb1 = modal.Sandbox.create(
    app=app,
    image=modal.Image.debian_slim().apt_install("curl"),
    proxy=proxy,
    timeout=180,
)
direct_ok = _run("direct", sb1, f'curl -s --max-time 10 http://{HOST}:{PORT}/ || echo DIRECT_CURL_FAIL')

# ── Test 2: DinD, host-networked container (harbor's real path) ──────────────
print(f"[dind] DinD sandbox (proxy={PROXY}) -> host-net container -> http://{HOST}:{PORT}/", flush=True)
dind_img = modal.Image.from_registry("docker:28.3.3-dind").dockerfile_commands(
    'RUN mkdir -p /etc/docker && echo \'{"iptables": false, "bridge": "none"}\' > /etc/docker/daemon.json'
)
sb2 = modal.Sandbox.create(
    app=app,
    image=dind_img,
    proxy=proxy,
    block_network=False,
    experimental_options={"enable_docker": True},
    timeout=300,
)
dind_script = (
    "for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 1; done; "
    "docker info >/dev/null 2>&1 || { echo DIND_DOCKERD_FAIL; exit 0; }; "
    f"docker run --rm --network host curlimages/curl:8.11.1 -s --max-time 10 http://{HOST}:{PORT}/ "
    "|| echo DIND_CURL_FAIL"
)
dind_ok = _run("dind", sb2, dind_script)

print(f"\n=== RESULT: direct={'PASS' if direct_ok else 'FAIL'}  dind={'PASS' if dind_ok else 'FAIL'} ===", flush=True)
sys.exit(0 if (direct_ok and dind_ok) else 1)
