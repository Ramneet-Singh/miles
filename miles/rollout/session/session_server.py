"""Standalone Session Server that proxies through the inference router.

This decouples session/TITO logic from the Miles Router, allowing sessions
to work with the SGLang Rust Router or any other backend.  Inference
requests are proxied through the router (sglang or miles), which handles
load balancing and forwarding to worker engines.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.rollout.session.sessions import setup_session_routes
from miles.utils.logging_utils import configure_logger

logger = logging.getLogger(__name__)


class SessionServer:
    """Lightweight FastAPI server that manages sessions and proxies inference
    requests through the inference router (sglang or miles)."""

    def __init__(self, args, backend_url: str):
        self.backend_url = backend_url
        self.app = FastAPI()

        timeout = getattr(args, "miles_router_timeout", 600.0)
        # Expire idle keepalive connections quickly so a router-side close never
        # leaves a half-open socket to be reused (which hangs the next proxy
        # request until `timeout`). Pairs with the same hygiene on the rollout
        # driver's client (miles.utils.http_utils).
        self.client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=1024,
                max_keepalive_connections=1024,
                keepalive_expiry=float(os.getenv("MILES_HTTP_KEEPALIVE_EXPIRY_SEC", "30")),
            ),
            timeout=httpx.Timeout(timeout),
        )

        # Offload the CPU-bound per-turn TITO tokenization (and large-response
        # JSON parsing) off the single event loop. The HF fast tokenizer releases
        # the GIL during encode, so worker threads tokenize different sessions in
        # true parallel — removing the head-of-line blocking that made the server
        # slow under high concurrency (which in turn triggered client retries).
        # Per-session ordering is still serialized by session.lock.
        n_threads = int(os.getenv("SESSION_SERVER_TOKENIZE_THREADS", str(min(32, (os.cpu_count() or 8)))))
        self.tokenize_pool = ThreadPoolExecutor(max_workers=n_threads, thread_name_prefix="session-tokenize")

        # Release the httpx pool + worker threads when uvicorn shuts down.
        self.app.router.on_shutdown.append(self.client.aclose)
        self.app.router.on_shutdown.append(lambda: self.tokenize_pool.shutdown(wait=False))

        setup_session_routes(self.app, self, args)

    async def do_proxy(
        self,
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        url = f"{self.backend_url}/{path}"
        if request.url.query:
            url = f"{url}?{request.url.query}"

        if body is None:
            body = await request.body()
        if headers is None:
            headers = dict(request.headers)
        headers = {
            k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding", "host")
        }

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
        except httpx.TransportError as exc:
            logger.warning("Proxy transport error for %s %s: %s", request.method, path, exc)
            error_body = json.dumps({"error": f"backend transport error: {type(exc).__name__}: {exc}"}).encode()
            return {
                "request_body": body,
                "response_body": error_body,
                "status_code": 502,
                "headers": {"content-type": "application/json"},
            }
        content = await response.aread()
        return {
            "request_body": body,
            "response_body": content,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }

    def build_proxy_response(self, result: dict) -> Response:
        content = result["response_body"]
        status_code = result["status_code"]
        # Drop wire-level framing headers from upstream so Starlette rebuilds them
        # from the body we actually send: transfer-encoding is hop-by-hop
        headers = {
            k: v
            for k, v in result["headers"].items()
            if k.lower() not in ("content-length", "transfer-encoding", "content-encoding")
        }
        content_type = headers.get("content-type", "")
        try:
            data = json.loads(content)
            return JSONResponse(content=data, status_code=status_code, headers=headers)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)


def run_session_server(args, backend_url: str):
    """Entry point to start the standalone session server as a subprocess."""
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger()
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-session-server")

    server = SessionServer(args, backend_url)
    logger.info(
        "[session-server] Starting on %s:%s, proxying to %s",
        args.session_server_ip,
        args.session_server_port,
        backend_url,
    )
    uvicorn.run(server.app, host=args.session_server_ip, port=args.session_server_port, log_level="info")
