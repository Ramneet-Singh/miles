import asyncio
import functools
import json
import logging
import time

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.session_errors import (
    SessionError,
    SessionNotFoundError,
    TokenizationError,
    UpstreamResponseError,
)
from miles.rollout.session.session_types import GetSessionResponse, SessionRecord
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)

# Multi-MB-per-turn arrays SGLang returns in ``choice.meta_info`` under R3
# (routed_experts) + logprobs. The agent (mini-swe-agent) ignores them but stores
# every response it receives, so accumulated across a long trajectory they
# OOM-kill the agent process (exit 137) and corrupt its multi-GB trajectory.json.
# Training reads them from the session RECORD, never from what the agent gets, so
# we strip them from the agent-facing copy only (the record keeps everything).
_HEAVY_META_KEYS = frozenset(
    {
        "routed_experts",
        "output_token_logprobs",
        "input_token_logprobs",
        "output_top_logprobs",
        "input_top_logprobs",
    }
)


def _strip_heavy_meta(obj):
    """Recursively prune the heavy SGLang extension payloads from a response so the
    agent never receives/stores them: the whole ``sglext`` blob (where SGLang puts
    the multi-MB base64 ``routed_experts`` the agent sees) plus the R3/logprob
    arrays wherever they appear. Returns a pruned copy — the heavy keys are dropped
    without traversing their (huge) values — and never mutates the input, which the
    session record aliases. Training reads routed_experts/logprobs from the record's
    ``meta_info``, not from what the agent gets, so this is agent-side only."""
    if isinstance(obj, dict):
        return {k: _strip_heavy_meta(v) for k, v in obj.items() if k != "sglext" and k not in _HEAVY_META_KEYS}
    if isinstance(obj, list):
        return [_strip_heavy_meta(v) for v in obj]
    return obj


def _record_to_result(record: SessionRecord) -> dict:
    """Rebuild a ``do_proxy``-shaped result from a stored committed record, so a
    replayed turn returns through the same ``build_proxy_response`` path.

    Only the JSON body and status are reconstructed; upstream response headers
    are not (the agent reads the JSON body, and a committed turn is always a 200).
    The heavy R3/logprob meta is stripped — the agent never needs it (see
    ``_strip_heavy_meta``); the record itself is untouched.
    """
    return {
        "response_body": json.dumps(_strip_heavy_meta(record.response)).encode(),
        "status_code": record.status_code,
        "headers": {"content-type": "application/json"},
    }


def _transient_error_result(message: str, status_code: int = 503) -> dict:
    """A ``do_proxy``-shaped transient error, so the agent's client retries (and
    re-syncs) instead of consuming a stale/uncommitted turn."""
    return {
        "response_body": json.dumps({"error": {"message": message, "type": "server_error"}}).encode(),
        "status_code": status_code,
        "headers": {"content-type": "application/json"},
    }


def _context_exceeded_result(max_seq_len: int, prompt_tokens: int, reserved: int) -> dict:
    """An OpenAI ``context_length_exceeded`` 400 so the agent's client raises
    ContextWindowExceededError and ends the trajectory cleanly — instead of
    growing its multi-turn context toward the model's hard limit (which produces
    late context 400s and oversized training samples that OOM the step)."""
    err = {
        "error": {
            "message": (
                f"This model's maximum context length is {max_seq_len} tokens. However, your "
                f"messages resulted in {prompt_tokens} tokens plus {reserved} reserved for the "
                f"completion. Please reduce the length of the messages."
            ),
            "type": "invalid_request_error",
            "param": "messages",
            "code": "context_length_exceeded",
        }
    }
    return {
        "response_body": json.dumps(err).encode(),
        "status_code": 400,
        "headers": {"content-type": "application/json"},
    }


def setup_session_routes(app, backend, args):
    hf_checkpoint = getattr(args, "hf_checkpoint", None)
    if not hf_checkpoint:
        logger.info("[session] Skipping session routes (hf_checkpoint not set).")
        return

    session_server_instance_id = getattr(args, "session_server_instance_id", None)

    tokenizer = load_tokenizer(
        hf_checkpoint, chat_template_path=getattr(args, "chat_template_path", None), trust_remote_code=True
    )

    tito_tokenizer = get_tito_tokenizer(
        tokenizer,
        tokenizer_type=getattr(args, "tito_model", "default"),
        chat_template_kwargs=getattr(args, "apply_chat_template_kwargs", None),
        allowed_append_roles=getattr(args, "tito_allowed_append_roles", None),
    )

    registry = SessionRegistry(args, tokenizer, tito_tokenizer=tito_tokenizer)

    @app.get("/health")
    async def health():
        body = {"status": "ok"}
        if session_server_instance_id is not None:
            body["session_server_instance_id"] = session_server_instance_id
        return body

    # --- DEBUG: track in-flight chat_completions ---
    _inflight_chat = {"count": 0}

    @app.middleware("http")
    async def debug_request_logger(request: Request, call_next):
        client = request.client
        client_info = f"{client.host}:{client.port}" if client else "unknown"
        logger.info(
            f"[session-server] REQUEST ARRIVED: {request.method} {request.url.path} from={client_info} inflight_chat={_inflight_chat['count']}"
        )
        t0 = time.time()
        response = await call_next(request)
        elapsed = time.time() - t0
        logger.info(
            f"[session-server] REQUEST DONE: {request.method} {request.url.path} status={response.status_code} elapsed={elapsed:.3f}s from={client_info}"
        )
        return response

    @app.exception_handler(SessionError)
    async def session_error_handler(request: Request, exc: SessionError):
        return JSONResponse(status_code=exc.status_code, content={"error": str(exc)})

    @app.post("/sessions")
    async def create_session():
        session_id = registry.create_session()
        return {"session_id": session_id}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = registry.get_session(session_id)
        metadata = {}
        try:
            mismatch = registry.compute_session_mismatch(session)
        except TokenizationError:
            logger.exception("Failed to compute tito_session_mismatch for session %s", session_id)
            mismatch = None
        if mismatch is not None:
            metadata["tito_session_mismatch"] = mismatch
        metadata["accumulated_token_ids"] = session.token_ids
        metadata["max_trim_tokens"] = registry.tito_tokenizer.max_trim_tokens
        response = GetSessionResponse(
            session_id=session_id,
            records=session.records,
            metadata=metadata,
        )
        # Serialize the (multi-MB, full-trajectory) records OFF the event loop.
        # FastAPI's default path (jsonable_encoder + json.dumps) runs on the single
        # event loop and blocks all proxying/collection — py-spy showed the loop
        # pinned in model_dump/iterencode here once the per-turn proxy was unblocked.
        # pydantic-core's Rust serializer is far faster; we run it in the tokenize
        # pool, mirroring the per-turn tokenization offload above.
        body = await asyncio.get_running_loop().run_in_executor(
            backend.tokenize_pool, response.model_dump_json
        )
        return Response(content=body, media_type="application/json")

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        session = registry.get_session(session_id)
        if session.closing:
            raise SessionNotFoundError(f"session not found: session_id={session_id}")
        session.closing = True
        logger.debug(
            f"[session-server] DELETE waiting for lock: session={session_id} lock_locked={session.lock.locked()}"
        )
        await session.lock.acquire()
        logger.debug(f"[session-server] DELETE acquired lock: session={session_id}")
        try:
            registry.remove_session(session_id)
        finally:
            session.lock.release()
        return Response(status_code=204)

    @app.post("/sessions/{session_id}/v1/chat/completions")
    async def chat_completions(request: Request, session_id: str):
        """Proxy a chat completion through SGLang with TITO token tracking.

        Flow: prepare pretokenized input_ids (lock held briefly) → inject
        SGLang flags → proxy to backend (NO lock) → validate response →
        update trajectory checkpoint (lock held briefly) → append session record.

        The lock is NOT held during the slow proxy call to avoid blocking
        DELETE/other operations when the agent disconnects mid-request.
        """
        _inflight_chat["count"] += 1
        try:
            session = registry.get_session(session_id)
            if session.closing:
                raise SessionNotFoundError(f"session not found: session_id={session_id}")

            # --- Phase 1: prepare request (lock held briefly) ---
            async with session.lock:
                # Double-check: session may have been marked closing while waiting for lock.
                if session.closing:
                    raise SessionNotFoundError(f"session not found: session_id={session_id}")

                body = await request.body()
                request_body = json.loads(body) if body else {}

                # TITO token tracking requires Miles-owned input_ids plus SGLang
                # output-token metadata:
                #   logprobs=True     → populates meta_info.output_token_logprobs
                #   return_meta_info  → wraps the above in choice.meta_info
                # Both flags are hardcoded (not set default) to prevent agent-side
                # overrides from breaking the token accumulation invariants.
                request_body["logprobs"] = True
                request_body["return_meta_info"] = True
                if getattr(args, "use_rollout_routing_replay", False):
                    request_body["return_routed_experts"] = True
                if getattr(args, "use_rollout_indexer_replay", False):
                    request_body["return_indexer_topk"] = True
                # Must be False so stop-token text is trimmed from assistant
                # message content; token IDs are still taken from logprobs below.
                request_body["no_stop_trim"] = False

                request_messages = request_body.get("messages", [])

                # Idempotent replay: if the agent re-requests a turn the session
                # already committed (its client retried a request we already
                # generated and stored), return the committed response verbatim
                # instead of regenerating. Otherwise the agent keeps an assistant
                # turn the session didn't store, desyncing its history from the
                # stored prefix so every later turn fails the append-only check.
                replay = session.find_committed_replay(request_messages)
                if replay is not None:
                    logger.info("[session-server] idempotent replay for session %s (turn already committed)", session_id)
                    return backend.build_proxy_response(_record_to_result(replay))

                # Offload the CPU-bound tokenization to the worker pool (the fast
                # tokenizer releases the GIL, so different sessions tokenize in
                # parallel). We still hold session.lock across the await, so this
                # session's own turns stay serialized; only cross-session work
                # parallelizes. Mutations inside prepare_pretokenized are thus
                # exclusive to this session.
                prompt_token_ids = await asyncio.get_running_loop().run_in_executor(
                    backend.tokenize_pool,
                    functools.partial(
                        session.prepare_pretokenized,
                        request_messages,
                        tools=request_body.get("tools"),
                        tito_tokenizer=registry.tito_tokenizer,
                    ),
                )
                request_body["input_ids"] = prompt_token_ids
                logger.debug(
                    "Using TITO input_ids: %d tokens",
                    len(prompt_token_ids),
                )

                # Bound the agent's context here: model_info does NOT actually limit
                # mini-swe-agent, so without this the multi-turn context grows toward
                # the model's hard limit (~230K seen) -> late context-length 400s and
                # oversized training samples that OOM the step. End the trajectory
                # cleanly once prompt + reserved completion would exceed max_seq_len.
                _max_seq = int(getattr(args, "max_seq_len", 0) or 0)
                _reserved = int(getattr(args, "rollout_max_response_len", 0) or 0)
                if _max_seq > 0 and len(prompt_token_ids) + _reserved > _max_seq:
                    logger.info(
                        "[session-server] context budget exceeded for %s (%d + %d > %d) -> 400",
                        session_id, len(prompt_token_ids), _reserved, _max_seq,
                    )
                    return backend.build_proxy_response(
                        _context_exceeded_result(_max_seq, len(prompt_token_ids), _reserved)
                    )

                body = json.dumps(request_body).encode()
                expected_num_assistant = session.num_assistant
            # --- lock released here ---

            # --- Phase 2: proxy to SGLang (NO lock held) ---
            result = await backend.do_proxy(request, "v1/chat/completions", body=body)

            # If SGLang returned a non-200 error (e.g. 400 for context too long),
            # pass it through to the agent without recording — the agent can retry
            # or handle the error.
            if result["status_code"] != 200:
                return backend.build_proxy_response(result)

            # Parse off the event loop too: with R3 on, responses carry multi-MB
            # routing payloads whose parse would otherwise block the single loop.
            response = await asyncio.get_running_loop().run_in_executor(
                backend.tokenize_pool, json.loads, result["response_body"]
            )

            choice = response.get("choices", [{}])[0]

            meta_info = choice.get("meta_info")
            if not isinstance(meta_info, dict) or "output_token_logprobs" not in meta_info:
                raise UpstreamResponseError(
                    "meta_info and output_token_logprobs must be in choice (requires logprobs=True)"
                )
            assistant_message = choice.get("message", {})
            if assistant_message.get("content") is None:
                raise UpstreamResponseError(
                    "assistant message content is None, when tool call parser failed SGLang should still return "
                    "an empty content rather than None. Please check your modified SGLang version."
                )

            output_token_logprobs = meta_info["output_token_logprobs"]
            completion_tokens = meta_info["completion_tokens"]

            actual_output_logprobs_len = len(output_token_logprobs)
            if actual_output_logprobs_len != completion_tokens:
                raise UpstreamResponseError(
                    "invalid chat completion response: "
                    f"len(output_token_logprobs)={actual_output_logprobs_len} "
                    f"!= completion_tokens={completion_tokens}. "
                    f"Please check whether you use the correct SGLang branch which has fix the tokenizer batch decode issue."
                )

            completion_token_ids = [t[1] for t in output_token_logprobs]

            # --- Phase 3: update state (lock held briefly) ---
            async with session.lock:
                if session.closing:
                    logger.warning(f"Session {session_id} closed during proxy, skipping state update")
                    return backend.build_proxy_response(result)

                if session.num_assistant != expected_num_assistant:
                    logger.warning(
                        f"Session {session_id} state changed during proxy "
                        f"(expected num_assistant={expected_num_assistant}, "
                        f"got {session.num_assistant}); returning the committed turn"
                    )
                    # A concurrent request committed this turn while we generated.
                    # Return the COMMITTED response, not our now-stale local
                    # generation, so the agent stays in lockstep with the stored
                    # prefix (returning the stale turn is the desync that makes a
                    # later turn fail the append-only check).
                    if expected_num_assistant < len(session.records):
                        return backend.build_proxy_response(
                            _record_to_result(session.records[expected_num_assistant])
                        )
                    # A concurrent rollback truncated past this turn's record, so
                    # there is no committed turn to replay. Return a transient
                    # error (not the stale local generation) so the agent retries
                    # and re-syncs cleanly rather than silently desyncing.
                    return backend.build_proxy_response(
                        _transient_error_result("session state changed concurrently; please retry")
                    )

                session.update_pretokenized_state(
                    request_messages,
                    assistant_message,
                    prompt_token_ids=prompt_token_ids,
                    completion_token_ids=completion_token_ids,
                    max_trim_tokens=registry.tito_tokenizer.max_trim_tokens,
                )

                record = SessionRecord(
                    timestamp=time.time(),
                    method=request.method,
                    path="/v1/chat/completions",
                    status_code=result["status_code"],
                    request=request_body,
                    response=response,
                )
                session.append_record(record)
            # --- lock released here ---

            # Return the response to the agent WITHOUT the heavy R3/logprob meta
            # (it's already committed in full to the record above for training).
            # Re-serializing is cheap now that the multi-MB arrays are gone.
            agent_result = {**result, "response_body": json.dumps(_strip_heavy_meta(response)).encode()}
            return backend.build_proxy_response(agent_result)
        finally:
            _inflight_chat["count"] -= 1

    @app.api_route("/sessions/{session_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def session_proxy(request: Request, session_id: str, path: str):
        result = await backend.do_proxy(request, path)
        return backend.build_proxy_response(result)
