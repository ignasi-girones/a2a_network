"""HTTP routes for the debate persistence layer.

These are NOT A2A protocol endpoints — they are propietary infrastructure
that the *frontend* talks to. Other A2A network agents keep using the
official ``POST /`` JSON-RPC entry point untouched.

Exposed endpoints
-----------------
``POST /debates``
    Body ``{"prompt": "..."}``. Creates a new debate row and kicks off
    ``AgenticOrchestrator.run()`` in a background task. Returns the new
    ``debate_id`` immediately, or **409 Conflict** if another debate is
    already running (only one active at a time, by design).

``GET /debates``
    Paginated list ``{"debates": [...], "count": N}`` ordered most-recent
    first. Powers the sidebar.

``GET /debates/active``
    The single ``running`` debate as a row, or ``null``. The frontend
    calls this on mount to decide whether to auto-resume.

``GET /debates/<id>``
    One debate's metadata.

``GET /debates/<id>/events?since=N``
    All events with ``seq > N`` as a JSON array. One-shot; used by the
    sidebar's "open old debate" path which doesn't need follow.

``GET /debates/<id>/stream?since=N``
    Server-Sent Events. First emits the catch-up (seq > N), then
    tail-follows live events from the in-memory pub/sub. Closes after
    the synthetic ``verdict``/``failed`` event for terminal debates,
    or when the client disconnects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from sse_starlette.sse import EventSourceResponse
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from agents.orchestrator.agent_registry import registry
from agents.orchestrator.agentic_orchestrator import AgenticOrchestrator
from agents.orchestrator.debate_store import (
    AttachmentRow,
    DebateAlreadyActiveError,
    DebateStore,
)
from agents.orchestrator.persisting_progress import PersistingProgressCallback
from agents.orchestrator.plan_executor import ProgressCallback
from agents.orchestrator.worker_spawner import get_spawner
from common.attachments import extract_text
from common.config import settings
from common.telemetry.log_context import current_debate_id

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────


def _store(request: Request) -> DebateStore:
    """Pull the shared DebateStore out of app.state (set in __main__)."""
    store = getattr(request.app.state, "debate_store", None)
    if store is None:
        raise RuntimeError(
            "debate_store not attached to app.state — wire it in __main__"
        )
    return store


def _build_progress_chain(
    store: DebateStore, debate_id: str
) -> ProgressCallback:
    """Build the callback chain for a /debates POST background run.

    Outer to inner:
        PersistingProgressCallback ⊃ MetricsProgressCallback ⊃ NoOp

    Unlike the A2A-path chain (``OrchestratorExecutor.execute``), there's
    no SSE inner: the live stream goes out via the DebateStore pub/sub
    that ``record_event`` already publishes to.
    """
    inner: ProgressCallback = ProgressCallback()  # no-op base
    if settings.telemetry_enabled:
        from common.telemetry.progress_metrics import MetricsProgressCallback
        inner = MetricsProgressCallback(inner, agent_id="orchestrator")
    return PersistingProgressCallback(inner, store, debate_id)


async def _run_debate(
    store: DebateStore,
    debate_id: str,
    prompt: str,
    attachments: list[AttachmentRow] | None = None,
) -> None:
    """Background task: actually run the debate and mark its terminal state."""
    current_debate_id.set(debate_id)
    progress = _build_progress_chain(store, debate_id)

    extra_context: str | None = None
    if attachments:
        texts = [
            f"[{a.filename}]\n{a.extracted_text}"
            for a in attachments
            if a.extracted_text
        ]
        if texts:
            extra_context = "\n\n---\n\n".join(texts)

    try:
        orchestrator = AgenticOrchestrator(
            registry=registry,
            spawner=get_spawner(),
            progress=progress,
        )
        verdict = await orchestrator.run(prompt, extra_context=extra_context)
        await store.mark_completed(debate_id, verdict)
    except Exception as e:
        logger.exception("Debate %s failed: %s", debate_id, e)
        try:
            await store.mark_failed(debate_id, f"{type(e).__name__}: {e}")
        except Exception as mark_err:
            logger.error(
                "Could not mark debate %s as failed: %s", debate_id, mark_err
            )


# ── Routes ─────────────────────────────────────────────────────────────────


async def create_debate(request: Request) -> JSONResponse:
    """POST /debates — kicks off a new debate; rejects if one is already active.

    Accepts both ``application/json`` (legacy) and ``multipart/form-data``
    (with optional file attachments).
    """
    content_type = (request.headers.get("content-type") or "").split(";")[0].strip()
    uploaded_files: list[tuple[str, str, bytes]] = []  # (filename, mime, content)

    if content_type == "multipart/form-data":
        form = await request.form()
        prompt = (form.get("prompt") or "").strip() if isinstance(form.get("prompt"), str) else ""
        for upload in form.getlist("files"):
            if hasattr(upload, "filename"):
                content = await upload.read()
                uploaded_files.append((upload.filename, upload.content_type or "", content))
        await form.close()
    else:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"error": "invalid_json", "message": "body must be JSON"},
                status_code=400,
            )
        prompt = (body.get("prompt") or "").strip() if isinstance(body, dict) else ""

    if not prompt:
        return JSONResponse(
            {"error": "missing_prompt", "message": "prompt is required"},
            status_code=400,
        )

    # ── Validate attachments ──────────────────────────────────────────
    max_bytes = settings.attachments_max_size_mb * 1024 * 1024
    if len(uploaded_files) > settings.attachments_max_files:
        return JSONResponse(
            {
                "error": "too_many_files",
                "message": f"max {settings.attachments_max_files} files allowed",
            },
            status_code=400,
        )
    for fname, fmime, fcontent in uploaded_files:
        if fmime not in settings.attachments_allowed_mime:
            return JSONResponse(
                {
                    "error": "invalid_mime",
                    "message": f"file '{fname}' has unsupported type '{fmime}'",
                },
                status_code=400,
            )
        if len(fcontent) > max_bytes:
            return JSONResponse(
                {
                    "error": "file_too_large",
                    "message": f"file '{fname}' exceeds {settings.attachments_max_size_mb}MB limit",
                },
                status_code=400,
            )

    store = _store(request)
    try:
        debate_id = await store.create_debate(prompt)
    except DebateAlreadyActiveError:
        active = await store.get_active()
        return JSONResponse(
            {
                "error": "debate_active",
                "message": "another debate is already running",
                "active": active.to_dict() if active else None,
            },
            status_code=409,
        )

    # ── Persist attachments & extract text ────────────────────────────
    attachments: list[AttachmentRow] = []
    if uploaded_files:
        upload_dir = Path(settings.attachments_storage_dir) / debate_id
        os.makedirs(upload_dir, exist_ok=True)
        for fname, fmime, fcontent in uploaded_files:
            file_path = upload_dir / fname
            file_path.write_bytes(fcontent)
            extracted = extract_text(file_path, fmime)
            att_id = await store.add_attachment(
                debate_id=debate_id,
                filename=fname,
                mime_type=fmime,
                size_bytes=len(fcontent),
                storage_path=str(file_path),
                extracted_text=extracted,
            )
            attachments.append(
                AttachmentRow(
                    id=att_id,
                    debate_id=debate_id,
                    filename=fname,
                    mime_type=fmime,
                    size_bytes=len(fcontent),
                    storage_path=str(file_path),
                    extracted_text=extracted,
                    created_at="",
                )
            )
        logger.info(
            "Debate %s: saved %d attachment(s) to %s",
            debate_id, len(attachments), upload_dir,
        )

    asyncio.create_task(_run_debate(store, debate_id, prompt, attachments or None))

    return JSONResponse(
        {
            "debate_id": debate_id,
            "status": "running",
            "attachments": len(attachments),
        },
        status_code=201,
    )


async def list_debates(request: Request) -> JSONResponse:
    """GET /debates?limit=&offset= — paginated list, newest first."""
    store = _store(request)
    try:
        limit = max(1, min(int(request.query_params.get("limit", "50")), 500))
    except ValueError:
        limit = 50
    try:
        offset = max(0, int(request.query_params.get("offset", "0")))
    except ValueError:
        offset = 0
    rows = await store.list_debates(limit=limit, offset=offset)
    return JSONResponse(
        {"count": len(rows), "debates": [r.to_dict() for r in rows]}
    )


async def get_active(request: Request) -> JSONResponse:
    """GET /debates/active — the running debate row, or null."""
    store = _store(request)
    row = await store.get_active()
    return JSONResponse(row.to_dict() if row else None)


async def get_debate(request: Request) -> JSONResponse:
    """GET /debates/<id> — metadata."""
    store = _store(request)
    debate_id = request.path_params["debate_id"]
    row = await store.get_debate(debate_id)
    if row is None:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return JSONResponse(row.to_dict())


async def get_events(request: Request) -> JSONResponse:
    """GET /debates/<id>/events?since=N — one-shot replay."""
    store = _store(request)
    debate_id = request.path_params["debate_id"]
    if await store.get_debate(debate_id) is None:
        return JSONResponse({"error": "not_found"}, status_code=404)
    try:
        since = int(request.query_params.get("since", "-1"))
    except ValueError:
        since = -1
    events = await store.get_events(debate_id, since_seq=since)
    return JSONResponse(
        {"count": len(events), "events": [e.to_dict() for e in events]}
    )


async def stream_debate(request: Request) -> EventSourceResponse:
    """GET /debates/<id>/stream?since=N — SSE catch-up + tail-follow.

    Subscribes to the in-memory pub/sub *before* running the catch-up so
    no events fired during the SELECT can slip through unnoticed. After
    catch-up the loop forwards live events, deduping by seq.
    """
    store = _store(request)
    debate_id = request.path_params["debate_id"]
    debate = await store.get_debate(debate_id)
    if debate is None:
        return JSONResponse({"error": "not_found"}, status_code=404)

    try:
        since = int(request.query_params.get("since", "-1"))
    except ValueError:
        since = -1

    async def event_generator():
        # Subscribe FIRST — race-free catch-up: any event written between
        # the SELECT and the suscribe yield would otherwise be missed.
        async with store.subscribe(debate_id) as queue:
            # 1. Catch-up: every event with seq > since.
            catch_up = await store.get_events(debate_id, since_seq=since)
            last_seq = since
            for ev in catch_up:
                yield {"event": "progress", "data": json.dumps(ev.to_dict())}
                last_seq = ev.seq

            # If the debate is already terminal and the synthetic final
            # event has been emitted, we're done.
            current = await store.get_debate(debate_id)
            if current is not None and current.status != "running":
                # Make sure the verdict/failed synthetic event is in the
                # catch-up; if it is, last_seq points at it.
                terminal_events = await store.get_events(
                    debate_id, since_seq=last_seq
                )
                for ev in terminal_events:
                    yield {"event": "progress", "data": json.dumps(ev.to_dict())}
                return

            # 2. Tail-follow until terminal.
            while True:
                if await request.is_disconnected():
                    return
                try:
                    payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Heartbeat / disconnect probe — sse-starlette also has
                    # ping but explicit re-check is cheap and keeps the
                    # generator responsive on slow networks.
                    continue
                if payload["seq"] <= last_seq:
                    # Duplicate (we already emitted it during catch-up).
                    continue
                yield {"event": "progress", "data": json.dumps(payload)}
                last_seq = payload["seq"]
                # Close after the synthetic terminal event.
                if payload["stage"] in ("verdict", "failed"):
                    return

    return EventSourceResponse(event_generator(), ping=20)


async def get_attachments(request: Request) -> JSONResponse:
    """GET /debates/<id>/attachments — list file attachments for a debate."""
    store = _store(request)
    debate_id = request.path_params["debate_id"]
    if await store.get_debate(debate_id) is None:
        return JSONResponse({"error": "not_found"}, status_code=404)
    rows = await store.get_attachments(debate_id)
    return JSONResponse(
        {"count": len(rows), "attachments": [r.to_dict() for r in rows]}
    )


# ── Routes export ──────────────────────────────────────────────────────────

debate_routes = [
    Route("/debates", create_debate, methods=["POST"]),
    Route("/debates", list_debates, methods=["GET"]),
    Route("/debates/active", get_active, methods=["GET"]),
    Route("/debates/{debate_id}", get_debate, methods=["GET"]),
    Route("/debates/{debate_id}/events", get_events, methods=["GET"]),
    Route("/debates/{debate_id}/stream", stream_debate, methods=["GET"]),
    Route("/debates/{debate_id}/attachments", get_attachments, methods=["GET"]),
]
