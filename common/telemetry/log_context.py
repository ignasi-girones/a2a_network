"""Correlate log lines with the active debate via a ContextVar.

A `logging.Filter` reads `current_debate_id` and prefixes the line with
``[debate=<id>]`` when the var is set. Promtail extracts that prefix
into a Loki label so the Grafana logs dashboard can filter by debate.

Lifecycle (orchestrator process)
--------------------------------
1. ``install_debate_id_filter()`` once at startup, after
   ``logging.basicConfig`` — attaches the filter to the root logger.
2. When ``POST /debates`` kicks off ``_run_debate`` as an asyncio task,
   the task calls ``current_debate_id.set(debate_id)`` on entry.
3. Every ``logger.info(...)`` inside that task (and any task it spawns
   via ``asyncio.create_task`` / ``run_in_executor``) inherits the
   ContextVar value via asyncio's context propagation, so they all get
   the ``[debate=<id>]`` prefix without any further plumbing.
4. Lines emitted from unrelated tasks (e.g. the SSE stream handler for
   a *different* HTTP request) don't see the var → no prefix.

The filter is non-destructive: it injects the prefix into the
``message`` attribute used by the default formatter, but leaves
``record.msg`` and ``record.args`` alone. So structured log readers
(if any) still see the original template.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

current_debate_id: ContextVar[str | None] = ContextVar(
    "a2a_debate_id", default=None
)


_INSTALLED_SENTINEL = "_a2a_debate_id_factory_installed"


def install_debate_id_filter() -> None:
    """Wrap the global LogRecord factory so every record gets the prefix.

    Why the factory and not a logging.Filter:
      - Filters on a *logger* don't apply to records its children create;
        only the originating logger's filters do.
      - Filters on a *handler* only apply to that one handler. Uvicorn's
        ``uvicorn.access`` logger has ``propagate=False`` and owns its own
        StreamHandler, so it never sees filters attached to root.
      - ``logging.setLogRecordFactory`` runs ONCE per record creation,
        before any logger/handler dispatch — covers every line in the
        process regardless of which named logger emitted it.

    Idempotent via a sentinel attribute on the installed factory so test
    suites and re-imports don't stack wrappers.
    """
    base = logging.getLogRecordFactory()
    if getattr(base, _INSTALLED_SENTINEL, False):
        return

    def factory(*args, **kwargs):
        record = base(*args, **kwargs)
        debate_id = current_debate_id.get()
        # Only wrap real string templates — leave dict/object payloads
        # untouched so structured-log consumers aren't corrupted.
        if debate_id and isinstance(record.msg, str):
            record.msg = f"[debate={debate_id}] {record.msg}"
        return record

    setattr(factory, _INSTALLED_SENTINEL, True)
    logging.setLogRecordFactory(factory)
