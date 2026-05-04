"""Function-wrapping hooks for LLM and MCP tool instrumentation.

Applied once at process startup via ``install_telemetry()``. The original
functions (``llm_complete``, ``llm_stream``, ``_call_mcp_tool``) are
wrapped in-place — they never import ``prometheus_client``.
"""

from __future__ import annotations

import functools
import logging
import sys
import time
from typing import Any

from common.telemetry.metrics import (
    LLM_CALLS,
    LLM_DURATION,
    LLM_TOKENS,
    MCP_CALLS,
    MCP_DURATION,
)

logger = logging.getLogger(__name__)


def install_llm_hooks(agent_id: str) -> None:
    """Wrap ``common.llm_provider.llm_complete`` with timing/counting."""
    import common.llm_provider as llm_mod

    original_complete = llm_mod.llm_complete

    @functools.wraps(original_complete)
    async def _instrumented_complete(
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> str:
        start = time.perf_counter()
        status = "ok"
        try:
            result = await original_complete(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            return result
        except Exception:
            status = "error"
            raise
        finally:
            elapsed = time.perf_counter() - start
            LLM_CALLS.labels(agent_id=agent_id, model=model, status=status).inc()
            LLM_DURATION.labels(agent_id=agent_id, model=model).observe(elapsed)
            # Read token usage exposed by llm_complete via ContextVar
            usage = llm_mod._last_usage.get(None)
            if usage:
                LLM_TOKENS.labels(
                    agent_id=agent_id, model=model, direction="prompt"
                ).inc(usage["prompt_tokens"])
                LLM_TOKENS.labels(
                    agent_id=agent_id, model=model, direction="completion"
                ).inc(usage["completion_tokens"])
                llm_mod._last_usage.set(None)

    # Replace module-level function
    llm_mod.llm_complete = _instrumented_complete

    # Patch all modules that already did `from common.llm_provider import llm_complete`
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is llm_mod:
            continue
        if getattr(mod, "llm_complete", None) is original_complete:
            mod.llm_complete = _instrumented_complete

    # Also wrap llm_stream if available
    original_stream = getattr(llm_mod, "llm_stream", None)
    if original_stream is not None:

        @functools.wraps(original_stream)
        async def _instrumented_stream(
            model: str,
            messages: list[dict[str, str]],
            temperature: float = 0.7,
            max_tokens: int = 2048,
        ):
            start = time.perf_counter()
            status = "ok"
            try:
                async for chunk in original_stream(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                ):
                    yield chunk
            except Exception:
                status = "error"
                raise
            finally:
                elapsed = time.perf_counter() - start
                LLM_CALLS.labels(agent_id=agent_id, model=model, status=status).inc()
                LLM_DURATION.labels(agent_id=agent_id, model=model).observe(elapsed)

        llm_mod.llm_stream = _instrumented_stream

        for mod_name, mod in list(sys.modules.items()):
            if mod is None or mod is llm_mod:
                continue
            if getattr(mod, "llm_stream", None) is original_stream:
                mod.llm_stream = _instrumented_stream


def install_mcp_hooks(agent_id: str) -> None:
    """Wrap ``agents.specialized.executor._call_mcp_tool`` with timing.

    This is a best-effort hook: if the module hasn't been imported yet
    (e.g. on non-specialized agents), we silently skip.
    """
    try:
        import agents.specialized.executor as spec_mod
    except ImportError:
        return

    original_call: Any = getattr(spec_mod, "_call_mcp_tool", None)
    if original_call is None:
        return

    # Substrings that MCP tools use to signal errors in their return value
    # (they catch exceptions and return friendly error strings).
    _ERROR_MARKERS = ("Error", "failed:", "no article found", "no results")

    @functools.wraps(original_call)
    async def _instrumented_mcp(
        tool: str, args: dict, *, whitelist: list[str]
    ) -> str | None:
        start = time.perf_counter()
        status = "ok"
        try:
            result = await original_call(tool, args, whitelist=whitelist)
            if result is None or (
                isinstance(result, str)
                and any(marker in result for marker in _ERROR_MARKERS)
            ):
                status = "error"
            return result
        except Exception:
            status = "error"
            raise
        finally:
            elapsed = time.perf_counter() - start
            MCP_CALLS.labels(agent_id=agent_id, tool=tool, status=status).inc()
            MCP_DURATION.labels(agent_id=agent_id, tool=tool).observe(elapsed)

    spec_mod._call_mcp_tool = _instrumented_mcp

    # Patch any module that imported _call_mcp_tool directly
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod is spec_mod:
            continue
        if getattr(mod, "_call_mcp_tool", None) is original_call:
            mod._call_mcp_tool = _instrumented_mcp
