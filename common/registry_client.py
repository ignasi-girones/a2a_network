"""Client-side helpers for workers to register/deregister with the
orchestrator's AgentRegistry.

Used by every worker agent's __main__.py on startup (and optionally shutdown)
to announce its presence and capabilities to the central registry.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

import httpx

from common.config import settings

logger = logging.getLogger(__name__)


def agent_card_to_dict(
    name: str,
    description: str,
    skills: Iterable[Any],
    streaming: bool = False,
) -> dict[str, Any]:
    """Build a JSON-serializable subset of an AgentCard for the registry.

    We serialize the fields the Planner actually uses to match tasks to workers
    (skills with id/name/description/tags, plus display metadata). Avoiding full
    AgentCard serialization sidesteps protobuf/pydantic edge cases in the SDK.
    """
    skills_out: list[dict[str, Any]] = []
    for s in skills:
        # AgentSkill may be protobuf-backed — read attributes defensively.
        skill_dict = {
            "id": getattr(s, "id", None),
            "name": getattr(s, "name", None),
            "description": getattr(s, "description", ""),
            "tags": list(getattr(s, "tags", []) or []),
        }
        skills_out.append({k: v for k, v in skill_dict.items() if v is not None})

    return {
        "name": name,
        "description": description,
        "streaming": streaming,
        "skills": skills_out,
    }


async def register_self_with_orchestrator(
    agent_id: str,
    url: str,
    card: dict[str, Any],
    *,
    max_retries: int = 30,
    retry_delay: float = 2.0,
) -> bool:
    """Register this agent with the orchestrator's registry.

    Retries until the orchestrator is up (in parallel startup, the orchestrator
    may come online after the worker). Returns True on success, False if all
    retries exhausted.
    """
    registry_url = f"{settings.orchestrator_url()}/registry/register"
    payload = {"agent_id": agent_id, "url": url, "card": card}

    async with httpx.AsyncClient(timeout=5.0) as http:
        for attempt in range(1, max_retries + 1):
            try:
                response = await http.post(registry_url, json=payload)
                if response.status_code == 200:
                    logger.info(
                        "Registered %s with orchestrator registry", agent_id
                    )
                    return True
                logger.warning(
                    "Registration returned %d (attempt %d/%d): %s",
                    response.status_code, attempt, max_retries, response.text,
                )
            except (httpx.ConnectError, httpx.TimeoutException):
                logger.debug(
                    "Orchestrator not reachable yet (attempt %d/%d)",
                    attempt, max_retries,
                )
            except Exception as e:
                logger.warning(
                    "Registration attempt %d/%d failed: %s",
                    attempt, max_retries, e,
                )

            if attempt < max_retries:
                await asyncio.sleep(retry_delay)

    logger.error(
        "Failed to register %s after %d attempts", agent_id, max_retries
    )
    return False


async def deregister_self(agent_id: str) -> None:
    """Best-effort deregistration on shutdown."""
    registry_url = f"{settings.orchestrator_url()}/registry/{agent_id}"
    try:
        async with httpx.AsyncClient(timeout=2.0) as http:
            await http.delete(registry_url)
            logger.info("Deregistered %s from registry", agent_id)
    except Exception as e:
        logger.debug("Deregistration failed (orchestrator down?): %s", e)
