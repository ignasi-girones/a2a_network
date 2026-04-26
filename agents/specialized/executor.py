import json
import logging
import re

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agents.specialized.agent_state import AgentState
from common.config import settings
from common.llm_provider import llm_complete

logger = logging.getLogger(__name__)

MCP_URL = settings.mcp_url()


async def _call_mcp_tool(tool: str, args: dict) -> str | None:
    """Call a tool on the MCP server. Returns result text or None on failure."""
    try:
        async with streamablehttp_client(MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool, args)
                if result.content:
                    return result.content[0].text
    except Exception as e:
        logger.warning("MCP tool call failed (%s): %s", tool, e)
    return None


def _extract_search_query(text: str, max_len: int = 80) -> str:
    """Extract the first meaningful claim from an argument as a search query.

    Strips markdown formatting and skips deliberation section headers
    (AGREEMENTS:, REFINEMENT:, etc.) so the query targets actual content,
    not the structural label.
    """
    # Remove markdown headers, bullets, blockquotes
    clean = re.sub(r'^[#*\->\s]+', '', text, flags=re.MULTILINE).strip()
    # Remove inline markdown bold/italic markers
    clean = re.sub(r'[*_]{1,2}', '', clean).strip()

    # Walk lines, skipping section headers (short ALL-CAPS lines ending in ':')
    content = ""
    for raw_line in clean.split('\n'):
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(':') and len(line) <= 25 and line[:-1].replace(' ', '').isalpha():
            continue
        # Strip an inline section prefix like "AGREEMENTS: actual text..."
        line = re.sub(r'^[A-Z][A-Z\s]{2,30}:\s*', '', line)
        if line:
            content = line
            break
    if not content:
        content = clean

    first_sentence = re.split(r'[.?!]', content)[0].strip()
    query = first_sentence if first_sentence else content
    return query[:max_len]


class SpecializedExecutor(AgentExecutor):
    """Dynamic AgentExecutor that reads its behavior from AgentState.

    Debate flow (human-like: argue first, then search to support/refute):

    1. LLM call  → generate initial argument from assigned perspective
    2. Extract   → derive a focused search query from the first claim
    3. MCP call  → web_search with that meaningful query
    4. LLM call  → refine argument incorporating search evidence
    5. Emit      → single Task(COMPLETED) with optional tool metadata Part
                   followed by the final response Part

    Emitting a single Task (no intermediate TaskStatusUpdateEvent) avoids the
    ClientTaskManager "Task is already set" conflict that occurs when
    status_update events pre-set _current_task before the final Task arrives.
    Tool metadata is embedded as a JSON Part in the Task message and extracted
    by send_and_get_text before relaying to the on_intermediate callback.
    """

    def __init__(self, state: AgentState, model: str):
        self.state = state
        self.model = model

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        user_input = context.get_user_input()
        role = await self.state.get_role()
        logger.info(
            "Agent %s (%s) processing: %s",
            self.state.agent_id, role, user_input[:100],
        )

        system_prompt = await self.state.get_system_prompt()

        # ── Step 1: Generate initial argument (think before searching) ──
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        try:
            initial_argument = await llm_complete(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=600,
            )
        except Exception as e:
            logger.error("LLM call (initial) failed for %s: %s", self.state.agent_id, e)
            initial_argument = f"[Error generating argument: {e}]"

        # ── Step 2: Extract meaningful search query from the argument ──
        search_query = _extract_search_query(initial_argument)
        logger.info("Agent %s searching for: %s", self.state.agent_id, search_query)

        # ── Step 3: Web search to find supporting / refuting evidence ──
        search_results = await _call_mcp_tool("web_search", {"query": search_query})

        # ── Step 4: Refine argument with evidence (if search succeeded) ──
        if search_results:
            refine_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": initial_argument},
                {
                    "role": "user",
                    "content": (
                        f"Web search results for '{search_query}':\n{search_results}\n\n"
                        "Incorporate this evidence where it strengthens your argument "
                        "or helps refute the opposing view. Keep your response concise."
                    ),
                },
            ]
            try:
                final_response = await llm_complete(
                    model=self.model,
                    messages=refine_messages,
                    temperature=0.7,
                    max_tokens=800,
                )
            except Exception as e:
                logger.error(
                    "LLM call (refine) failed for %s: %s", self.state.agent_id, e
                )
                final_response = initial_argument  # fall back to unrefined argument
        else:
            final_response = initial_argument

        # ── Step 5: Emit single Task(COMPLETED) ──
        # When search was used, prepend a JSON metadata Part so the orchestrator
        # can relay a tool_use event to the frontend via the on_intermediate callback.
        # send_and_get_text detects Parts whose text is JSON with a "stage" key and
        # routes them to on_intermediate instead of accumulating them as response text.
        parts: list[Part] = []
        if search_results:
            parts.append(
                Part(
                    text=json.dumps(
                        {
                            "stage": "tool_use",
                            "message": f"{self.state.agent_id.upper()} usó web_search",
                            "data": {
                                "agent": self.state.agent_id,
                                "tool": "web_search",
                                "query": search_query,
                            },
                        },
                        ensure_ascii=False,
                    )
                )
            )
        parts.append(Part(text=final_response))

        await event_queue.enqueue_event(
            Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.TASK_STATE_COMPLETED,
                    message=Message(
                        role=Role.ROLE_AGENT,
                        parts=parts,
                    ),
                ),
            )
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_CANCELED),
            )
        )
