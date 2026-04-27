"""SpecializedExecutor — Phase 3 dialectic-role-aware worker executor.

Flow per dispatched subtask:

1. Read the persona set by /internal/configure.
2. Render the persona's system_prompt_template against (topic, claim,
   peers_outputs) extracted from the orchestrator's prompt.
3. LLM call → initial argument tailored to the role.
4. If the persona has search tools whitelisted, derive a query from the
   first claim and call the allowed search tool. The whitelist is enforced
   in `_call_mcp_tool` — calling an unauthorised tool raises.
5. LLM call → refine using evidence (only if a search ran).
6. Emit a single Task(COMPLETED) with optional tool-use metadata Part +
   final response Part. Same single-Task pattern as before to avoid the
   "Task is already set" conflict in the SDK.

The role_id and stratagem id (if any) are embedded in the tool-use metadata
so the orchestrator's progress relay surfaces them in the frontend graph.
"""

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

from agents.orchestrator.persona_catalog import render_system_prompt
from agents.specialized.agent_state import AgentState
from agents.specialized.belief_updater import BeliefDelta, update_belief
from common.config import settings
from common.llm_provider import llm_complete
from common.models import BeliefState, PersonaContract

logger = logging.getLogger(__name__)

MCP_URL = settings.mcp_url()


class ToolNotAllowedError(RuntimeError):
    """Raised when a persona tries to call a tool outside its whitelist."""


async def _call_mcp_tool(
    tool: str, args: dict, *, whitelist: list[str]
) -> str | None:
    """Invoke an MCP tool, gated by the persona's whitelist.

    Returning None on failure (network/MCP error) keeps the executor
    resilient: the agent continues with its un-refined argument. But a
    whitelist violation is a *configuration bug*, not a runtime hiccup,
    so it raises loudly.
    """
    if tool not in whitelist:
        raise ToolNotAllowedError(
            f"Tool {tool!r} not in persona whitelist {whitelist!r}"
        )
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
    """Pull the first substantive sentence out of an argument."""
    clean = re.sub(r"^[#*\->\s]+", "", text, flags=re.MULTILINE).strip()
    clean = re.sub(r"[*_]{1,2}", "", clean).strip()
    first_sentence = re.split(r"[.?!\n]", clean)[0].strip()
    query = first_sentence if first_sentence else clean
    return query[:max_len]


def _split_topic_and_peers(user_input: str) -> tuple[str, str]:
    """Split the orchestrator-built prompt into (topic, peers_outputs).

    Phase 3 format:
        Goal: <topic>
        Task: <description>
        ...
        Context from previous steps:
        [t1]
        <output>

    Phase 4 round-aware format (handled here too — the marker just changes):
        Goal: <topic>
        Task: <description>
        Round: R / N
        Your previous position:
        <text>
        Discussion ledger so far:
        <ledger view>
    """
    topic_match = re.search(r"^Goal:\s*(.+?)$", user_input, flags=re.MULTILINE)
    topic = topic_match.group(1).strip() if topic_match else user_input[:200]

    # Try Phase 4 marker first, fall back to Phase 3.
    for marker in ("Discussion ledger so far:", "Context from previous steps:"):
        if marker in user_input:
            return topic, user_input.split(marker, 1)[1].strip()
    return topic, ""


def _extract_round_context(user_input: str) -> dict[str, object] | None:
    """Phase 4: pull round/ledger/your-position fields out of the prompt.

    Returns ``None`` if the prompt is in Phase 3 single-shot format (no
    ``Round:`` line). Otherwise returns a dict with the four placeholder
    values the persona template needs in round-aware mode.
    """
    round_match = re.search(
        r"^Round:\s*(\d+)\s*/\s*(\d+)\s*$",
        user_input,
        flags=re.MULTILINE,
    )
    if round_match is None:
        return None
    round_number = int(round_match.group(1))
    max_rounds = int(round_match.group(2))

    # Self position: between "Your previous position:" and the next blank line
    # before "Discussion ledger so far:".
    self_match = re.search(
        r"Your previous position:\s*\n(.*?)\n\nDiscussion ledger so far:",
        user_input,
        flags=re.DOTALL,
    )
    your_last_text = (
        self_match.group(1).strip() if self_match else "(sin posición previa)"
    )

    ledger_match = re.search(
        r"Discussion ledger so far:\s*\n(.*)$",
        user_input,
        flags=re.DOTALL,
    )
    ledger_view = (
        ledger_match.group(1).strip()
        if ledger_match
        else "(aún no hay intervenciones)"
    )
    return {
        "round_number": round_number,
        "max_rounds": max_rounds,
        "your_last_text": your_last_text,
        "ledger_view": ledger_view,
    }


class SpecializedExecutor(AgentExecutor):
    """Worker AgentExecutor — runs a single dialectic role per dispatch."""

    def __init__(self, state: AgentState, model: str):
        self.state = state
        self.model = model

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        user_input = context.get_user_input()
        persona: PersonaContract = await self.state.get_persona()
        claim = await self.state.get_claim() or ""

        logger.info(
            "Agent %s (%s) running role=%s stratagem=%s",
            self.state.agent_id,
            persona.display_name,
            persona.role_id,
            persona.eristic_stratagem_id,
        )

        topic, peers_outputs = _split_topic_and_peers(user_input)
        round_ctx = _extract_round_context(user_input)
        if round_ctx is not None:
            system_prompt = render_system_prompt(
                persona,
                topic=topic,
                claim=claim,
                peers_outputs=peers_outputs,
                round_number=round_ctx["round_number"],     # type: ignore[arg-type]
                max_rounds=round_ctx["max_rounds"],         # type: ignore[arg-type]
                ledger_view=round_ctx["ledger_view"],       # type: ignore[arg-type]
                your_last_text=round_ctx["your_last_text"], # type: ignore[arg-type]
            )
        else:
            system_prompt = render_system_prompt(
                persona,
                topic=topic,
                claim=claim,
                peers_outputs=peers_outputs,
            )

        # ── Step 1: Initial LLM call (role-shaped) ──
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        try:
            initial_argument = await llm_complete(
                model=self.model,
                messages=messages,
                temperature=persona.temperature,
                max_tokens=700,
            )
        except Exception as e:
            logger.error(
                "LLM call (initial) failed for %s: %s", self.state.agent_id, e
            )
            initial_argument = f"[Error generating argument: {e}]"

        # ── Step 2 + 3: Search via the persona's preferred MCP tool. ──
        # Phase C: each role taps a different documentary pool.
        #   - Analista uses Wikipedia (encyclopedic, named-entity oriented).
        #   - Buscador uses web_search (general-purpose) and could fall
        #     back to arxiv for academic queries.
        #   - Devil's Advocate uses web_search.
        #   - Synthesizer has an empty whitelist and skips this entirely.
        # We pick the FIRST search-capable tool in whitelist order, which
        # is the persona_catalog's stated preference.
        search_query: str | None = None
        search_results: str | None = None
        search_tool: str | None = None
        search_args: dict | None = None

        _SEARCH_TOOLS = ("web_search", "wikipedia", "arxiv")
        chosen = next(
            (t for t in persona.tool_whitelist if t in _SEARCH_TOOLS), None
        )
        if chosen is not None:
            search_tool = chosen
            if chosen == "wikipedia":
                # Wikipedia expects an article title — the topic from the
                # orchestrator's prompt is a much cleaner approximation than
                # a sentence pulled from the agent's free-text argument.
                search_query = topic[:80] or _extract_search_query(initial_argument)
                search_args = {"title": search_query}
            else:  # web_search or arxiv
                search_query = _extract_search_query(initial_argument)
                search_args = {"query": search_query}

            logger.info(
                "Agent %s (%s) → %s: %s",
                self.state.agent_id, persona.role_id, chosen, search_query,
            )
            search_results = await _call_mcp_tool(
                chosen,
                search_args,
                whitelist=persona.tool_whitelist,
            )

        # ── Step 4: Refine if search produced something ──
        if search_results:
            tool_label = (search_tool or "search").replace("_", " ")
            refine_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": initial_argument},
                {
                    "role": "user",
                    "content": (
                        f"{tool_label.title()} results for '{search_query}':\n"
                        f"{search_results}\n\n"
                        "Incorporate this evidence where it strengthens your "
                        "position or helps you press your role's specific "
                        "objective. Stay in character. Keep the response concise."
                    ),
                },
            ]
            try:
                final_response = await llm_complete(
                    model=self.model,
                    messages=refine_messages,
                    temperature=persona.temperature,
                    max_tokens=900,
                )
            except Exception as e:
                logger.error(
                    "LLM call (refine) failed for %s: %s",
                    self.state.agent_id, e,
                )
                final_response = initial_argument
        else:
            final_response = initial_argument

        # ── Step 4b: Bayesian belief update over the central claim ──
        # Phase 3 / Pillar 2: the agent's posture lives in a scalar log_odds
        # value that the orchestrator's progress relay turns into a frontend
        # trajectory chart. We update twice when a search ran (once after
        # the initial argument, once after refine) so the chart shows the
        # *direction* in which evidence moved the agent.
        belief: BeliefState | None = await self.state.get_belief()
        belief_deltas: list[tuple[str, BeliefDelta]] = []
        if belief is not None:
            try:
                d_initial = await update_belief(
                    belief,
                    evidence_text=initial_argument,
                    model=self.model,
                    stage="post_initial",
                )
                belief_deltas.append(("post_initial", d_initial))
                if search_results:
                    d_refine = await update_belief(
                        belief,
                        evidence_text=final_response,
                        model=self.model,
                        stage="post_refine",
                    )
                    belief_deltas.append(("post_refine", d_refine))
            except Exception as e:
                logger.warning(
                    "Belief update failed for %s: %s",
                    self.state.agent_id, e,
                )

        # ── Step 5: Emit single Task(COMPLETED) with optional metadata Part ──
        parts: list[Part] = []
        if search_results and search_tool:
            parts.append(
                Part(
                    text=json.dumps(
                        {
                            "stage": "tool_use",
                            "message": (
                                f"{persona.display_name} usó {search_tool}"
                            ),
                            "data": {
                                "agent": self.state.agent_id,
                                "role_id": persona.role_id,
                                "stratagem_id": persona.eristic_stratagem_id,
                                "tool": search_tool,
                                "query": search_query,
                            },
                        },
                        ensure_ascii=False,
                    )
                )
            )
        # Emit one belief_update Part per delta — the frontend stitches
        # these into the per-agent trajectory series.
        for stage, delta in belief_deltas:
            parts.append(
                Part(
                    text=json.dumps(
                        {
                            "stage": "belief_update",
                            "message": (
                                f"{persona.display_name}: log_odds "
                                f"{delta.new_log_odds:+.2f} "
                                f"(Δ {delta.delta_log_odds:+.2f})"
                            ),
                            "data": {
                                "agent": self.state.agent_id,
                                "role_id": persona.role_id,
                                "claim": belief.claim if belief else "",
                                "log_odds": delta.new_log_odds,
                                "delta": delta.delta_log_odds,
                                "llr": delta.llr,
                                "rationale": delta.rationale,
                                "phase": stage,
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
