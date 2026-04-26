"""Debate Flow Manager — core orchestration logic.

Manages the full lifecycle of a debate:
1. Normalize user input
2. Decide roles for AE1/AE2 via LLM
3. Configure AE agents dynamically
4. Collect initial opinions (parallel)
5. Run debate rounds (max N)
6. Evaluate consensus after each round
7. Generate summary and send to Feedback agent
"""

import asyncio
import json
import logging
from uuid import uuid4

import httpx

from common.a2a_helpers import create_a2a_client, send_and_get_text
from common.config import settings
from common.llm_provider import llm_complete
from common.models import (
    AgentRoleConfig,
    DebateRound,
    DebateState,
    FlowResult,
    RoleDecision,
)

logger = logging.getLogger(__name__)


ROLE_DECISION_PROMPT = """\
Assign contrasting roles for two debate agents on the given topic.
Return ONLY valid JSON:
{
  "ae1_config": {"role": "...", "perspective": "...", "skills": [{"id": "...", "name": "..."}]},
  "ae2_config": {"role": "...", "perspective": "...", "skills": [{"id": "...", "name": "..."}]},
  "max_rounds": 2
}
Rules: roles must be professional titles, perspectives must be opposing, max_rounds always 2."""

CONSENSUS_CHECK_PROMPT = """\
You are evaluating whether two debate agents have reached substantial consensus.

AE1 ({ae1_role}) latest position:
{ae1_text}

AE2 ({ae2_role}) latest position:
{ae2_text}

Do these positions substantially agree on the core question? \
Minor stylistic differences don't count — focus on substantive agreement.

Return ONLY valid JSON: {{"consensus": true/false, "reason": "brief explanation"}}"""

SUMMARY_PROMPT = """\
Summarize this debate concisely. Include the key arguments from each side, \
points of agreement, points of disagreement, and whether consensus was reached.

Debate topic: {topic}
AE1 role: {ae1_role}
AE2 role: {ae2_role}

Debate content:
{debate_content}

Return a clear text summary (not JSON)."""


class ProgressCallback:
    """Callback interface for streaming progress to the frontend."""

    async def on_progress(self, stage: str, message: str, data: dict | None = None):
        """Called at each milestone."""
        pass


class FlowManager:
    def __init__(self, progress: ProgressCallback | None = None):
        self.progress = progress or ProgressCallback()

    async def run_debate(self, user_input: str) -> str:
        """Execute the full debate flow and return the final verdict."""

        # Step 1: Normalize
        await self.progress.on_progress("normalize", "Normalizando entrada...")
        normalized_json = await self._normalize(user_input)
        logger.info("Normalized: %s", normalized_json[:200])

        # Step 2: Decide roles
        await self.progress.on_progress("roles", "Decidiendo roles para los agentes...")
        role_decision = await self._decide_roles(normalized_json)
        await self.progress.on_progress(
            "roles_decided",
            f"AE1: {role_decision.ae1_config.role} | AE2: {role_decision.ae2_config.role}",
            {
                "ae1_role": role_decision.ae1_config.role,
                "ae2_role": role_decision.ae2_config.role,
                "max_rounds": role_decision.max_rounds,
            },
        )

        # Step 3: Configure AE agents
        await self.progress.on_progress("configure", "Configurando agentes especializados...")
        await self._configure_agents(role_decision)

        # Step 4: Initial opinions (parallel)
        await self.progress.on_progress("opinions", "Agentes formulando opiniones iniciales...")
        ae1_opinion, ae2_opinion = await self._get_initial_opinions(
            normalized_json, role_decision
        )
        await self.progress.on_progress(
            "ae1_opinion", f"AE1 ({role_decision.ae1_config.role}): opinión inicial",
            {"agent": "ae1", "text": ae1_opinion},
        )
        await self.progress.on_progress(
            "ae2_opinion", f"AE2 ({role_decision.ae2_config.role}): opinión inicial",
            {"agent": "ae2", "text": ae2_opinion},
        )

        # Step 5: Debate loop
        debate_state = DebateState(
            topic=normalized_json,
            ae1_role=role_decision.ae1_config.role,
            ae2_role=role_decision.ae2_config.role,
            initial_opinions={"ae1": ae1_opinion, "ae2": ae2_opinion},
            max_rounds=role_decision.max_rounds,
        )

        # Each list is the agent's full chronological output: [initial, round1, round2, ...]
        ae1_arguments: list[str] = [ae1_opinion]
        ae2_arguments: list[str] = [ae2_opinion]
        context_id = str(uuid4())

        for round_num in range(1, role_decision.max_rounds + 1):
            await self.progress.on_progress(
                "debate_round", f"Ronda {round_num}/{role_decision.max_rounds}",
                {"round": round_num},
            )

            # AE2 responds to AE1's latest position (initial or last round)
            ae2_prompt = self._build_round_prompt(
                topic=normalized_json,
                role=role_decision.ae2_config.role,
                own_arguments=ae2_arguments,
                opponent_arguments=ae1_arguments,
                round_num=round_num,
                max_rounds=role_decision.max_rounds,
            )
            ae2_response = await self._send_to_agent(
                port=settings.ae2_port,
                text=ae2_prompt,
                context_id=context_id,
            )
            ae2_arguments.append(ae2_response)
            await self.progress.on_progress(
                "ae2_argues",
                f"Ronda {round_num}: AE2 responde",
                {"agent": "ae2", "round": round_num, "text": ae2_response},
            )

            # AE1 responds to AE2's just-issued argument
            ae1_prompt = self._build_round_prompt(
                topic=normalized_json,
                role=role_decision.ae1_config.role,
                own_arguments=ae1_arguments,
                opponent_arguments=ae2_arguments,
                round_num=round_num,
                max_rounds=role_decision.max_rounds,
            )
            ae1_response = await self._send_to_agent(
                port=settings.ae1_port,
                text=ae1_prompt,
                context_id=context_id,
            )
            ae1_arguments.append(ae1_response)
            await self.progress.on_progress(
                "ae1_argues",
                f"Ronda {round_num}: AE1 responde",
                {"agent": "ae1", "round": round_num, "text": ae1_response},
            )

            debate_state.rounds.append(
                DebateRound(
                    round_number=round_num,
                    ae1_argument=ae1_response,
                    ae2_argument=ae2_response,
                )
            )

            # Check consensus on the two latest positions
            consensus = await self._check_consensus(
                ae1_response, ae2_response,
                role_decision.ae1_config.role,
                role_decision.ae2_config.role,
            )
            if consensus:
                debate_state.consensus_reached = True
                await self.progress.on_progress(
                    "consensus", f"Consenso alcanzado en ronda {round_num}"
                )
                break

        if not debate_state.consensus_reached:
            await self.progress.on_progress(
                "max_rounds", "Máximo de rondas alcanzado, sintetizando..."
            )

        # Step 6: Generate summary
        await self.progress.on_progress("summary", "Generando resumen del debate...")
        summary = await self._generate_summary(debate_state)

        # Step 7: Get formatted verdict from Feedback agent
        await self.progress.on_progress("feedback", "Generando veredicto final...")
        flow_result = FlowResult(
            topic=debate_state.topic,
            ae1_role=debate_state.ae1_role,
            ae2_role=debate_state.ae2_role,
            ae1_initial_opinion=ae1_opinion,
            ae2_initial_opinion=ae2_opinion,
            debate_rounds=debate_state.rounds,
            consensus_reached=debate_state.consensus_reached,
            summary=summary,
        )

        verdict = await self._get_feedback(flow_result)
        await self.progress.on_progress("complete", "Debate completado", {"verdict": verdict})

        return verdict

    # ── Private methods ──

    async def _normalize(self, user_input: str) -> str:
        """Send input to Normalizer agent via A2A."""
        client, _ = await create_a2a_client(
            settings.agent_url(settings.normalizer_port)
        )
        try:
            return await send_and_get_text(client, user_input)
        finally:
            await client.close()

    async def _decide_roles(self, normalized_json: str) -> RoleDecision:
        """Use orchestrator's LLM to decide agent roles."""
        messages = [
            {"role": "system", "content": ROLE_DECISION_PROMPT},
            {"role": "user", "content": normalized_json},
        ]

        for attempt in range(2):
            result = await llm_complete(
                model=settings.orchestrator_model,
                messages=messages,
                temperature=0.7,
                max_tokens=400,
                response_format={"type": "json_object"},
            )
            try:
                data = json.loads(result)
                return RoleDecision(**data)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Role decision parse failed (attempt %d): %s", attempt + 1, e)
                messages.append({"role": "assistant", "content": result})
                messages.append({
                    "role": "user",
                    "content": "That JSON was invalid. Return ONLY valid JSON.",
                })

        # Fallback
        return RoleDecision(
            ae1_config=AgentRoleConfig(
                role="Advocate",
                perspective="In favor",
            ),
            ae2_config=AgentRoleConfig(
                role="Critic",
                perspective="Against",
            ),
            max_rounds=2,
        )

    async def _configure_agents(self, decision: RoleDecision) -> None:
        """Configure both AE agents via internal API (not A2A)."""
        async with httpx.AsyncClient() as http:
            ae1_url = f"{settings.agent_url(settings.ae1_port)}/internal/configure"
            ae2_url = f"{settings.agent_url(settings.ae2_port)}/internal/configure"

            ae1_task = http.post(ae1_url, json=decision.ae1_config.model_dump())
            ae2_task = http.post(ae2_url, json=decision.ae2_config.model_dump())

            results = await asyncio.gather(ae1_task, ae2_task, return_exceptions=True)
            for i, r in enumerate(results):
                agent = "AE1" if i == 0 else "AE2"
                if isinstance(r, Exception):
                    logger.error("Failed to configure %s: %s", agent, r)
                    raise RuntimeError(f"Failed to configure {agent}: {r}")
                if r.status_code != 200:
                    logger.error("%s config returned %d: %s", agent, r.status_code, r.text)

    async def _get_initial_opinions(
        self, normalized_json: str, decision: RoleDecision
    ) -> tuple[str, str]:
        """Get initial opinions from both AEs in parallel via A2A."""
        prompt = (
            f"Analyze this topic from your assigned perspective and formulate "
            f"your initial position:\n\n{normalized_json}"
        )

        async def get_opinion(port: int) -> str:
            client, _ = await create_a2a_client(settings.agent_url(port))
            try:
                return await send_and_get_text(
                    client, prompt, on_intermediate=self._relay_intermediate
                )
            finally:
                await client.close()

        ae1_opinion, ae2_opinion = await asyncio.gather(
            get_opinion(settings.ae1_port),
            get_opinion(settings.ae2_port),
        )
        return ae1_opinion, ae2_opinion

    async def _send_to_agent(
        self, port: int, text: str, context_id: str | None = None
    ) -> str:
        """Send a message to an agent via A2A."""
        client, _ = await create_a2a_client(settings.agent_url(port))
        try:
            return await send_and_get_text(
                client, text, context_id=context_id,
                on_intermediate=self._relay_intermediate,
            )
        finally:
            await client.close()

    async def _relay_intermediate(self, metadata: dict) -> None:
        """Relay intermediate events from AE agents to the frontend SSE stream."""
        stage = metadata.get("stage", "info")
        message = metadata.get("message", "")
        data = metadata.get("data")
        await self.progress.on_progress(stage, message, data)

    def _build_round_prompt(
        self,
        *,
        topic: str,
        role: str,
        own_arguments: list[str],
        opponent_arguments: list[str],
        round_num: int,
        max_rounds: int,
    ) -> str:
        """Build a structured per-round prompt with full debate history.

        own_arguments[0] is the agent's initial opinion; the rest are its
        responses from prior rounds. opponent_arguments works the same way,
        and its last entry is the move this turn must respond to.
        """
        is_final = round_num == max_rounds and max_rounds >= 2
        if is_final:
            goal = (
                "FINAL ROUND — synthesis. The adversarial phase is over. "
                "Drop your assigned-perspective stance and propose a single "
                "integrated answer that combines the strongest points from "
                "both sides. Lead with the shared ground you have already "
                "built, then offer a unified verdict you would accept as "
                "the conclusion of this deliberation."
            )
        elif round_num == 1:
            goal = (
                "Round 1 — argue your position from your assigned perspective, "
                "but explicitly acknowledge any valid points from the opponent's "
                "opening before pushing back. Do NOT just restate your initial "
                "opinion; engage with what they actually said."
            )
        else:
            goal = (
                f"Round {round_num} of {max_rounds} — focus on convergence. "
                "Lead with the points of AGREEMENT you have already reached, "
                "then narrow the remaining disagreements. Concede explicitly "
                "where the opponent has changed your mind. Each round your "
                "AGREEMENTS section should grow."
            )

        def label(i: int) -> str:
            return "opening" if i == 0 else f"round {i}"

        own_block = "\n\n".join(
            f"[Your {label(i)} argument]\n{a}"
            for i, a in enumerate(own_arguments)
        )
        opp_block = "\n\n".join(
            f"[Opponent's {label(i)} argument]\n{a}"
            for i, a in enumerate(opponent_arguments)
        )

        return (
            f"[Topic]\n{topic}\n\n"
            f"[Your role] {role}\n\n"
            f"{own_block}\n\n"
            f"{opp_block}\n\n"
            f"[Goal for this round]\n{goal}"
        )

    async def _check_consensus(
        self, ae1_text: str, ae2_text: str, ae1_role: str, ae2_role: str
    ) -> bool:
        """Evaluate if agents have reached consensus."""
        prompt = CONSENSUS_CHECK_PROMPT.format(
            ae1_role=ae1_role,
            ae2_role=ae2_role,
            ae1_text=ae1_text,
            ae2_text=ae2_text,
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            result = await llm_complete(
                model=settings.orchestrator_model,
                messages=messages,
                temperature=0.3,
                max_tokens=200,
                response_format={"type": "json_object"},
            )
            data = json.loads(result)
            return data.get("consensus", False)
        except Exception as e:
            logger.warning("Consensus check failed: %s", e)
            return False

    async def _generate_summary(self, state: DebateState) -> str:
        """Generate a text summary of the entire debate."""
        debate_content = f"Initial opinions:\nAE1: {state.initial_opinions.get('ae1', '')}\nAE2: {state.initial_opinions.get('ae2', '')}\n\n"
        for r in state.rounds:
            debate_content += f"Round {r.round_number}:\nAE1: {r.ae1_argument}\nAE2: {r.ae2_argument}\n\n"

        prompt = SUMMARY_PROMPT.format(
            topic=state.topic,
            ae1_role=state.ae1_role,
            ae2_role=state.ae2_role,
            debate_content=debate_content,
        )
        messages = [{"role": "user", "content": prompt}]

        return await llm_complete(
            model=settings.orchestrator_model,
            messages=messages,
            temperature=0.5,
            max_tokens=800,
        )

    async def _get_feedback(self, flow_result: FlowResult) -> str:
        """Send debate results to Feedback agent via A2A."""
        client, _ = await create_a2a_client(
            settings.agent_url(settings.feedback_port)
        )
        try:
            return await send_and_get_text(client, flow_result.model_dump_json())
        finally:
            await client.close()
