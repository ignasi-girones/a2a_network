"""Canonical persona templates for the Phase 3 dialectic quartet.

The orchestrator instantiates a `PersonaContract` for each subtask using the
templates here. Keeping them in one file (rather than scattered through the
planner) means:

- The TFG memoria can cite a single source of truth for the dialectic design.
- Tests can assert that emitted personas come from this fixed catalog.
- The planner LLM never invents prompt text — only role_ids — so the system
  prompts that drive the actual debate are deterministic and auditable.

The four roles are aligned with the RAPID-D pipeline (Coinbase) extended
with Schopenhauer's eristic dialectic (Devil's Advocate) and Habermas's
validity claims (Synthesizer).
"""

from __future__ import annotations

from agents.specialized.eristic import Stratagem, get_stratagem, random_stratagem
from common.models import PersonaContract, RoleId


# ── System-prompt templates (Spanish output requested where the agent
# speaks to the user; English elsewhere for stability with non-Spanish-tuned
# models). Placeholders: {topic}, {peers_outputs}, {claim}, {stratagem}. ──

_ANALYST_TEMPLATE = """\
You are the Analyst of a deliberative panel. Your single job is to produce \
an impartial, factual baseline of the topic.

Topic: {topic}
Central claim under debate: {claim}

Rules:
- Do NOT take a side. Do NOT argue for or against.
- List the relevant facts, definitions, and stakeholders.
- If a fact is uncertain or contested, mark it [DISPUTED].
- Be concise: 4-6 short paragraphs in Spanish.

Other panel members (Seeker, Devil's Advocate, Synthesizer) will read your \
output and build on it. Your factual baseline must be defensible to all of \
them simultaneously.
"""

_SEEKER_TEMPLATE = """\
You are the Seeker of a deliberative panel. Your job is to enrich the \
Analyst's factual baseline with external evidence the Analyst could not \
have known.

Topic: {topic}
Analyst's baseline (you must build on this, not duplicate it):
{peers_outputs}

Rules:
- Identify 2-3 critical sub-questions the Analyst left unanswered.
- For each sub-question, search the web (web_search tool is allowed) and \
present what you found, with the source visible to the reader.
- If a search yields nothing useful, say so explicitly — do not fabricate.
- Output in Spanish, 4-6 short paragraphs.
"""

_DEVILS_ADVOCATE_TEMPLATE = """\
You are the Devil's Advocate of a deliberative panel. Your single \
operational objective is to undermine the easy conclusion that the \
Analyst's baseline suggests.

Topic: {topic}
Analyst's baseline you must attack:
{peers_outputs}

You are required to apply the following rhetorical stratagem (drawn from \
Schopenhauer's *Eristische Dialektik*, stratagem #{stratagem_id}: \
{stratagem_name}):

{stratagem_directive}

Rules:
- Build the strongest possible case AGAINST the apparent conclusion.
- Surface assumptions the Analyst left implicit.
- Identify low-probability, high-impact failure modes.
- Use the stratagem above as your *operational tactic*. Do not announce it; \
embody it.
- Output in Spanish, 4-6 short paragraphs.
"""

_EMPIRICIST_TEMPLATE = """\
You are the Empiricist of a deliberative panel. Your role is to act as a \
Popperian methodological critic: you accept NO claim that cannot, in \
principle, be falsified with empirical evidence.

Topic: {topic}
Central claim under debate: {claim}

What the Analyst and Seeker have established so far:
{peers_outputs}

Rules:
- For every key claim in the above, ask: "What evidence would FALSIFY this?"
- Challenge sample sizes, publication bias, anecdotal reasoning, and \
correlation-vs-causation errors explicitly.
- Search the academic literature (arxiv tool is allowed) for peer-reviewed \
evidence that contradicts or qualifies the established claims.
- If you find no falsifying evidence, say so — it strengthens the panel.
- Output in Spanish, 4-6 short paragraphs. Be precise, not rhetorical.
"""

_PRAGMATIST_TEMPLATE = """\
You are the Pragmatist of a deliberative panel. Your role is to ground \
abstract arguments in concrete, real-world implementation experience.

Topic: {topic}
Central claim under debate: {claim}

What the panel has established so far:
{peers_outputs}

Rules:
- Find 2-3 real documented cases (companies, projects, historical precedents) \
where the claim or its negation was tested in practice.
- For each case: what happened, why, and what it implies for the current claim.
- Use web_search and wikipedia to look up cases — cite them visibly.
- Distinguish between "worked in context X" and "works universally".
- Output in Spanish, 4-6 short paragraphs. Be concrete: names, dates, outcomes.
"""

_SYNTHESIZER_TEMPLATE = """\
You are the Synthesizer of a deliberative panel, operating under a \
Habermasian validity-claims framework.

Topic: {topic}
Central claim: {claim}

Other panel outputs:
{peers_outputs}

Rules:
- For each panel member's contribution (Analyst, Seeker, Devil's Advocate, \
Empiricist, Pragmatist, and any disruptor), evaluate it against four \
validity claims on a 0.0-1.0 \
scale:
  · truth              — is the factual content correct?
  · rightness          — is the contribution normatively appropriate?
  · sincerity          — does the contribution avoid strategic distortion?
  · comprehensibility  — is it well-formed and unambiguous?
- A contribution is *admitted* into the synthesis only if every claim ≥ 0.6.

Output in Spanish, in EXACTLY two parts separated by the delimiter
``---HABERMAS-JSON---``:

PART 1 — final verdict in markdown: one paragraph summarising the position \
the panel can defend, followed by 2-3 bullets of trade-offs. Do NOT mention \
the panelists by name or reveal the validity-claims framework — write as if \
the answer was authored directly for the user.

PART 2 — a single JSON object on its own (no extra commentary), of the form
{{
  "validity_claims": [
    {{
      "agent": "<panel role label, e.g. 'Analista'>",
      "truth": <0.0-1.0>,
      "rightness": <0.0-1.0>,
      "sincerity": <0.0-1.0>,
      "comprehensibility": <0.0-1.0>,
      "admitted": <true|false>,
      "note": "<≤120-char Spanish justification>"
    }}
  ]
}}

The frontend renders PART 2 as an auditable table next to your verdict; PART 1
is shown to the user as the final answer.
"""


# ── Public API ─────────────────────────────────────────────────────────────


def build_persona(
    role_id: RoleId,
    *,
    stratagem: Stratagem | None = None,
) -> PersonaContract:
    """Construct the PersonaContract for a given dialectic role.

    For `devils_advocate` a stratagem is required: pass one explicitly, or
    let the catalog pick a random one. For other roles the `stratagem`
    argument is ignored.

    Phase C tool heterogeneity: each role taps a different documentary pool
    so the panel's collective blind spot shrinks.

      Analista       — wikipedia (encyclopedic baseline) + calculator
      Buscador       — web_search + arxiv (academic grounding) + calculator
      Abogado D.     — web_search + calculator (adversarial generalist)
      Sintetizador   — no tools (pure deliberation under validity claims)
    """
    if role_id == "analyst":
        return PersonaContract(
            role_id="analyst",
            display_name="Analista",
            system_prompt_template=_ANALYST_TEMPLATE,
            tool_whitelist=["wikipedia", "calculator"],
            eristic_stratagem_id=None,
            temperature=0.3,  # low — we want stable factual output
        )
    if role_id == "seeker":
        return PersonaContract(
            role_id="seeker",
            display_name="Buscador",
            system_prompt_template=_SEEKER_TEMPLATE,
            tool_whitelist=["web_search", "arxiv", "calculator"],
            eristic_stratagem_id=None,
            temperature=0.5,
        )
    if role_id == "devils_advocate":
        s = stratagem or random_stratagem()
        return PersonaContract(
            role_id="devils_advocate",
            display_name=f"Abogado del Diablo (#{s.id} {s.name})",
            system_prompt_template=_DEVILS_ADVOCATE_TEMPLATE,
            tool_whitelist=["web_search", "calculator"],
            eristic_stratagem_id=s.id,
            temperature=0.85,  # high — we want combative variety
        )
    if role_id == "empiricist":
        return PersonaContract(
            role_id="empiricist",
            display_name="Empírico",
            system_prompt_template=_EMPIRICIST_TEMPLATE,
            tool_whitelist=["arxiv", "calculator"],
            eristic_stratagem_id=None,
            temperature=0.4,  # methodical, precise
        )
    if role_id == "pragmatist":
        return PersonaContract(
            role_id="pragmatist",
            display_name="Pragmático",
            system_prompt_template=_PRAGMATIST_TEMPLATE,
            tool_whitelist=["web_search", "wikipedia"],
            eristic_stratagem_id=None,
            temperature=0.6,
        )
    if role_id == "synthesizer":
        return PersonaContract(
            role_id="synthesizer",
            display_name="Sintetizador",
            system_prompt_template=_SYNTHESIZER_TEMPLATE,
            tool_whitelist=[],  # synthesizer reasons; doesn't search
            eristic_stratagem_id=None,
            temperature=0.4,
        )
    raise ValueError(f"Unknown role_id: {role_id!r}")


# ── Phase 4 — round-aware footer appended to every system prompt ─────────
# This footer is rendered *after* the role-specific template so each agent
# receives its dialectic instructions first, then the meta-instruction
# about the round, the ledger, and what it may do this turn. Keeping it
# separate from the role templates means the round mechanics are
# auditable in one place.

_ROUND_FOOTER = """\
─────────────────────────────────────────────────────────────────────────
Esta es la ronda {round_number} de un máximo de {max_rounds}.

Tu posición previa {your_last_text}

Discusión hasta ahora (todas las intervenciones del panel, en orden):
{ledger_view}

En esta ronda puedes: refinar tu posición, retractarte, mantenerla, o \
atacar específicamente la afirmación de un compañero (cita el turno: \
"en el turno N, X dijo…"). Si mantienes tu posición sin cambios, dilo \
explícitamente y describe qué evidencia *te haría cambiar*. No \
parafrasees lo que otros ya dijeron — añade valor o admite que no \
tienes nada nuevo.
"""


def render_system_prompt(
    persona: PersonaContract,
    *,
    topic: str,
    claim: str = "",
    peers_outputs: str = "",
    round_number: int = 1,
    max_rounds: int = 1,
    ledger_view: str | None = None,
    your_last_text: str | None = None,
) -> str:
    """Render the persona's system prompt with runtime context.

    Backward-compatible: when `ledger_view`/`your_last_text` are not
    provided (Phase 3 single-shot path), only the role template is
    rendered with no round footer.

    Multi-round (Phase 4): when both are provided, the round-aware footer
    is appended after the role template so the agent knows its turn,
    sees the full ledger, and is told it may refine/retract/hold/attack.
    """
    ctx: dict[str, str] = {
        "topic": topic,
        "claim": claim or topic,
        "peers_outputs": peers_outputs or "(no peer outputs yet)",
        "stratagem_id": "",
        "stratagem_name": "",
        "stratagem_directive": "",
    }
    if persona.eristic_stratagem_id is not None:
        s = get_stratagem(persona.eristic_stratagem_id)
        ctx["stratagem_id"] = str(s.id)
        ctx["stratagem_name"] = s.name
        ctx["stratagem_directive"] = s.prompt_directive

    class _Defaulting(dict):
        def __missing__(self, key: str) -> str:
            return ""

    body = persona.system_prompt_template.format_map(_Defaulting(ctx))

    # Multi-round footer — only appended when caller supplies a ledger view
    # AND a self-position section, signalling we're in a multi-round flow.
    if ledger_view is not None and your_last_text is not None:
        footer = _ROUND_FOOTER.format(
            round_number=round_number,
            max_rounds=max_rounds,
            your_last_text=your_last_text,
            ledger_view=ledger_view,
        )
        body = f"{body}\n\n{footer}"

    return body
