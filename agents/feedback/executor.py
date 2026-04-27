import logging

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

from common.config import settings
from common.llm_provider import llm_complete

logger = logging.getLogger(__name__)

FEEDBACK_PROMPT = """\
Eres un analista de debates. Recibirás los argumentos finales de un debate \
estructurado entre tres agentes (AE1, AE2 y AE3, donde AE3 actúa como \
mediador neutral).

IMPORTANTE: Responde SIEMPRE en castellano, independientemente del idioma del debate \
de entrada. Si el debate viene en inglés, traduce los argumentos al castellano en tu informe.

Produce un informe claro y bien formateado para un lector humano. Mantén los \
títulos de sección EXACTAMENTE así:

## Resumen ejecutivo
Visión general en 2-3 frases del tema del debate y su desenlace.

## Participantes
- AE1, AE2 y AE3: rol y perspectiva de cada uno (señala que AE3 es el mediador neutral).

## Argumentos clave
Resume en pocas líneas los argumentos más sólidos de cada agente.

## Puntos de acuerdo
¿En qué convergieron los tres agentes?

## Puntos de desacuerdo
¿Qué disensos quedaron sin resolver y por qué?

## Veredicto final
A la luz del debate, ¿cuál es la conclusión más equilibrada? Si AE3 propuso \
una síntesis viable, recógela aquí.

## Estado del debate
Esta sección NO trata sobre la calidad del debate ni sobre tu nivel de \
confianza. Analiza brevemente (3-5 frases) si el debate **quedó concluido con \
un consenso real** entre los agentes, o si **no llegaron a un acuerdo común** \
y persisten posiciones encontradas. Justifica usando los puntos de acuerdo \
y desacuerdo anteriores. Empieza la sección con una etiqueta clara en \
**negrita** entre estas tres opciones:
  - **Consenso alcanzado** — los tres convergieron en una respuesta común.
  - **Consenso parcial** — hay terreno común pero quedan disensos relevantes.
  - **Sin consenso** — siguen en posiciones contrapuestas.

Usa un tono profesional pero accesible. Todo el texto debe estar en castellano."""


class FeedbackExecutor(AgentExecutor):
    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        user_input = context.get_user_input()
        logger.info("Generating feedback report")

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            )
        )

        messages = [
            {"role": "system", "content": FEEDBACK_PROMPT},
            {"role": "user", "content": user_input},
        ]

        try:
            result = await llm_complete(
                model=settings.feedback_model,
                messages=messages,
                temperature=0.5,
                max_tokens=3000,
            )
        except Exception as e:
            logger.error("Feedback LLM failed: %s, falling back to Groq", e)
            result = await llm_complete(
                model=settings.orchestrator_model,
                messages=messages,
                temperature=0.5,
                max_tokens=3000,
            )

        await event_queue.enqueue_event(
            Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.TASK_STATE_COMPLETED,
                    message=Message(
                        role=Role.ROLE_AGENT,
                        parts=[Part(text=result)],
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
