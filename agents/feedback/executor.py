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
Eres un analista de debates. Recibirás un resumen en JSON de un debate estructurado \
entre dos agentes con roles y perspectivas diferentes.

IMPORTANTE: Responde SIEMPRE en castellano, independientemente del idioma del debate \
de entrada. Si el debate viene en inglés, traduce los argumentos al castellano en tu informe.

Tu tarea es producir un informe claro y bien formateado para un lector humano. \
Estructura tu respuesta con estas secciones (mantén los títulos exactamente así):

## Resumen ejecutivo
Una visión general en 2-3 frases del tema del debate y su desenlace.

## Participantes
- Quién era cada agente y qué perspectiva representaba.

## Argumentos clave
Resume los argumentos más sólidos de cada lado.

## Puntos de acuerdo
¿En qué convergieron los agentes?

## Puntos de desacuerdo
¿En qué divergieron y por qué?

## Veredicto final
A la luz del debate, ¿cuál es la conclusión más equilibrada?

## Nivel de confianza
¿Cómo de fiable es este veredicto? (Alto / Medio / Bajo) y por qué.

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
