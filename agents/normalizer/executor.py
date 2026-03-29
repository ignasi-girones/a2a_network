import json
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

NORMALIZE_PROMPT = """\
You are an input normalizer. Your job is to analyze the user's message and extract structured information.

Return a JSON object with exactly these fields:
- "topic": the main topic or question (string)
- "domain": the knowledge domain (e.g. "finance", "technology", "hr", "law") (string)
- "question_type": one of "opinion", "decision", "analysis", "comparison" (string)
- "constraints": any constraints or requirements mentioned (array of strings)
- "suggested_perspectives": two contrasting perspectives that could debate this topic (array of exactly 2 strings)

Return ONLY valid JSON, no markdown, no explanation."""


class NormalizerExecutor(AgentExecutor):
    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        user_input = context.get_user_input()
        logger.info("Normalizing input: %s", user_input[:100])

        # Signal working state
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            )
        )

        messages = [
            {"role": "system", "content": NORMALIZE_PROMPT},
            {"role": "user", "content": user_input},
        ]

        # Try up to 2 times to get valid JSON
        for attempt in range(2):
            try:
                result = await llm_complete(
                    model=settings.normalizer_model,
                    messages=messages,
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                # Validate it's parseable JSON
                json.loads(result)
                break
            except (json.JSONDecodeError, Exception) as e:
                logger.warning("Attempt %d failed: %s", attempt + 1, e)
                if attempt == 1:
                    # Fallback: minimal structure
                    result = json.dumps({
                        "topic": user_input,
                        "domain": "general",
                        "question_type": "analysis",
                        "constraints": [],
                        "suggested_perspectives": [
                            "Advocate perspective",
                            "Critical perspective",
                        ],
                    })
                else:
                    messages.append({"role": "assistant", "content": result})
                    messages.append({
                        "role": "user",
                        "content": "That was not valid JSON. Please return ONLY a valid JSON object.",
                    })

        # Send completed response
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
