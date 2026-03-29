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
You are a debate analyst. You will receive a JSON summary of a structured debate \
between two agents with different roles and perspectives.

Your job is to produce a clear, well-formatted report for a human reader. \
Structure your response with these sections:

## Executive Summary
A 2-3 sentence overview of the debate topic and outcome.

## Participants
- Who each agent was and what perspective they represented.

## Key Arguments
Summarize the strongest arguments from each side.

## Points of Agreement
Where did the agents converge?

## Points of Disagreement
Where did they diverge and why?

## Final Verdict
Based on the debate, what is the most balanced conclusion?

## Confidence Level
How confident is this verdict? (High / Medium / Low) and why.

Write in a professional but accessible tone. Use the original language of the debate."""


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
