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

from agents.specialized.agent_state import AgentState
from common.llm_provider import llm_complete

logger = logging.getLogger(__name__)


class SpecializedExecutor(AgentExecutor):
    """Dynamic AgentExecutor that reads its behavior from AgentState.

    The system prompt and skills are set by the orchestrator before each
    debate flow via the internal config API.
    """

    def __init__(self, state: AgentState, model: str):
        self.state = state
        self.model = model

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        user_input = context.get_user_input()
        role = await self.state.get_role()
        logger.info("Agent %s (%s) processing: %s", self.state.agent_id, role, user_input[:100])

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            )
        )

        system_prompt = await self.state.get_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            result = await llm_complete(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )
        except Exception as e:
            logger.error("LLM call failed for %s: %s", self.state.agent_id, e)
            result = f"[Error: Agent {self.state.agent_id} failed to generate response: {e}]"

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
