import json
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Artifact,
    Message,
    Part,
    Role,
    Task,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)

from agents.orchestrator.agent_registry import registry
from agents.orchestrator.agentic_orchestrator import AgenticOrchestrator
from agents.orchestrator.plan_executor import ProgressCallback
from agents.orchestrator.worker_spawner import get_spawner

logger = logging.getLogger(__name__)


class SSEProgressCallback(ProgressCallback):
    """Streams debate progress to the frontend via A2A SSE events."""

    def __init__(self, task_id: str, context_id: str, event_queue: EventQueue):
        self.task_id = task_id
        self.context_id = context_id
        self.event_queue = event_queue

    async def on_progress(self, stage: str, message: str, data: dict | None = None):
        logger.info("[%s] %s", stage, message)

        # Build metadata payload for the frontend
        metadata = {"stage": stage, "message": message}
        if data:
            metadata["data"] = data

        await self.event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=self.task_id,
                context_id=self.context_id,
                status=TaskStatus(
                    state=TaskState.TASK_STATE_WORKING,
                    message=Message(
                        role=Role.ROLE_AGENT,
                        parts=[Part(text=json.dumps(metadata, ensure_ascii=False))],
                    ),
                ),
            )
        )


class OrchestratorExecutor(AgentExecutor):
    """Orchestrator AgentExecutor — receives user prompts and runs the debate flow."""

    async def execute(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        user_input = context.get_user_input()
        logger.info("Orchestrator received: %s", user_input[:100])

        # Create progress callback that streams SSE events
        progress = SSEProgressCallback(
            task_id=context.task_id,
            context_id=context.context_id,
            event_queue=event_queue,
        )

        try:
            orchestrator = AgenticOrchestrator(
                registry=registry,
                spawner=get_spawner(),
                progress=progress,
            )
            verdict = await orchestrator.run(user_input)

            # Send final completed task with verdict
            await event_queue.enqueue_event(
                Task(
                    id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(
                        state=TaskState.TASK_STATE_COMPLETED,
                        message=Message(
                            role=Role.ROLE_AGENT,
                            parts=[Part(text=verdict)],
                        ),
                    ),
                    artifacts=[
                        Artifact(
                            artifact_id="verdict",
                            parts=[Part(text=verdict)],
                        )
                    ],
                )
            )

        except Exception as e:
            logger.exception("Debate flow failed: %s", e)
            await event_queue.enqueue_event(
                Task(
                    id=context.task_id,
                    context_id=context.context_id,
                    status=TaskStatus(
                        state=TaskState.TASK_STATE_FAILED,
                        message=Message(
                            role=Role.ROLE_AGENT,
                            parts=[Part(text=f"Debate flow failed: {e}")],
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
