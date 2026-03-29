import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.client.helpers import create_text_message_object
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Part,
    Role,
    SendMessageRequest,
    StreamResponse,
)


def build_agent_card(
    name: str,
    description: str,
    url: str,
    skills: list[AgentSkill] | None = None,
    streaming: bool = True,
) -> AgentCard:
    """Build an A2A AgentCard with sensible defaults."""
    return AgentCard(
        name=name,
        description=description,
        provider=AgentProvider(organization="A2A Network - University Project"),
        version="1.0.0",
        capabilities=AgentCapabilities(
            streaming=streaming,
        ),
        supported_interfaces=[
            AgentInterface(
                url=url,
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        skills=skills or [],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
    )


def build_skill(
    skill_id: str,
    name: str,
    description: str,
    tags: list[str] | None = None,
) -> AgentSkill:
    """Build an A2A AgentSkill."""
    return AgentSkill(
        id=skill_id,
        name=name,
        description=description,
        tags=tags or [],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    )


async def create_a2a_client(base_url: str) -> tuple:
    """Create an A2A client for a remote agent.

    Returns:
        Tuple of (client, agent_card)
    """
    http_client = httpx.AsyncClient(base_url=base_url, timeout=120.0)
    resolver = A2ACardResolver(httpx_client=http_client, base_url=base_url)
    card = await resolver.get_agent_card()

    config = ClientConfig(
        streaming=False,
        httpx_client=http_client,
    )
    factory = ClientFactory(config=config)
    client = factory.create(card)
    return client, card


async def send_and_get_text(
    client,
    text: str,
    context_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Send a text message to an agent and return the response text.

    Args:
        client: A2A BaseClient instance
        text: The text content to send
        context_id: Optional context ID for multi-turn
        task_id: Optional task ID for continuing a task

    Returns:
        The agent's text response.
    """
    message = create_text_message_object(role=Role.ROLE_USER, content=text)
    if context_id:
        message.context_id = context_id
    if task_id:
        message.task_id = task_id

    request = SendMessageRequest(message=message)

    result_text = ""
    async for event in client.send_message(request):
        stream_response: StreamResponse = event[0]
        if stream_response.HasField("message"):
            for part in stream_response.message.parts:
                if part.text:
                    result_text += part.text
        elif stream_response.HasField("task"):
            task = stream_response.task
            if task.artifacts:
                for artifact in task.artifacts:
                    for part in artifact.parts:
                        if part.text:
                            result_text += part.text
            if task.status and task.status.message:
                for part in task.status.message.parts:
                    if part.text:
                        result_text += part.text

    return result_text


def extract_text_from_parts(parts) -> str:
    """Extract text from a list of Part objects."""
    texts = []
    for part in parts:
        if part.text:
            texts.append(part.text)
    return "\n".join(texts)
