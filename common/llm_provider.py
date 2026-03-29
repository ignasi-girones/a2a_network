import asyncio
import os
from collections.abc import AsyncIterator

import litellm
from litellm import acompletion
from litellm.exceptions import RateLimitError

from common.config import settings

# Configure LiteLLM with API keys from settings
os.environ["GROQ_API_KEY"] = settings.groq_api_key
os.environ["GEMINI_API_KEY"] = settings.gemini_api_key
os.environ["MISTRAL_API_KEY"] = settings.mistral_api_key
os.environ["CEREBRAS_API_KEY"] = settings.cerebras_api_key

# Suppress LiteLLM verbose logging
litellm.suppress_debug_info = True


async def llm_complete(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
    response_format: dict | None = None,
) -> str:
    """Send a completion request to any LLM provider via LiteLLM.

    Args:
        model: LiteLLM model string (e.g. "groq/llama-3.3-70b-versatile")
        messages: Chat messages in OpenAI format
        temperature: Sampling temperature
        max_tokens: Max tokens in response
        response_format: Optional JSON mode (e.g. {"type": "json_object"})

    Returns:
        The assistant's response text.
    """
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if model.startswith("ollama/"):
        kwargs["api_base"] = settings.ollama_api_base

    if response_format:
        kwargs["response_format"] = response_format

    for attempt in range(3):
        try:
            response = await acompletion(**kwargs)
            return response.choices[0].message.content
        except RateLimitError:
            if attempt == 2:
                raise
            wait = [10, 25][attempt]
            await asyncio.sleep(wait)


async def llm_stream(
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> AsyncIterator[str]:
    """Stream a completion response token by token.

    Yields:
        Text chunks as they arrive.
    """
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    if model.startswith("ollama/"):
        kwargs["api_base"] = settings.ollama_api_base

    response = await acompletion(**kwargs)
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
