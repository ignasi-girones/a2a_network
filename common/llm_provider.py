import asyncio
import contextvars
import os
from collections.abc import AsyncIterator

import litellm
from litellm import acompletion, aembedding
from litellm.exceptions import RateLimitError

from common.config import settings

# Exposes token usage from the last llm_complete call to the telemetry hook
# without changing the function's return type (str).
_last_usage: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "_last_usage", default=None
)

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
            if hasattr(response, "usage") and response.usage:
                _last_usage.set({
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
                })
            return response.choices[0].message.content
        except RateLimitError:
            if attempt == 2:
                raise
            wait = [10, 25][attempt]
            await asyncio.sleep(wait)


async def llm_embed(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """Embed a list of texts using LiteLLM's aembedding.

    Used by the consensus metrics module to anchor agent positions to real
    text similarity instead of relying on an LLM evaluator's subjective
    intuition.

    Args:
        texts: list of strings to embed.
        model: LiteLLM-style embedding model slug. If None, falls back to
            settings.embedding_model.

    Returns:
        A list of embedding vectors (list of floats), aligned with the
        input. Raises if the embedding call fails — the caller decides
        whether to degrade gracefully.
    """
    if not texts:
        return []
    chosen = model or settings.embedding_model
    kwargs: dict = {
        "model": chosen,
        "input": texts,
    }
    if chosen.startswith("ollama/"):
        kwargs["api_base"] = settings.ollama_api_base

    response = await aembedding(**kwargs)
    # litellm returns a list of {"embedding": [...]} dicts in `data`.
    return [d["embedding"] for d in response.data]


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
