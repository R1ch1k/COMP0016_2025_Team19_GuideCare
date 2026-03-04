"""
LLM wrapper — routes to OpenAI API or a local gpt-oss-20b server.

Toggle via LLM_MODE in .env:
  LLM_MODE=api    → OpenAI API  (requires OPENAI_API_KEY)
  LLM_MODE=local  → local model (requires LOCAL_MODEL_URL running)

All LLM interactions go through the `generate()` function.
Triage always uses the API via `generate_api_only()` regardless of LLM_MODE.
"""

import logging
from typing import Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


async def generate(
    prompt: str,
    max_tokens: int = 300,
    temperature: float = 0.0,
    system_message: Optional[str] = None,
) -> str:
    """Generate a response using the configured LLM backend.

    Routes to OpenAI API or a local OpenAI-compatible server
    based on the LLM_MODE setting.
    """
    if settings.LLM_MODE == "local":
        return await _generate_local(prompt, max_tokens, temperature, system_message)
    return await _generate_api(prompt, max_tokens, temperature, system_message)


async def _generate_api(
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_message: Optional[str],
) -> str:
    """Call OpenAI API."""
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not configured. Set it in .env or environment."
        )

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    logger.debug(
        "LLM call (API): model=%s, max_tokens=%d, temp=%.1f",
        settings.OPENAI_MODEL,
        max_tokens,
        temperature,
    )

    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = response.choices[0].message.content or ""
    logger.debug("LLM response (API): %d chars", len(result))
    return result


async def _generate_local(
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_message: Optional[str],
) -> str:
    """Call a local OpenAI-compatible server (e.g. vLLM, text-generation-inference).

    The local server must expose a /v1/chat/completions endpoint.
    Set LOCAL_MODEL_URL and LOCAL_MODEL_NAME in .env.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=settings.LOCAL_MODEL_URL,
        api_key="not-needed",
    )

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    logger.debug(
        "LLM call (local): model=%s, url=%s, max_tokens=%d, temp=%.1f",
        settings.LOCAL_MODEL_NAME,
        settings.LOCAL_MODEL_URL,
        max_tokens,
        temperature,
    )

    response = await client.chat.completions.create(
        model=settings.LOCAL_MODEL_NAME,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = response.choices[0].message.content or ""
    logger.debug("LLM response (local): %d chars", len(result))
    return result


async def generate_api_only(
    prompt: str,
    max_tokens: int = 400,
    temperature: float = 0.0,
    system_message: Optional[str] = None,
) -> str:
    """Always use OpenAI API regardless of LLM_MODE.

    Used for triage — must use GPT-4 for reliable urgency classification.
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not configured. Triage requires the OpenAI API."
        )

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    logger.debug(
        "LLM call (API-only triage): model=%s, max_tokens=%d, temp=%.1f",
        settings.OPENAI_MODEL,
        max_tokens,
        temperature,
    )

    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    result = response.choices[0].message.content or ""
    logger.debug("LLM response (API-only triage): %d chars", len(result))
    return result
