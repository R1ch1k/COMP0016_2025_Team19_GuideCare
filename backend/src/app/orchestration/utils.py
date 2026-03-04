import asyncio
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("orch")


async def with_retry_timeout(
    fn: Callable[..., Any],
    *args,
    timeout: float = 3.0,
    retries: int = 1,
    **kwargs,
) -> Any:
    last: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(fn(*args, **kwargs), timeout=timeout)
        except Exception as e:
            last = e
            if attempt == retries:
                raise
    raise last  # pragma: no cover


def log_step(conversation_id: str, step: str, **fields) -> None:
    details = " ".join([f"{k}={v}" for k, v in fields.items()])
    logger.info("[%s] step=%s %s", conversation_id, step, details)
