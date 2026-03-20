"""Exponential-backoff retry decorator for transient failures."""

from __future__ import annotations

import functools
import logging
import time

_logger = logging.getLogger(__name__)


def retry(max_attempts=3, backoff_base=2, retryable=(Exception,), label=None):
    """Exponential backoff retry decorator for transient failures."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            tag = label or fn.__name__
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except retryable as e:
                    if attempt == max_attempts:
                        _logger.error("[retry:%s] Failed after %d attempts: %s", tag, max_attempts, e)
                        raise
                    delay = backoff_base ** attempt
                    _logger.warning("[retry:%s] Attempt %d/%d failed: %s — retrying in %ds", tag, attempt, max_attempts, e, delay)
                    time.sleep(delay)
        return wrapper
    return decorator
