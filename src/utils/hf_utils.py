from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import TypeVar

import httpx
import requests
from datasets import Dataset, load_dataset
from huggingface_hub.errors import HfHubHTTPError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ------------------------------------------------------------------------------
# Retriable error conditions from:
# https://github.com/huggingface/datasets/blob/17f40a318a1f8c7d33c2a4dd17934f81d14a7f57/src/datasets/utils/file_utils.py

# HTTP status codes for retryable errors
RATE_LIMIT_CODE = 429
SERVER_UNAVAILABLE_CODE = 504

# Connection errors that should trigger retries
CONNECTION_ERRORS_TO_RETRY = (
    asyncio.TimeoutError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    httpx.RequestError,
)
# ------------------------------------------------------------------------------


def _is_retryable_hf_error(exception: BaseException) -> bool:
    """Check if an exception is a retryable HuggingFace HTTP error.

    Retryable errors include:
    - HfHubHTTPError with 429 (Too Many Requests) status code
    - HfHubHTTPError with 504 (Server Unavailable) status code
    - Various connection errors (timeout, connection errors, etc.)

    Parameters
    ----------
    exception : BaseException
        The exception to check

    Returns
    -------
    bool
        True if the exception should trigger a retry, False otherwise
    """
    # Check for connection errors
    if isinstance(exception, CONNECTION_ERRORS_TO_RETRY):
        return True

    # Check for HuggingFace HTTP errors with retryable status codes
    if isinstance(exception, HfHubHTTPError):
        if exception.response is None:
            return False
        status_code = exception.response.status_code
        return status_code in (RATE_LIMIT_CODE, SERVER_UNAVAILABLE_CODE)

    return False


def retry_hf_function(
    func: Callable[..., T],
    max_retries: int = 10,
    initial_backoff: float = 1.0,
    backoff_multiplier: float = 2.0,
) -> Callable[..., T]:
    """Wrap a callable with exponential backoff retry logic for HuggingFace errors.

    This function creates a wrapper around any callable that automatically retries
    on transient errors using the tenacity library. Retryable errors include:
    - HTTP 429 (Rate Limit) errors
    - HTTP 504 (Server Unavailable) errors
    - Connection errors (timeouts, disconnections, etc.)

    Parameters
    ----------
    func : Callable[..., T]
        The function to wrap with retry logic
    max_retries : int, default=10
        Maximum number of retry attempts
    initial_backoff : float, default=1.0
        Initial backoff delay in seconds
    backoff_multiplier : float, default=2.0
        Multiplier for exponential backoff

    Returns
    -------
    Callable[..., T]
        The wrapped function with retry logic
    """
    retry_decorator = retry(
        retry=retry_if_exception(_is_retryable_hf_error),
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(
            multiplier=initial_backoff,
            max=initial_backoff * (backoff_multiplier**max_retries),
            exp_base=backoff_multiplier,
        ),
        before_sleep=before_sleep_log(logger, logging.WARN),
        reraise=True,
    )
    return retry_decorator(func)


def load_hf_dataset(
    *args,
    max_retries: int = 10,
    initial_backoff: float = 1.0,
    backoff_multiplier: float = 2.0,
    **kwargs,
) -> Dataset:
    """Load a HuggingFace dataset with exponential backoff retry logic.

    This function wraps the HuggingFace `load_dataset` call with automatic
    retry logic for handling transient HTTP errors using the tenacity library.

    Parameters
    ----------
    *args
        Positional arguments passed to `load_dataset`
    max_retries : int, default=10
        Maximum number of retry attempts
    initial_backoff : float, default=1.0
        Initial backoff delay in seconds
    backoff_multiplier : float, default=2.0
        Multiplier for exponential backoff
    **kwargs
        Keyword arguments passed to `load_dataset`

    Returns
    -------
    Dataset
        The loaded HuggingFace dataset

    Raises
    ------
    HfHubHTTPError
        If all retry attempts fail
    """
    wrapped_load = retry_hf_function(
        lambda: load_dataset(*args, **kwargs),
        max_retries=max_retries,
        initial_backoff=initial_backoff,
        backoff_multiplier=backoff_multiplier,
    )
    return wrapped_load()
