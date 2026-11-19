"""Shared retry timing for transient model-provider failures."""

from tenacity import wait_exponential, wait_none


def model_retry_wait(provider: str):
    """Use production backoff while keeping deterministic fake tests immediate."""

    if provider == "fake":
        return wait_none()
    return wait_exponential(multiplier=2, min=2, max=8)
