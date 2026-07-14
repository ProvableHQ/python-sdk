"""Shared models, error types, retry logic and JWT helpers for AleoNetworkClient."""
from __future__ import annotations

import random
import time
from typing import Any, Callable, TypeVar
from urllib.parse import urlparse

_T = TypeVar("_T")


class AleoError(Exception):
    """Base class for every error raised by the SDK.

    Defined in this lowest-level shared module so the network/scanner error
    types and the facade error types can all subclass it — a single
    ``except AleoError`` catches everything.
    """


class AleoNetworkError(AleoError):
    """Raised when an Aleo network API call fails."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class AleoProvingError(AleoError):
    """Raised when DPS proving fails (non-retried errors)."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


FIVE_MINUTES_MS: int = 5 * 60 * 1000
DEFAULT_HOST: str = "https://api.provable.com"
DEFAULT_NETWORK: str = "mainnet"

# The hosted Provable API splits its services across path prefixes off a single
# origin (reads at /v2, delegated proving at /prove, hosted scanner at /scanner,
# JWT auth at /jwts). We detect it by host so that EVERY other endpoint (devnode,
# local, or any custom node) is treated as a literal read base — no /v2 magic,
# and no prover/scanner wired up (those services only exist on the hosted API).
PROVABLE_API_HOSTS: frozenset[str] = frozenset({"api.provable.com"})


def is_provable_host(url: str) -> bool:
    """True if *url* points at the hosted Provable API (api.provable.com)."""
    return (urlparse(url).hostname or "").lower() in PROVABLE_API_HOSTS
SDK_HEADERS: set[str] = {"x-aleo-sdk-version", "x-aleo-environment", "x-aleo-method"}


def package_version() -> str:
    from importlib.metadata import version

    # "aleo" is the pre-0.2 distribution name; keep it as a fallback so
    # older installs still report their real version.
    for dist in ("aleo-sdk", "aleo"):
        try:
            return version(dist)
        except Exception:
            continue
    return "0.0.0"


def user_agent() -> str:
    """The SDK's ``User-Agent`` string, sent on every call.

    Identifies the Python SDK (and its version) in the standard, always-logged
    header, overriding the underlying ``python-requests`` / ``python-httpx``
    default.  Only injected on the default transport (see :func:`method_headers`
    and the scanners' header builders); when the caller supplies their own
    transport they own the headers, so the SDK does not set it.
    """
    return f"aleo-python-sdk/{package_version()}"


def make_default_headers() -> dict[str, str]:
    return {
        "X-Aleo-SDK-Version": package_version(),
        "X-Aleo-environment": "python",
    }


def user_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return only non-SDK headers (for custom transport mode)."""
    return {k: v for k, v in headers.items() if k.lower() not in SDK_HEADERS}


def method_headers(
    headers: dict[str, str],
    method: str,
    has_custom_transport: bool,
) -> dict[str, str]:
    if has_custom_transport:
        return user_headers(headers)
    return {**headers, "X-ALEO-METHOD": method, "User-Agent": user_agent()}


def jwt_origin(host: str) -> str:
    """Derive the JWT origin from a host URL (scheme+host+port, no path)."""
    parsed = urlparse(host)
    return f"{parsed.scheme}://{parsed.netloc}"


def now_ms() -> int:
    return int(time.time() * 1000)


def jwt_expired(jwt_data: dict[str, Any]) -> bool:
    """Return True if jwt_data is missing or expiring within 5 minutes."""
    if not jwt_data:
        return True
    exp: int = jwt_data.get("expiration", 0)
    return now_ms() >= exp - FIVE_MINUTES_MS


def validate_block_range(start: int, end: int) -> None:
    if start < 0:
        raise ValueError("start must be >= 0")
    if start > end:
        raise ValueError("start must be <= end")
    if end - start > 50:
        raise ValueError("Block range cannot exceed 50 blocks")


def strip_quotes(s: str) -> str:
    """Strip quote characters (TS semantics for deployment tx IDs)."""
    return s.replace('"', "")


def retry_with_backoff(
    fn: Callable[[], _T],
    *,
    attempts: int = 5,
    base_delay: float = 0.1,
) -> _T:
    """Retry fn() up to `attempts` times on AleoNetworkError with status>=500."""
    last_err: Exception | None = None
    for n in range(attempts):
        try:
            return fn()
        except AleoNetworkError as exc:
            if exc.status is not None and exc.status >= 500:
                last_err = exc
                delay = base_delay * (2 ** n) + random.uniform(0, 0.05)
                time.sleep(delay)
            else:
                raise
    raise last_err  # type: ignore[misc]


async def async_retry_with_backoff(
    fn: Callable[[], Any],
    *,
    attempts: int = 5,
    base_delay: float = 0.1,
) -> Any:
    """Async version of retry_with_backoff."""
    import asyncio
    last_err: Exception | None = None
    for n in range(attempts):
        try:
            return await fn()
        except AleoNetworkError as exc:
            if exc.status is not None and exc.status >= 500:
                last_err = exc
                delay = base_delay * (2 ** n) + random.uniform(0, 0.05)
                await asyncio.sleep(delay)
            else:
                raise
    raise last_err  # type: ignore[misc]
