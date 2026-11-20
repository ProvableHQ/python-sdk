"""Request attribution: the ``X-Aleo-Bridge`` header carried by the Aleo calls of a bridge action.

Each public ``Bridge`` action runs inside :func:`bridge_context`, which labels every request the
``aleo`` SDK's network client makes on its behalf (reads, delegated proving, broadcast) with the SDK
version, the action, the protocol, the route and a random per-transfer identifier. The label lets a
Provable gateway attribute requests; it carries no key, address or amount. EVM, Solana, Circle,
Verulink and record-scanner requests are not labelled.
"""
from __future__ import annotations

import contextlib
import contextvars
import re
import secrets
from collections.abc import Generator
from typing import Any

BRIDGE_CONTEXT_HEADER = "X-Aleo-Bridge"

_TRANSFER_ID = re.compile(r"[0-9a-f]{16}")
_UNSAFE = re.compile(r"[^\x21-\x7e]|;")
_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar("aleo_bridge_context_active", default=False)


def new_transfer_id() -> str:
    return secrets.token_hex(8)


def is_transfer_id(value: Any) -> bool:
    return isinstance(value, str) and _TRANSFER_ID.fullmatch(value) is not None


def _token(value: str) -> str:
    return _UNSAFE.sub("_", value)


def context_header(action: str, *, protocol: str | None = None, route_id: str | None = None,
                   transfer_id: str | None = None) -> str:
    from . import __version__
    parts = [f"sdk=aleo-bridge-python/{__version__}", f"action={_token(action)}"]
    if protocol:
        parts.append(f"protocol={_token(protocol)}")
    if route_id:
        parts.append(f"route={_token(route_id)}")
    if is_transfer_id(transfer_id):
        parts.append(f"transfer={transfer_id}")
    return "; ".join(parts)


@contextlib.contextmanager
def bridge_context(action: str, plan: Any = None, *, protocol: str | None = None,
                   route_id: str | None = None, transfer_id: str | None = None) -> Generator[None, None, None]:
    if _ACTIVE.get():
        yield
        return
    try:
        from aleo import request_headers
    except ImportError:
        yield
        return
    if plan is not None:
        protocol = getattr(plan, "protocol", protocol)
        route_id = getattr(plan, "route_id", route_id)
        transfer_id = getattr(plan, "transfer_id", transfer_id)
    header = context_header(action, protocol=protocol, route_id=route_id, transfer_id=transfer_id)
    token = _ACTIVE.set(True)
    try:
        with request_headers({BRIDGE_CONTEXT_HEADER: header}):
            yield
    finally:
        _ACTIVE.reset(token)
