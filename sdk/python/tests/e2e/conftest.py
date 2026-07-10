"""Fixtures for devnode end-to-end tests.

The ``devnode`` fixture is **function-scoped** so every test gets an isolated,
freshly-mined ledger on an auto-allocated port (safe under ``pytest-xdist``).
When the ``aleo-devnode`` binary is absent, :meth:`Devnode.start` raises
:class:`~aleo.testing.devnode.DevnodeError`; the fixture catches it and
``pytest.skip``s so the suite collects and skips cleanly in CI without a binary.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from aleo._client_common import AleoNetworkError
from aleo.testing import Devnode
from aleo.testing.devnode import DevnodeError

# Substrings that mark a *node-side incompatibility* rather than a bug in the
# code under test: the installed ``aleo-devnode`` binary enforces a consensus
# fee schedule / record version that the SDK's bundled snarkVM does not match
# (version skew).  When a broadcast is rejected for one of these reasons the
# test is skipped — the scanner / verb ladder is behaving correctly, the node is
# simply from a different snarkVM era.
_SKEW_MARKERS: tuple[str, ...] = (
    "insufficient base fee",       # SDK execution_cost < node's required base fee
    "must be Version 0",           # record version predates the node's consensus
    "Consensus V",                 # any consensus-version gate
)


def skip_on_devnode_skew(exc: AleoNetworkError) -> None:
    """``pytest.skip`` if *exc* is an SDK/devnode version-skew rejection; else re-raise."""
    msg = str(exc)
    if any(marker in msg for marker in _SKEW_MARKERS):
        pytest.skip(f"installed aleo-devnode is incompatible with the SDK snarkVM: {msg}")
    raise exc


@pytest.fixture(scope="function")
def devnode() -> Iterator[Devnode]:
    """Yield a started, per-test :class:`Devnode`; skip if the binary is absent."""
    try:
        node = Devnode().start()
    except DevnodeError as exc:
        pytest.skip(f"aleo-devnode binary not available: {exc}")
    try:
        yield node
    finally:
        node.stop()
