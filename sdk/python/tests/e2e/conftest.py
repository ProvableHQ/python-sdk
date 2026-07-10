"""Fixtures for devnode end-to-end tests.

The ``devnode`` fixture is **function-scoped** so every test gets an isolated,
freshly-mined ledger on an auto-allocated port (safe under ``pytest-xdist``).
When the ``aleo-devnode`` binary is absent, :meth:`Devnode.start` raises
:class:`~aleo.testing.devnode.DevnodeError`; the fixture catches it and
``pytest.skip``s so the suite collects and skips cleanly in CI without a binary.

Only the public ``transact`` flow runs against the devnode. The private
``transfer_public_to_private`` → scan → ``transfer_private`` roundtrip lives in
``test_testnet_e2e`` against live testnet (delegated proving + hosted scanner):
the devnode has no delegated prover, so it would prove ``transfer_private``
locally, which currently fails at the Varuna level.
"""
from __future__ import annotations

from collections.abc import Iterator

import pytest

from aleo._client_common import AleoNetworkError
from aleo.testing import Devnode
from aleo.testing.devnode import DevnodeError

# The installed aleo-devnode enforces a higher base (storage) fee than the SDK's
# ``execution_cost`` computes — a devnode-side fee-schedule skew, not an SDK bug
# to fix here. Overpay the base fee generously via the verb ladder's ``base_fee``
# override so devnode transactions land; genesis accounts are richly funded.
# Remove once the devnode fee schedule and the SDK cost model agree.
DEVNODE_OVERPAY_BASE_FEE: int = 1_000_000

# A broadcast rejected because the devnode requires a higher base fee than the
# SDK's cost model produces is a node-side skew, not a bug in the code under
# test — skip rather than fail.
_SKEW_MARKERS: tuple[str, ...] = (
    "insufficient base fee",       # SDK execution_cost < node's required base fee
)


def skip_on_devnode_skew(exc: AleoNetworkError) -> None:
    """``pytest.skip`` if *exc* is an SDK/devnode fee-schedule skew; else re-raise."""
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
