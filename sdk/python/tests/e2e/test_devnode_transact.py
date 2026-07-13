"""Devnode e2e: a public transfer confirms and credits the recipient.

Requires the ``aleo-devnode`` binary (the ``devnode`` fixture skips otherwise).
Real local proving downloads SNARK parameters on first run (minutes) — expected
for ``@pytest.mark.devnode``.
"""
from __future__ import annotations

import pytest

from aleo._client_common import AleoNetworkError
from aleo.testing import Devnode

from .conftest import DEVNODE_OVERPAY_BASE_FEE, skip_on_devnode_skew


@pytest.mark.devnode
def test_transact_transfer_public(devnode: Devnode) -> None:
    """transfer_public of 1 microcredit lands and the recipient balance reflects it."""
    aleo = devnode.aleo
    sender = devnode.accounts[0]
    recipient = aleo.account.create()

    try:
        tx = (
            aleo.programs.get("credits.aleo")
            .functions.transfer_public(str(recipient.address), 1)
            .transact(sender, base_fee=DEVNODE_OVERPAY_BASE_FEE)
        )
    except AleoNetworkError as exc:
        skip_on_devnode_skew(exc)
        raise  # unreachable — skip_on_devnode_skew re-raises non-skew errors

    devnode.advance(1)  # mine the block containing the tx (manual mining)
    aleo.network.wait_for_transaction(tx)

    assert aleo.get_balance(str(recipient.address)) >= 1
