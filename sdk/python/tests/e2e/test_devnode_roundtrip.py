"""Devnode e2e: public -> private -> scan -> private -> scan roundtrip.

Mints a private credits record (``transfer_public_to_private``), finds it with a
:class:`~aleo.testing.local_scanner.LocalRecordScanner`, spends it
(``transfer_private``), and re-scans to confirm the original record's tag now
resolves (spent) while a new unspent private record exists.

The devnode has no delegated proving service, so ``.transact()`` proves locally
(self-pay).  First run downloads SNARK parameters (minutes) — expected for
``@pytest.mark.devnode``.  The ``devnode`` fixture skips when the binary is
absent.
"""
from __future__ import annotations

import pytest

from aleo._client_common import AleoNetworkError
from aleo.testing import Devnode, LocalRecordScanner

from .conftest import DEVNODE_OVERPAY_BASE_FEE, skip_on_devnode_skew


@pytest.mark.devnode
def test_public_private_roundtrip(devnode: Devnode) -> None:
    """A private record is found, spent, and its spend is observable via tags."""
    aleo = devnode.aleo
    sender = devnode.accounts[0]
    aleo.default_account = sender
    credits = aleo.programs.get("credits.aleo")

    # 1) Mint a private credits record owned by the sender.
    try:
        mint_tx = (
            credits.functions.transfer_public_to_private(str(sender.address), 100_000)
            .transact(sender, base_fee=DEVNODE_OVERPAY_BASE_FEE)
        )
    except AleoNetworkError as exc:
        skip_on_devnode_skew(exc)
        raise  # unreachable
    devnode.advance(1)
    aleo.network.wait_for_transaction(mint_tx)

    # 2) The scanner finds the new unspent private credits record.
    scanner = LocalRecordScanner(aleo, sender)
    rec = scanner.get_unspent(program="credits.aleo", record="credits")
    assert rec is not None, "scanner did not find the minted private record"
    original_nonce = str(rec.nonce)

    # 3) Spend that record with a private transfer back to self.
    try:
        spend_tx = (
            credits.functions.transfer_private(rec, str(sender.address), 1)
            .transact(sender, base_fee=DEVNODE_OVERPAY_BASE_FEE)
        )
    except AleoNetworkError as exc:
        skip_on_devnode_skew(exc)
        raise  # unreachable
    devnode.advance(1)
    aleo.network.wait_for_transaction(spend_tx)

    # 4) Re-scan: the original record's tag now resolves (spent), and a fresh
    #    unspent private record (different nonce) exists.
    new_rec = scanner.get_unspent(
        program="credits.aleo",
        record="credits",
        exclude_nonces=(original_nonce,),
    )
    assert new_rec is not None, "scanner did not find the change record"
    assert str(new_rec.nonce) != original_nonce

    # The originally-found record must now be reported as spent.
    still_unspent_nonces = {str(r.nonce) for r in scanner.find(program="credits.aleo")}
    assert original_nonce not in still_unspent_nonces
