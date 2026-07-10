"""Devnode e2e (async): the sync transact + roundtrip tests, driven via ``AsyncAleo``.

Mirrors :mod:`test_devnode_transact` and :mod:`test_devnode_roundtrip` but runs
the verb ladder with ``await`` on an :class:`~aleo.facade.async_client.AsyncAleo`
client.  Record finding still uses the **sync** :class:`LocalRecordScanner`
(pointed at the same node URL) — the scanner does plain HTTP reads, so there is
no async scanner to build.

Requires the ``aleo-devnode`` binary (the ``devnode`` fixture skips otherwise).
Broadcasts rejected for SDK/devnode version skew (fee schedule / record version)
are skipped, not failed — see :func:`skip_on_devnode_skew`.
"""
from __future__ import annotations

import pytest

from aleo import Aleo, AsyncAleo
from aleo._client_common import AleoNetworkError
from aleo.testing import Devnode, LocalRecordScanner

from .conftest import DEVNODE_OVERPAY_BASE_FEE, skip_on_devnode_skew


def _async_client(devnode: Devnode) -> AsyncAleo:
    return AsyncAleo(AsyncAleo.HTTPProvider(devnode.base_url, network="testnet"))


def _sync_scanner(devnode: Devnode, account: object) -> LocalRecordScanner:
    """A sync scanner over a sync client bound to the same node (plain HTTP reads)."""
    sync = Aleo(Aleo.HTTPProvider(devnode.base_url, network="testnet"))
    return LocalRecordScanner(sync, account)


@pytest.mark.devnode
@pytest.mark.asyncio
async def test_async_transact_transfer_public(devnode: Devnode) -> None:
    """Async transfer_public of 1 microcredit lands and credits the recipient."""
    a = _async_client(devnode)
    sender = devnode.accounts[0]
    recipient = a.account.create()  # account ops are sync on AsyncAleo

    credits = await a.programs.get("credits.aleo")
    bound = credits.functions.transfer_public(str(recipient.address), 1)
    try:
        tx = await bound.transact(sender, base_fee=DEVNODE_OVERPAY_BASE_FEE)
    except AleoNetworkError as exc:
        skip_on_devnode_skew(exc)
        raise  # unreachable

    devnode.advance(1)  # mine the block (manual mining)
    await a.network.wait_for_transaction(tx)

    assert await a.get_balance(str(recipient.address)) >= 1


@pytest.mark.devnode
@pytest.mark.asyncio
async def test_async_roundtrip(devnode: Devnode) -> None:
    """Async public->private->scan->private->scan; scanning via the sync scanner."""
    a = _async_client(devnode)
    sender = devnode.accounts[0]
    a.default_account = sender
    credits = await a.programs.get("credits.aleo")

    # 1) Mint a private credits record owned by the sender (async verb ladder).
    mint = credits.functions.transfer_public_to_private(str(sender.address), 100_000)
    try:
        mint_tx = await mint.transact(sender, base_fee=DEVNODE_OVERPAY_BASE_FEE)
    except AleoNetworkError as exc:
        skip_on_devnode_skew(exc)
        raise  # unreachable
    devnode.advance(1)
    await a.network.wait_for_transaction(mint_tx)

    # 2) Find the new unspent private credits record with the sync scanner.
    scanner = _sync_scanner(devnode, sender)
    rec = scanner.get_unspent(program="credits.aleo", record="credits")
    assert rec is not None, "scanner did not find the minted private record"
    original_nonce = str(rec.nonce)

    # 3) Spend it with a private transfer back to self (async verb ladder).
    spend = credits.functions.transfer_private(rec, str(sender.address), 1)
    try:
        spend_tx = await spend.transact(sender, base_fee=DEVNODE_OVERPAY_BASE_FEE)
    except AleoNetworkError as exc:
        skip_on_devnode_skew(exc)
        raise  # unreachable
    devnode.advance(1)
    await a.network.wait_for_transaction(spend_tx)

    # 4) Re-scan: original tag now resolves (spent); a fresh unspent record exists.
    new_rec = scanner.get_unspent(
        program="credits.aleo",
        record="credits",
        exclude_nonces=(original_nonce,),
    )
    assert new_rec is not None, "scanner did not find the change record"
    assert str(new_rec.nonce) != original_nonce

    still_unspent = {str(r.nonce) for r in scanner.find(program="credits.aleo")}
    assert original_nonce not in still_unspent
