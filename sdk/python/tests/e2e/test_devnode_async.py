"""Devnode e2e (async): the transact test driven via ``AsyncAleo``.

Mirrors :mod:`test_devnode_transact` but runs the verb ladder with ``await`` on
an :class:`~aleo.facade.async_client.AsyncAleo` client.

The private ``transfer_public_to_private`` → scan → ``transfer_private`` roundtrip
is NOT a devnode test: the devnode has no delegated proving service, so it would
prove ``transfer_private`` locally, which currently fails at the Varuna level
(record-version/circuit skew). That roundtrip lives in :mod:`test_testnet_e2e`
against live testnet, where the DPS proves it and the hosted scanner discovers
the record.

Requires the ``aleo-devnode`` binary (the ``devnode`` fixture skips otherwise).
"""
from __future__ import annotations

import pytest

from aleo import AsyncAleo
from aleo._client_common import AleoNetworkError
from aleo.testing import Devnode

from .conftest import DEVNODE_OVERPAY_BASE_FEE, skip_on_devnode_skew


def _async_client(devnode: Devnode) -> AsyncAleo:
    return AsyncAleo(AsyncAleo.HTTPProvider(devnode.base_url, network="testnet"))


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
