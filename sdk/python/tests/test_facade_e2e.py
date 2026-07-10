"""Slow end-to-end tests for the F5 facade verb ladder.

Marked ``@pytest.mark.slow`` and deselected by default (``-m "not slow"``).
They need live network access (``trace.prepare`` fetches the latest state root)
and, on first use, hundreds of MB of SNARK proving parameters — the first run
can take several minutes.  Run locally with::

    python -m pytest python/tests/test_facade_e2e.py -v -m slow

Endpoint + retry idiom mirror ``test_proving.py``.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aleo import Aleo, HTTPProvider
from aleo.mainnet import Program as RawProgram, PrivateKey
from aleo.facade.programs import Program
from aleo.facade.call import BoundCall

ENDPOINT = "https://api.explorer.provable.com/v2"

CREDITS_SOURCE = RawProgram.credits().source


def _client() -> Aleo:
    return Aleo(HTTPProvider(ENDPOINT, network="mainnet"))


def _bound(a: Aleo, acct: Any) -> BoundCall:
    prog = Program(a, RawProgram.from_source(CREDITS_SOURCE))
    bc = prog.functions.transfer_public(str(acct.address), 10)
    assert isinstance(bc, BoundCall)
    return bc


@pytest.mark.slow
def test_transact_end_to_end_returns_tx_id() -> None:
    """ONE full transact: real authorize→execute→prepare→prove→fee→submit.

    Asserts a transaction id string comes back.  Uses a random (unfunded) key,
    so the node may reject broadcast — we accept either a returned tx id or a
    network rejection, but the local proving+assembly must succeed.
    """
    import time

    from aleo.facade.errors import AleoNetworkError, ExecutionError

    a = _client()
    acct = a.account.from_private_key(PrivateKey.random())
    a.default_account = acct
    bc = _bound(a, acct)

    # Build (prove) with a small retry to absorb transient state-root 503s.
    last: Exception | None = None
    tx_result = None
    for attempt in range(3):
        try:
            tx_result = bc.build_transaction()
            break
        except ExecutionError as exc:
            last = exc
            if attempt < 2:
                time.sleep(20.0)
    assert tx_result is not None, f"build_transaction failed: {last}"

    assert tx_result.id  # a real transaction id
    assert tx_result.decoded()[0]["function"] == "transfer_public"

    # Broadcast may be rejected (unfunded signer); a tx id or a clean network
    # rejection are both acceptable outcomes for the e2e wiring.
    try:
        tx_id = bc._client.network.submit_transaction(tx_result.raw)
        assert isinstance(tx_id, str) and tx_id
    except AleoNetworkError:
        pass


@pytest.mark.slow
def test_delegate_mocked_prover_round_trip() -> None:
    """ONE delegate against a MOCKED prover: assert the request is posted."""
    a = _client()
    acct = a.account.from_private_key(PrivateKey.random())
    bc = _bound(a, acct)

    mock = MagicMock(return_value={"transaction_id": "at1e2edelegate"})
    a._client.submit_proving_request = mock  # type: ignore[attr-defined]

    out = bc.delegate(acct)
    assert out == {"transaction_id": "at1e2edelegate"}
    mock.assert_called_once()
    request = mock.call_args.args[0]
    assert request.fee_authorization() is None  # fee master pays by default
    assert str(request.authorization().function_name()) == "transfer_public"
