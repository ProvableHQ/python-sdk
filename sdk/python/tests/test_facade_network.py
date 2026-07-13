"""Tests for F3 facade: aleo.network module.

All tests are mocked — no live network required.  The ``responses`` library
intercepts HTTP calls made by the underlying AleoNetworkClient.
"""
from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import patch, MagicMock

import pytest
import responses as resp_lib

from aleo import Aleo, HTTPProvider
from aleo.facade.network import NetworkModule
from aleo.facade.errors import (
    TransactionConfirmationTimeout,
    TransactionNotFound,
    AleoNetworkError,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE = "https://api.provable.com/v2"
NET = "mainnet"
HOST = f"{BASE}/{NET}"
TX_ID = "at1abc123"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_client(network: str = "mainnet") -> Aleo:
    return Aleo(HTTPProvider(BASE, network=network))


# ---------------------------------------------------------------------------
# Module attachment
# ---------------------------------------------------------------------------


def test_network_module_attached() -> None:
    """aleo.network is a NetworkModule instance."""
    a = make_client()
    assert isinstance(a.network, NetworkModule)


def test_network_module_same_instance() -> None:
    """aleo.network returns the same object on every access."""
    a = make_client()
    assert a.network is a.network


def test_network_module_repr() -> None:
    a = make_client()
    r = repr(a.network)
    assert "NetworkModule" in r
    assert "mainnet" in r


def test_network_module_repr_testnet() -> None:
    a = make_client(network="testnet")
    r = repr(a.network)
    assert "testnet" in r


# ---------------------------------------------------------------------------
# Block read pass-throughs
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_latest_height() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/height/latest", json=9999)
    a = make_client()
    assert a.network.get_latest_height() == 9999
    assert "/block/height/latest" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_latest_block() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/latest", json={"height": 100})
    a = make_client()
    result = a.network.get_latest_block()
    assert result["height"] == 100
    assert "/block/latest" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_block() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/42", json={"height": 42})
    a = make_client()
    result = a.network.get_block(42)
    assert result["height"] == 42
    assert "/block/42" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_block_by_hash() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/abcdef", json={"height": 7})
    a = make_client()
    result = a.network.get_block_by_hash("abcdef")
    assert result["height"] == 7
    assert "/block/abcdef" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_block_range() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/blocks", json=[{"height": 0}, {"height": 1}])
    a = make_client()
    result = a.network.get_block_range(0, 1)
    assert len(result) == 2
    url = resp_lib.calls[0].request.url
    assert "start=0" in url
    assert "end=1" in url


@resp_lib.activate
def test_get_latest_block_hash() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/block/hash/latest", json="hash1abc")
    a = make_client()
    result = a.network.get_latest_block_hash()
    assert result == "hash1abc"
    assert "/block/hash/latest" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_latest_committee() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/committee/latest", json={"members": ["aleo1abc"]})
    a = make_client()
    result = a.network.get_latest_committee()
    assert "members" in result
    assert "/committee/latest" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_committee_by_height() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/committee/100", json={"members": []})
    a = make_client()
    result = a.network.get_committee_by_height(100)
    assert result == {"members": []}
    assert "/committee/100" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_state_root() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/stateRoot/latest", json="sr1abc")
    a = make_client()
    result = a.network.get_state_root()
    assert result == "sr1abc"
    assert "/stateRoot/latest" in resp_lib.calls[0].request.url


# ---------------------------------------------------------------------------
# Program read pass-throughs
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_program() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/program/credits.aleo", json="program credits.aleo;")
    a = make_client()
    result = a.network.get_program("credits.aleo")
    assert "credits.aleo" in result
    assert "/program/credits.aleo" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_program_with_edition() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/program/credits.aleo/3", json="program credits.aleo;")
    a = make_client()
    result = a.network.get_program("credits.aleo", edition=3)
    assert "credits.aleo" in result
    assert "/program/credits.aleo/3" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_latest_program_edition() -> None:
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/program/credits.aleo/latest_edition",
        body=b"3",
        content_type="text/plain",
    )
    a = make_client()
    result = a.network.get_latest_program_edition("credits.aleo")
    assert result == 3
    assert "/program/credits.aleo/latest_edition" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_program_amendment_count() -> None:
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/program/credits.aleo/amendment_count",
        body=b"2",
        content_type="text/plain",
    )
    a = make_client()
    result = a.network.get_program_amendment_count("credits.aleo")
    assert result == 2
    assert "/program/credits.aleo/amendment_count" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_program_mapping_names() -> None:
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/program/credits.aleo/mappings",
        json=["account", "committee"],
    )
    a = make_client()
    result = a.network.get_program_mapping_names("credits.aleo")
    assert "account" in result
    assert "/program/credits.aleo/mappings" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_program_mapping_value() -> None:
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/program/credits.aleo/mapping/account/aleo1abc",
        json="500u64",
    )
    a = make_client()
    result = a.network.get_program_mapping_value("credits.aleo", "account", "aleo1abc")
    assert result == "500u64"
    assert "/program/credits.aleo/mapping/account/aleo1abc" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_public_balance() -> None:
    # The underlying client calls int() on the raw mapping value directly,
    # so mock the value as a plain integer string (as the live API returns).
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/program/credits.aleo/mapping/account/aleo1abc",
        json="1000000",
    )
    a = make_client()
    result = a.network.get_public_balance("aleo1abc")
    assert isinstance(result, int)
    assert result == 1000000


# ---------------------------------------------------------------------------
# Transaction read pass-throughs
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_transaction() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/transaction/{TX_ID}", json={"id": TX_ID})
    a = make_client()
    result = a.network.get_transaction(TX_ID)
    assert result["id"] == TX_ID
    assert f"/transaction/{TX_ID}" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_confirmed_transaction() -> None:
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/transaction/confirmed/{TX_ID}",
        json={"id": TX_ID, "status": "accepted"},
    )
    a = make_client()
    result = a.network.get_confirmed_transaction(TX_ID)
    assert result["status"] == "accepted"
    assert f"/transaction/confirmed/{TX_ID}" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_transaction_missing_raises_not_found() -> None:
    resp_lib.add(resp_lib.GET, f"{HOST}/transaction/{TX_ID}", status=404)
    a = make_client()
    with pytest.raises(TransactionNotFound) as exc_info:
        a.network.get_transaction(TX_ID)
    assert exc_info.value.tx_id == TX_ID


@resp_lib.activate
def test_get_confirmed_transaction_missing_raises_not_found() -> None:
    resp_lib.add(
        resp_lib.GET, f"{HOST}/transaction/confirmed/{TX_ID}", status=404
    )
    a = make_client()
    with pytest.raises(TransactionNotFound) as exc_info:
        a.network.get_confirmed_transaction(TX_ID)
    assert exc_info.value.tx_id == TX_ID


@resp_lib.activate
def test_get_transaction_object_pass_through() -> None:
    """get_transaction_object returns a Transaction network object via NetworkModule."""
    from unittest.mock import patch, MagicMock

    a = make_client()
    fake_tx_obj = MagicMock()
    # Patch the underlying network client's method
    with patch.object(a._client, "get_transaction_object", return_value=fake_tx_obj) as mock_method:
        result = a.network.get_transaction_object(TX_ID)
    assert result is fake_tx_obj
    mock_method.assert_called_once_with(TX_ID)


@resp_lib.activate
def test_get_transaction_object_404_raises_not_found() -> None:
    """get_transaction_object on a 404 raises TransactionNotFound."""
    from aleo._client_common import AleoNetworkError
    from unittest.mock import patch

    a = make_client()
    with patch.object(
        a._client,
        "get_transaction_object",
        side_effect=AleoNetworkError("not found", status=404),
    ):
        with pytest.raises(TransactionNotFound) as exc_info:
            a.network.get_transaction_object(TX_ID)
    assert exc_info.value.tx_id == TX_ID


@resp_lib.activate
def test_get_transactions() -> None:
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/block/42/transactions",
        json=[{"id": "tx1"}, {"id": "tx2"}],
    )
    a = make_client()
    result = a.network.get_transactions(42)
    assert len(result) == 2
    assert "/block/42/transactions" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_transactions_in_mempool() -> None:
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/memoryPool/transactions",
        json=[{"id": "pending1"}],
    )
    a = make_client()
    result = a.network.get_transactions_in_mempool()
    assert result[0]["id"] == "pending1"
    assert "/memoryPool/transactions" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_transition_id() -> None:
    input_id = "input1abc"
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/find/transitionID/{input_id}",
        json="transition1xyz",
    )
    a = make_client()
    result = a.network.get_transition_id(input_id)
    assert "transition1xyz" in result
    assert f"/find/transitionID/{input_id}" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_deployment_transaction_id_for_program() -> None:
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/find/transactionID/deployment/hello.aleo",
        json='"at1deploy123"',
    )
    a = make_client()
    result = a.network.get_deployment_transaction_id_for_program("hello.aleo")
    # Quotes are stripped by the underlying client
    assert '"' not in result
    assert "at1deploy123" in result
    assert "/find/transactionID/deployment/hello.aleo" in resp_lib.calls[0].request.url


@resp_lib.activate
def test_get_deployment_transaction_for_program() -> None:
    # First call fetches deployment tx id, second fetches the tx itself
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/find/transactionID/deployment/hello.aleo",
        json='"at1deploy123"',
    )
    resp_lib.add(
        resp_lib.GET,
        f"{HOST}/transaction/at1deploy123",
        json={"id": "at1deploy123", "type": "deploy"},
    )
    a = make_client()
    result = a.network.get_deployment_transaction_for_program("hello.aleo")
    assert result["type"] == "deploy"


# ---------------------------------------------------------------------------
# submit_transaction — accepts object and string
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_submit_transaction_string() -> None:
    """submit_transaction accepts a raw JSON string."""
    resp_lib.add(
        resp_lib.POST,
        f"{HOST}/transaction/broadcast",
        json=TX_ID,
    )
    a = make_client()
    result = a.network.submit_transaction('{"type":"execute"}')
    assert result == TX_ID


@resp_lib.activate
def test_submit_transaction_object() -> None:
    """submit_transaction accepts an object with __str__ serialisation."""

    class FakeTx:
        def __str__(self) -> str:
            return '{"type":"execute"}'

    resp_lib.add(
        resp_lib.POST,
        f"{HOST}/transaction/broadcast",
        json=TX_ID,
    )
    a = make_client()
    result = a.network.submit_transaction(FakeTx())
    assert result == TX_ID


# ---------------------------------------------------------------------------
# send_raw_transaction alias
# ---------------------------------------------------------------------------


def test_send_raw_transaction_is_submit_transaction() -> None:
    """send_raw_transaction shares the same underlying function as submit_transaction.

    Bound methods are distinct objects per access on an instance, so we compare
    their underlying function objects (``__func__``) to assert they are aliases.
    """
    a = make_client()
    # Bound methods wrap the same function object — compare __func__
    assert a.network.send_raw_transaction.__func__ is a.network.submit_transaction.__func__  # type: ignore[attr-defined]


@resp_lib.activate
def test_send_raw_transaction_behavior() -> None:
    """send_raw_transaction produces identical behavior to submit_transaction."""
    resp_lib.add(
        resp_lib.POST,
        f"{HOST}/transaction/broadcast",
        json=TX_ID,
    )
    a = make_client()
    result = a.network.send_raw_transaction('{"type":"execute"}')
    assert result == TX_ID


# ---------------------------------------------------------------------------
# wait_for_transaction — confirmed + timeout
# ---------------------------------------------------------------------------


def test_wait_for_transaction_resolves_on_confirmed() -> None:
    """wait_for_transaction returns data when the tx is confirmed."""
    confirmed_data = {"id": TX_ID, "status": "accepted"}

    # Use a transport mock so we don't need a live network
    call_count = 0

    def fake_transport(method: str, url: str, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        import requests as _req
        r = _req.Response()
        r.status_code = 200
        r._content = json.dumps(confirmed_data).encode()
        return r

    from aleo.network_client import AleoNetworkClient
    from aleo.facade.provider import HTTPProvider as HP

    # Build the client manually with a transport so no real HTTP happens
    provider = HP(BASE, network=NET)
    aleo = Aleo(provider)
    # Inject the fake transport into the underlying network client
    aleo._client._transport = fake_transport
    aleo._client._has_custom_transport = True

    with patch("time.sleep"):
        result = aleo.network.wait_for_transaction(TX_ID, timeout=10.0, poll_interval=0.1)
    assert result["status"] == "accepted"


def test_wait_for_transaction_raises_timeout() -> None:
    """wait_for_transaction raises TransactionConfirmationTimeout when wait expires."""
    # Return 404 every poll so it never confirms
    import requests as _req

    def fake_transport(method: str, url: str, **kwargs: Any) -> _req.Response:
        r = _req.Response()
        r.status_code = 404
        r._content = b"Not Found"
        return r

    from aleo.facade.provider import HTTPProvider as HP

    provider = HP(BASE, network=NET)
    aleo = Aleo(provider)
    aleo._client._transport = fake_transport
    aleo._client._has_custom_transport = True

    with patch("time.sleep"):
        # Patch monotonic so we immediately exceed timeout
        start = time.monotonic()
        call_count = [0]

        def fast_monotonic() -> float:
            call_count[0] += 1
            # On first call return start; thereafter return start + timeout + 1
            if call_count[0] <= 1:
                return start
            return start + 999.0

        with patch("time.monotonic", side_effect=fast_monotonic):
            with pytest.raises(TransactionConfirmationTimeout) as exc_info:
                aleo.network.wait_for_transaction(TX_ID, timeout=5.0, poll_interval=0.1)

    assert exc_info.value.tx_id == TX_ID
    assert exc_info.value.timeout == 5.0


def test_wait_for_transaction_polls_until_accepted() -> None:
    """wait_for_transaction polls multiple times before confirming."""
    confirmed_data = {"id": TX_ID, "status": "accepted"}
    attempt = [0]

    def fake_transport(method: str, url: str, **kwargs: Any) -> MagicMock:
        import requests as _req
        r = _req.Response()
        attempt[0] += 1
        if attempt[0] < 3:
            # First 2 polls: not yet confirmed (404)
            r.status_code = 404
            r._content = b"Not Found"
        else:
            r.status_code = 200
            r._content = json.dumps(confirmed_data).encode()
        return r

    from aleo.facade.provider import HTTPProvider as HP

    provider = HP(BASE, network=NET)
    aleo = Aleo(provider)
    aleo._client._transport = fake_transport
    aleo._client._has_custom_transport = True

    with patch("time.sleep"):
        result = aleo.network.wait_for_transaction(TX_ID, timeout=30.0, poll_interval=0.1)
    assert result["status"] == "accepted"
    assert attempt[0] == 3
