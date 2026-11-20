"""Tests for request attribution: the ``X-Aleo-Bridge`` header and the transfer identity."""
from dataclasses import replace

import pytest

from aleo_bridge import __version__
from aleo_bridge._context import (BRIDGE_CONTEXT_HEADER, bridge_context, context_header, is_transfer_id,
                                  new_transfer_id)
from aleo_bridge.checkpoint import Checkpoint, create_checkpoint
from aleo_bridge.client import _checkpoint_subject
from aleo_bridge.lifecycle import prepare, quote
from aleo_bridge.registry import DEFAULT_REGISTRY
from aleo_bridge.types import Receipt, Status
from tests.fakes.fake_bridge import EVM_ADDRESS, FakeBridge

ROUTE = "hyperlane:aleo/eth->ethereum/eth"
TRANSFER = "0123456789abcdef"


def _sent() -> dict:
    return pytest.importorskip("aleo._client_common").current_request_headers()


def _plan():
    return prepare(DEFAULT_REGISTRY, source_chain="aleo", source_asset="eth", destination_chain="ethereum",
                   destination_asset="eth", amount="0.01", recipient=EVM_ADDRESS)


def test_header_leads_with_the_sdk_name_and_version():
    assert context_header("quote") == f"sdk=aleo-bridge-python/{__version__}; action=quote"


def test_header_adds_protocol_route_and_transfer_when_known():
    assert context_header("status", protocol="hyperlane", route_id=ROUTE, transfer_id=TRANSFER) == (
        f"sdk=aleo-bridge-python/{__version__}; action=status; protocol=hyperlane; route={ROUTE}; "
        f"transfer={TRANSFER}")


def test_header_drops_a_malformed_transfer_id_and_neutralises_separators():
    header = context_header("status", route_id="a;b\r\nc d", transfer_id="abc\r\nX-Injected: 1")
    assert "transfer=" not in header and "route=a_b__c_d" in header


def test_transfer_ids_are_random_16_hex():
    first, second = new_transfer_id(), new_transfer_id()
    assert is_transfer_id(first) and first != second
    assert not is_transfer_id("0123456789ABCDEF") and not is_transfer_id(None)


def test_context_labels_aleo_requests_inside_the_block_only():
    plan = replace(_plan(), transfer_id=TRANSFER)
    with bridge_context("execute", plan):
        assert _sent() == {BRIDGE_CONTEXT_HEADER: context_header(
            "execute", protocol="hyperlane", route_id=ROUTE, transfer_id=TRANSFER)}
    assert BRIDGE_CONTEXT_HEADER not in _sent()


def test_the_enclosing_action_is_kept():
    with bridge_context("wait"):
        with bridge_context("status"):
            assert _sent()[BRIDGE_CONTEXT_HEADER] == context_header("wait")


def test_quote_mints_a_transfer_id_without_changing_plan_equality():
    q = quote(FakeBridge(ethereum=False), source_chain="aleo", source_asset="eth", destination_chain="ethereum",
              destination_asset="eth", amount="0.01", recipient=EVM_ADDRESS)
    assert is_transfer_id(q.plan.transfer_id)
    assert q.plan == _plan() and "transfer_id" not in _plan().to_dict()


def test_checkpoint_carries_the_transfer_id_and_drops_a_malformed_one():
    plan = replace(_plan(), transfer_id=TRANSFER)
    receipt = Receipt(id="at1source", protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                      source_tx_id="at1source", protocol_state={"routeId": ROUTE})
    saved = create_checkpoint(plan, receipt, DEFAULT_REGISTRY).to_dict()
    assert saved["transferId"] == TRANSFER
    assert Checkpoint.from_dict(saved).transfer_id == TRANSFER
    assert Checkpoint.from_dict({**saved, "transferId": "not-an-id"}).transfer_id is None
    assert _checkpoint_subject(saved) == {"protocol": "hyperlane", "route_id": ROUTE, "transfer_id": TRANSFER}


def test_checkpoint_subject_tolerates_unreadable_input():
    assert _checkpoint_subject("not json") == {} and _checkpoint_subject(None) == {}
