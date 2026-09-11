"""Every READ method of AleoNetworkClient / AsyncAleoNetworkClient against the
live hosted API (default: the open edge, ``ALEO_E2E_ENDPOINT`` overrides).

``test_testnet_objects_live`` covers the handful of reads the facade leans on;
this module sweeps the rest — blocks by height and hash, committee, mempool,
transactions and their confirmed/transition views, program metadata
(imports, mappings, editions, amendments, deployment tx), balances and state
paths — and asserts the async client returns the same answers.  Read-only,
no credentials, no funded key.  Module-level ``live`` marker.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import pytest

from aleo import AleoNetworkError
from aleo.async_network_client import AsyncAleoNetworkClient
from aleo.network_client import AleoNetworkClient

pytestmark = pytest.mark.live

_ENDPOINT = os.environ.get("ALEO_E2E_ENDPOINT", "https://edge.provable.com/api")
_NETWORK = "testnet"
PROGRAM = "shield_swap.aleo"          # deployed, has imports and mappings


@pytest.fixture(scope="module")
def sync_client() -> AleoNetworkClient:
    c = AleoNetworkClient(_ENDPOINT, network=_NETWORK)
    try:
        c.get_latest_height()
    except Exception as exc:  # noqa: BLE001 - offline CI must skip, not error
        pytest.skip(f"{_ENDPOINT} unreachable: {exc}")
    return c


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _find(obj: Any, pred: Any) -> Any:
    """First leaf in a nested JSON structure satisfying *pred*."""
    if pred(obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            hit = _find(v, pred)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find(v, pred)
            if hit is not None:
                return hit
    return None


@pytest.fixture(scope="module")
def busy_block(sync_client: AleoNetworkClient) -> tuple[int, dict[str, Any]]:
    """A recent block that carries at least one transaction."""
    height = sync_client.get_latest_height()
    for h in range(height, max(height - 200, 0), -1):
        block = sync_client.get_block(h)
        if block.get("transactions"):
            return h, block
    pytest.skip("no block with transactions in the last 200 heights")


def test_blocks_by_height_and_hash_agree(sync_client: AleoNetworkClient, busy_block):
    height, block = busy_block
    assert block["header"]["metadata"]["height"] == height
    by_hash = sync_client.get_block_by_hash(block["block_hash"])
    assert by_hash["block_hash"] == block["block_hash"]
    latest = sync_client.get_latest_block()
    assert sync_client.get_latest_block_hash() in (latest["block_hash"],
                                                   sync_client.get_latest_block()["block_hash"])
    committee = sync_client.get_committee_by_height(height)
    assert committee["members"], "committee at a recent height is never empty"


def test_transactions_and_their_views(sync_client: AleoNetworkClient, busy_block):
    height, block = busy_block
    txs = sync_client.get_transactions(height)
    assert txs and len(txs) == len(block["transactions"])
    tx_id = _find(txs[0], lambda v: isinstance(v, str) and v.startswith("at1"))
    assert tx_id, "no transaction id in the block's first transaction"
    tx = sync_client.get_transaction(tx_id)
    assert tx["id"] == tx_id
    confirmed = sync_client.get_confirmed_transaction(tx_id)
    assert _find(confirmed, lambda v: v == tx_id) is not None
    # An input/output id of any transition maps back to its transition id.
    transition = _find(tx, lambda v: isinstance(v, dict) and "outputs" in v and "id" in v)
    if transition is None:
        pytest.skip("first transaction carries no transitions with outputs (fee-only)")
    leaf_id = (transition["outputs"] or transition.get("inputs") or [{}])[0].get("id")
    assert leaf_id, "transition leaf without an id"
    assert sync_client.get_transition_id(leaf_id) == transition["id"]


def test_program_metadata_reads(sync_client: AleoNetworkClient):
    imports = sync_client.get_program_import_names(PROGRAM)
    assert imports and all(i.endswith(".aleo") for i in imports)
    sources = sync_client.get_program_imports(PROGRAM)
    assert set(sources) >= set(imports) - {"credits.aleo"}
    assert all(f"program {pid}" in src for pid, src in sources.items())
    mappings = sync_client.get_program_mapping_names("credits.aleo")
    assert "account" in mappings
    assert sync_client.get_latest_program_edition("credits.aleo") >= 0
    amendments = sync_client.get_program_amendment_count("credits.aleo")
    assert amendments["program_id"] == "credits.aleo" and amendments["amendment_count"] >= 0
    dep_id = sync_client.get_deployment_transaction_id_for_program(PROGRAM)
    assert dep_id.startswith("at1")
    dep = sync_client.get_deployment_transaction_for_program(PROGRAM)
    assert _find(dep, lambda v: v == dep_id) is not None
    with pytest.raises(AleoNetworkError):
        sync_client.get_program_import_names("definitely_not_a_program_xyz.aleo")


def test_balances_and_state_paths(sync_client: AleoNetworkClient, busy_block):
    from aleo.testnet import PrivateKey
    nobody = str(PrivateKey.random().address)
    assert sync_client.get_public_balance(nobody) == 0
    _, block = busy_block
    # A record output's id is its commitment — the input to a state-path proof.
    record_out = _find(block, lambda v: isinstance(v, dict) and v.get("type") == "record" and "id" in v)
    if record_out is None:
        pytest.skip("no record output in the chosen block")
    try:
        paths = sync_client.get_state_paths([record_out["id"]])
    except AleoNetworkError as exc:
        if exc.status == 502:
            # Same on edge and legacy (2026-09-11): the hosted API's statePaths
            # route answers 502 — a gateway gap upstream, not a client bug.
            pytest.skip("hosted API: /statePaths returns 502 on both hosts")
        raise
    assert len(paths) == 1 and paths[0]


def test_mempool_route_status(sync_client: AleoNetworkClient):
    # The hosted API (edge AND legacy, 2026-09-11) does not expose the node's
    # /memoryPool/transactions route; surface that as a skip with the status
    # rather than a client failure, and pass if it ever appears.
    try:
        assert isinstance(sync_client.get_transactions_in_mempool(), list)
    except AleoNetworkError as exc:
        if exc.status == 404:
            pytest.skip("hosted API: /memoryPool/transactions is not served (404)")
        raise


def test_async_client_answers_match_sync(sync_client: AleoNetworkClient, busy_block):
    height, block = busy_block

    async def go() -> dict[str, Any]:
        a = AsyncAleoNetworkClient(_ENDPOINT, network=_NETWORK)
        try:
            return {
                "block_hash": (await a.get_block(height))["block_hash"],
                "by_hash": (await a.get_block_by_hash(block["block_hash"]))["block_hash"],
                "n_txs": len(await a.get_transactions(height)),
                "committee": sorted((await a.get_committee_by_height(height))["members"]),
                "imports": await a.get_program_import_names(PROGRAM),
                "mappings": await a.get_program_mapping_names("credits.aleo"),
                "edition": await a.get_latest_program_edition("credits.aleo"),
                "dep_id": await a.get_deployment_transaction_id_for_program(PROGRAM),
                "height_ok": (await a.get_latest_height()) >= height,
            }
        finally:
            close = getattr(a, "aclose", None) or getattr(a, "close", None)
            if close:
                result = close()
                if asyncio.iscoroutine(result):
                    await result

    got = _run(go())
    assert got["block_hash"] == block["block_hash"] == got["by_hash"]
    assert got["n_txs"] == len(block["transactions"])
    assert got["committee"] == sorted(sync_client.get_committee_by_height(height)["members"])
    assert got["imports"] == sync_client.get_program_import_names(PROGRAM)
    assert got["mappings"] == sync_client.get_program_mapping_names("credits.aleo")
    assert got["edition"] == sync_client.get_latest_program_edition("credits.aleo")
    assert got["dep_id"] == sync_client.get_deployment_transaction_id_for_program(PROGRAM)
    assert got["height_ok"]
