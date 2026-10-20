"""Freeze-list proofs against the REAL mainnet: the USDCx list's root and published tree, the
SDK's automatic resolution, and — when the operator's shell enables it — a USDCx unshield that
submits a transaction with a proof the SDK built.  The private xReserve burn through ``execute``
is covered by ``test_lifecycle_live.py::test_aleo_xreserve`` (the same resolver fills its proof).

Read tier:  ``BRIDGE_LIVE_READS=1`` (+ ``BRIDGE_PRIVATE_KEY``).
Funded tier: the mainnet gates of ``config.py`` with the case ``aleo-usdcx-unshield`` listed, plus
``BRIDGE_LIVE_ALEO_SCANNER=1`` (record selection shares the view key with the hosted scanner).
"""
from __future__ import annotations

import os

import pytest

from aleo.compliance import EMPTY_MERKLE_PROOF_PAIR, EMPTY_TREE_ROOT, exclusion_proof_from_tree

from . import config as live_config, helpers as live_helpers

pytestmark = pytest.mark.live

TOKEN = "usdcx_stablecoin.aleo"
LIST = "usdcx_freezelist.aleo"
CASE = "aleo-usdcx-unshield"
UNSHIELD_AMOUNT = "0.1"


def test_published_tree_matches_the_on_chain_root(live_bridge):
    tree = live_bridge.aleo.network.get_program_freeze_list(LIST)
    assert tree is not None and len(tree) >= 3
    root = live_bridge.freezelist._proofs.root(LIST)
    assert tree[-1] == root
    proof = exclusion_proof_from_tree(tree, live_bridge.aleo_address(), root, "mainnet")
    assert proof.count("leaf_index") == 2
    if root == EMPTY_TREE_ROOT:
        assert proof == EMPTY_MERKLE_PROOF_PAIR
    assert live_bridge.aleo.network.get_program_freeze_list("credits.aleo") is None


def test_exclusion_proof_resolves_through_the_token_program_and_caches(live_bridge):
    fl = live_bridge.freezelist
    fl.clear_cache()
    assert fl.freeze_list_program(TOKEN) == LIST
    first = fl.exclusion_proof(live_bridge.aleo_address(), TOKEN)
    assert first == fl.exclusion_proof(live_bridge.aleo_address(), LIST)
    entry = fl._proofs._cache.current("mainnet", LIST, fl._proofs.root(LIST))
    assert entry is not None and entry.tree[-1] == entry.root
    # The mapping walk agrees with the published tree on the root.
    assert fl.tree(TOKEN)[-1] == entry.root


def test_usdcx_unshield_with_an_sdk_built_proof(record_property):
    if not live_config.mainnet_case_enabled(CASE):
        pytest.skip(f"enable with the mainnet gates and {live_config.MAINNET_CASES_VAR}={CASE}")
    if live_config.value("BRIDGE_LIVE_ALEO_SCANNER") != "1":
        pytest.skip("record selection needs the hosted scanner; set BRIDGE_LIVE_ALEO_SCANNER=1 to share the view key")
    bridge = live_helpers.build_bridge("mainnet")
    bridge.aleo.records.register(bridge.aleo.default_account)
    call = bridge.unshield("aleo/usdcx", amount=UNSHIELD_AMOUNT)
    assert call.function_name == "transfer_private_to_public"
    proof = call.inputs[-1]
    assert proof == bridge.freezelist.exclusion_proof(bridge.aleo_address(), TOKEN)
    record_property("proof", proof[:80])
    if not live_config.mainnet_execution_enabled():
        print(f"  rehearsal  {UNSHIELD_AMOUNT} USDCx unshield prepared with proof {proof[:40]}…; not submitted")
        return
    before = bridge.mapping_value(TOKEN, "balances", bridge.aleo_address())
    receipt = call.delegate()
    record_property("transaction_id", receipt.transaction_id)
    print(f"  submitted  {receipt.transaction_id}")
    after = bridge.mapping_value(TOKEN, "balances", bridge.aleo_address())
    print(f"  balance    public USDCx {before} → {after}")
    assert receipt.transaction_id.startswith("at1")
