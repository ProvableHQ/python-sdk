"""Compliance proof slots per entrypoint (amm-v3 ``MerkleProofReference.md`` at f6694b7),
caller-supplied bundle validation, the legacy ``wrapper_proofs`` form, and automatic
resolution through the client against empty and populated freeze lists."""
from __future__ import annotations

import pytest

from aleo import compliance as fl
from aleo.compliance import EMPTY_MERKLE_PROOF_PAIR, EMPTY_TREE_ROOT, FrozenAddressError, MerkleProofInput, MerkleProofPair
from aleo_shield_swap import ShieldSwap
from aleo_shield_swap._core import default_merkle_proofs
from aleo_shield_swap._routing import LP_ROUTER_ID, REBALANCE_ROUTER_ID, ROUTER_ID
from aleo_shield_swap.compliance import (
    SHIELD_SWAP_FREEZELIST, Slot, claim_slots, collect_slots, increase_slots, merge_legacy_wrapper_proofs,
    mint_slots, rebalance_slots, supplied_literals, swap_slots,
)

from .conftest import POOL_TEXT, SIGNER, SLOT_TEXT, StubAleo
from .test_liquidity import POSITION_TEXT, TOKEN0_RECORD

S, R, W = "aleo1signer", "aleo1recipient", "aleo1withdrawal"
T0, T1 = "1field", "2field"
LIT = EMPTY_MERKLE_PROOF_PAIR
ONE = MerkleProofInput((0,) * 16, 1)


def names(slots: list[Slot]) -> list[str]:
    return [s.name for s in slots]


def subjects(slots: list[Slot]) -> list[tuple[str, str, str | None]]:
    return [(s.kind, s.subject, s.token_id) for s in slots]


# ── Slot tables ───────────────────────────────────────────────────────────────

def test_core_functions_take_amm_bundles_only():
    assert subjects(mint_slots(False, False, S, R, W, T0, T1)) == [("amm", S, None), ("amm", R, None), ("amm", W, None)]
    assert names(mint_slots(False, False, S, R, W, T0, T1)) == [
        "signer_merkle_proofs", "recipient_merkle_proofs", "withdrawal_merkle_proofs"]
    assert subjects(collect_slots(False, False, S, W, T0, T1)) == [("amm", S, None), ("amm", W, None)]
    assert names(collect_slots(False, False, S, W, T0, T1)) == ["owner_merkle_proofs", "withdrawal_merkle_proofs"]
    assert subjects(claim_slots(False, False, False, S, T0, T1)) == [("amm", S, None)]
    assert subjects(claim_slots(False, False, True, S, T0, T1)) == [("amm", S, None)]
    assert swap_slots(False, S, T0) == [] and increase_slots(False, False, S, T0, T1) == []


def test_trading_router_table():
    assert subjects(swap_slots(True, S, T0)) == [("wrapper", S, T0)]                       # swap_from_wrapped
    assert names(claim_slots(False, False, True, S, T0, T1)) == ["merkle_proofs"]            # claim_to_arc20_no_refund
    assert subjects(claim_slots(True, False, True, S, T0, T1)) == [("amm", S, None), ("wrapper", S, T1)]  # claim_to_wrapped_no_refund
    assert names(claim_slots(True, False, True, S, T0, T1)) == ["merkle_proofs", "receiver_proof"]
    assert subjects(claim_slots(True, False, False, S, T0, T1)) == [("amm", S, None), ("wrapper", S, T1)]  # claim_to_wrapped_refund_arc20
    assert subjects(claim_slots(False, True, False, S, T0, T1)) == [("amm", S, None), ("wrapper", S, T0)]  # claim_to_arc20_refund_wrapped
    assert names(claim_slots(True, True, False, S, T0, T1)) == ["merkle_proofs", "out_receiver_proof", "refund_receiver_proof"]
    assert subjects(claim_slots(True, True, False, S, T0, T1)) == [("amm", S, None), ("wrapper", S, T1), ("wrapper", S, T0)]
    # A wrapped refund token on a no-refund claim keeps the router but takes no refund bundle.
    assert subjects(claim_slots(True, True, True, S, T0, T1)) == [("amm", S, None), ("wrapper", S, T1)]


def test_lp_router_table():
    amm3 = [("amm", S, None), ("amm", R, None), ("amm", W, None)]
    assert subjects(mint_slots(True, False, S, R, W, T0, T1)) == [("wrapper", S, T0), *amm3]               # mint_from_wrapped_arc20
    assert subjects(mint_slots(False, True, S, R, W, T0, T1)) == [("wrapper", S, T1), *amm3]               # mint_from_arc20_wrapped
    assert subjects(mint_slots(True, True, S, R, W, T0, T1)) == [("wrapper", S, T0), ("wrapper", S, T1), *amm3]
    assert names(mint_slots(True, True, S, R, W, T0, T1)) == [
        "sender_proof0", "sender_proof1", "amm_signer_merkle_proofs", "amm_recipient_merkle_proofs", "amm_withdrawal_merkle_proofs"]
    assert subjects(increase_slots(True, False, S, T0, T1)) == [("wrapper", S, T0)]
    assert subjects(increase_slots(False, True, S, T0, T1)) == [("wrapper", S, T1)]
    assert subjects(increase_slots(True, True, S, T0, T1)) == [("wrapper", S, T0), ("wrapper", S, T1)]
    owner2 = [("amm", S, None), ("amm", W, None)]
    # Receiver bundles screen the NFT's withdrawal address, not the signer.
    assert subjects(collect_slots(True, False, S, W, T0, T1)) == [*owner2, ("wrapper", W, T0)]
    assert subjects(collect_slots(False, True, S, W, T0, T1)) == [*owner2, ("wrapper", W, T1)]
    assert subjects(collect_slots(True, True, S, W, T0, T1)) == [*owner2, ("wrapper", W, T0), ("wrapper", W, T1)]
    assert names(collect_slots(True, True, S, W, T0, T1)) == [
        "amm_owner_merkle_proofs", "amm_withdrawal_merkle_proofs", "receiver_proof0", "receiver_proof1"]


@pytest.mark.parametrize("w0,w1,f0,f1,expected", [
    (False, False, False, False, []),                                            # plain_plain_none
    (False, False, True, False, []),                                             # plain_plain_one
    (False, False, True, True, []),                                              # plain_plain_both
    (True, False, False, False, [("wrapper", W, T0)]),                           # wrapped_plain_none
    (True, False, True, False, [("wrapper", S, T0), ("wrapper", W, T0)]),        # wrapped_plain_fund0
    (True, False, False, True, [("wrapper", W, T0)]),                            # wrapped_plain_fund1
    (True, False, True, True, [("wrapper", S, T0), ("wrapper", W, T0)]),         # wrapped_plain_both
    (False, True, False, False, [("wrapper", W, T1)]),                           # plain_wrapped_none
    (False, True, True, False, [("wrapper", W, T1)]),                            # plain_wrapped_fund0
    (False, True, False, True, [("wrapper", S, T1), ("wrapper", W, T1)]),        # plain_wrapped_fund1
    (False, True, True, True, [("wrapper", S, T1), ("wrapper", W, T1)]),         # plain_wrapped_both
    (True, True, False, False, [("wrapper", W, T0), ("wrapper", W, T1)]),        # wrapped_wrapped_none
    (True, True, True, False, [("wrapper", S, T0), ("wrapper", W, T0), ("wrapper", W, T1)]),   # wrapped_wrapped_one (side 0)
    (True, True, False, True, [("wrapper", S, T1), ("wrapper", W, T0), ("wrapper", W, T1)]),   # wrapped_wrapped_one (side 1)
    (True, True, True, True, [("wrapper", S, T0), ("wrapper", S, T1), ("wrapper", W, T0), ("wrapper", W, T1)]),  # wrapped_wrapped_both
])
def test_rebalance_router_table(w0, w1, f0, f1, expected):
    slots = rebalance_slots(w0, w1, f0, f1, S, W, T0, T1)
    assert subjects(slots) == [*expected, ("amm", S, None), ("amm", W, None)]
    assert names(slots)[-2:] == ["owner_proofs", "withdrawal_proofs"]
    assert 2 <= len(slots) <= 6


def test_rebalance_one_sided_wrapped_pair_names_the_single_sender_proof():
    assert names(rebalance_slots(True, True, True, False, S, W, T0, T1))[0] == "sender_proof"
    assert names(rebalance_slots(True, True, False, True, S, W, T0, T1))[0] == "sender_proof"
    assert names(rebalance_slots(True, True, True, True, S, W, T0, T1))[:2] == ["sender_proof0", "sender_proof1"]
    assert names(rebalance_slots(True, False, True, False, S, W, T0, T1))[0] == "sender_proof0"


# ── Caller-supplied bundles ──────────────────────────────────────────────────

def test_supplied_literals_accepts_every_bundle_form_in_order():
    slots = mint_slots(True, False, S, R, W, T0, T1)
    pair = MerkleProofPair(ONE, ONE)
    out = supplied_literals("mint", (LIT, pair, (ONE, ONE), LIT), slots)
    assert out == [LIT] * 4


def test_supplied_literals_count_mismatch_names_the_slots():
    with pytest.raises(ValueError, match=r"mint on this route takes 4 compliance proof bundle\(s\) \[sender_proof0 .*amm_withdrawal_merkle_proofs.*; received 3"):
        supplied_literals("mint", (LIT, LIT, LIT), mint_slots(True, False, S, R, W, T0, T1))
    with pytest.raises(ValueError, match=r"takes 0 compliance proof bundle\(s\) \[none\]; received 1"):
        supplied_literals("swap", (LIT,), [])
    with pytest.raises(ValueError, match="encoded"):
        supplied_literals("swap", ("nope",), swap_slots(True, S, T0))
    # A bare (MerkleProofInput, MerkleProofInput) tuple is one bundle, not two.
    assert supplied_literals("swap", (ONE, ONE), swap_slots(True, S, T0)) == [LIT]


def test_merge_legacy_wrapper_proofs_fills_wrapper_slots_only():
    slots = collect_slots(True, True, S, W, T0, T1)
    custom = LIT.replace("leaf_index: 1u32", "leaf_index: 3u32", 1)
    assert merge_legacy_wrapper_proofs(slots, ["A", "B", "x", "y"], custom) == ["A", "B", custom, custom]


# ── Through the client ───────────────────────────────────────────────────────

def _stub(wrapped=(), **over):
    mappings = {
        "pools": {"5field": POOL_TEXT}, "slots": {"5field": SLOT_TEXT},
        "fee_tiers": {"3000u16": "true"}, "fee_to_tick_spacing": {"3000u16": "60u32"},
        "used_blinded_addresses": {}, "from_wrapper_token_id": {t: "9field" for t in wrapped},
    }
    records = over.pop("records", [{"record_plaintext": TOKEN0_RECORD}, {"record_plaintext": POSITION_TEXT}])
    return StubAleo(mappings=mappings, records=records)


def _mint(dex, **kw):
    dex.mint(pool_key="5field", tick_lower=-4080, tick_upper=4080, amount0_desired=10**9, amount1_desired=100,
             token0_program="test_usdcx_stablecoin.aleo", token1_program="tok1.aleo", **kw)


def test_mint_resolves_every_slot_from_empty_lists_with_one_root_read_per_list():
    stub = _stub(wrapped={"1field"})
    dex = ShieldSwap(stub)
    _mint(dex)
    fn, args = stub.last_call
    assert (stub.last_program, fn) == (LP_ROUTER_ID, "mint_from_wrapped_arc20")
    assert args[2] == args[9] == args[10] == args[11] == default_merkle_proofs()
    roots = [r for r in stub.mapping_reads if r[1] == "freeze_list_root"]
    assert roots == [("test_usdcx_freezelist.aleo", "freeze_list_root", "1u8"), (SHIELD_SWAP_FREEZELIST, "freeze_list_root", "1u8")]
    assert stub.tree_reads == []                                          # empty lists never fetch the tree


def test_mint_supplied_bundles_are_used_verbatim_and_never_read_the_chain():
    stub = _stub(wrapped={"1field"})
    dex = ShieldSwap(stub)
    custom = [LIT.replace("leaf_index: 1u32", f"leaf_index: {i}u32", 1) for i in range(2, 6)]
    _mint(dex, compliance_proofs=tuple(custom))
    _, args = stub.last_call
    assert [args[2], args[9], args[10], args[11]] == custom
    assert not [r for r in stub.mapping_reads if r[1] == "freeze_list_root"]
    with pytest.raises(ValueError, match="takes 4 compliance proof bundle"):
        _mint(dex, compliance_proofs=(LIT, LIT, LIT))
    with pytest.raises(ValueError, match="not both"):
        _mint(dex, compliance_proofs=tuple(custom), wrapper_proofs=LIT)


def test_legacy_wrapper_proofs_fill_wrapper_slots_while_amm_slots_resolve():
    stub = _stub(wrapped={"1field"})
    dex = ShieldSwap(stub)
    custom = LIT.replace("leaf_index: 1u32", "leaf_index: 7u32", 1)
    _mint(dex, wrapper_proofs=custom)
    _, args = stub.last_call
    assert args[2] == custom and args[9] == args[10] == args[11] == default_merkle_proofs()
    roots = [r[0] for r in stub.mapping_reads if r[1] == "freeze_list_root"]
    assert roots == [SHIELD_SWAP_FREEZELIST]                              # the wrapper list is never read


def test_populated_lists_yield_real_witnesses_per_subject_and_cache_them():
    stub = _stub(wrapped={"1field"})
    dex = ShieldSwap(stub)
    frozen = "aleo1s3ws5tra87fjycnjrwsjcrnw2qxr8jfqqdugnf0xzqqw29q9m5pqem2u4t"
    other = "aleo15mstsvdtzqf5nw8rfzx8mrllwxt907amfpt8nx3p8cskj4wd3uxq4uywn9"     # sorts between the signer and the recipient
    tree = fl.build_tree(fl.generate_leaves([frozen, other]), "testnet")
    stub.roots[SHIELD_SWAP_FREEZELIST] = tree[-1]
    stub.freeze_list_trees[SHIELD_SWAP_FREEZELIST] = tree
    recipient = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
    _mint(dex, recipient=recipient, withdrawal=recipient)
    _, args = stub.last_call
    signer_proof, recipient_proof = args[9], args[10]
    assert args[2] == default_merkle_proofs()                             # the USDCx list is still empty
    assert signer_proof == fl.generate_freezelist_proof(tree, SIGNER, tree[-1], "testnet")
    assert recipient_proof == args[11] == fl.generate_freezelist_proof(tree, recipient, tree[-1], "testnet")
    assert signer_proof != recipient_proof
    _mint(dex, recipient=recipient, withdrawal=recipient)
    assert stub.tree_reads == [SHIELD_SWAP_FREEZELIST]                     # cached across calls while the root holds
    with pytest.raises(FrozenAddressError):
        _mint(dex, recipient=frozen)
    # The list changes: a new root discards the cache and the (stale) published tree is rejected.
    bigger = fl.build_tree(fl.generate_leaves([frozen, other, recipient]), "testnet")
    stub.roots[SHIELD_SWAP_FREEZELIST] = bigger[-1]
    with pytest.raises(fl.FreezeListError, match="does not match the on-chain root"):
        _mint(dex)
    stub.freeze_list_trees[SHIELD_SWAP_FREEZELIST] = bigger
    with pytest.raises(FrozenAddressError):
        _mint(dex, recipient=recipient)
    assert stub.tree_reads == [SHIELD_SWAP_FREEZELIST] * 3


def test_unreadable_root_is_fatal_and_names_the_fix():
    stub = _stub()
    stub.roots[SHIELD_SWAP_FREEZELIST] = None
    with pytest.raises(fl.FreezeListError, match=r"freeze_list_root\[1u8\] is unreadable; pass compliance_proofs="):
        _mint(ShieldSwap(stub))


def test_credits_wrapper_slot_gets_the_placeholder_without_a_list_read():
    stub = _stub(wrapped={"2field"})
    dex = ShieldSwap(stub)
    dex.mint(pool_key="5field", tick_lower=-4080, tick_upper=4080, amount0_desired=10**9, amount1_desired=100,
             token0_program="tok0.aleo", token1_program="credits.aleo")
    _, args = stub.last_call
    assert args[3] == default_merkle_proofs()
    roots = [r[0] for r in stub.mapping_reads if r[1] == "freeze_list_root"]
    assert roots == [SHIELD_SWAP_FREEZELIST]


def test_collect_receiver_slots_screen_the_nft_withdrawal_address():
    stub = _stub(wrapped={"1field"})
    dex = ShieldSwap(stub)
    dex._token_program = lambda tid: "test_usdcx_stablecoin.aleo"
    seen = []
    original = dex.freezelist.resolve
    dex.freezelist.resolve = lambda slots: (seen.extend(slots), original(slots))[1]
    dex.collect(pool_key="5field", amount0_requested=7, amount1_requested=8)
    assert [(s.freezelist_program, s.address) for s in seen] == [
        (SHIELD_SWAP_FREEZELIST, SIGNER), (SHIELD_SWAP_FREEZELIST, "aleo1me"), ("test_usdcx_freezelist.aleo", "aleo1me")]
    assert stub.last_call[0] == "collect_to_wrapped_arc20" and len(stub.last_call[1]) == 8


def test_claim_router_slots_resolve_against_both_lists():
    from .test_claim import SWAP_OUTPUT_TEXT, _handle
    stub = StubAleo(mappings={"pools": {"5field": POOL_TEXT}, "slots": {"5field": SLOT_TEXT},
                              "swap_outputs": {"77field": SWAP_OUTPUT_TEXT},
                              "from_wrapper_token_id": {"1field": "9field", "2field": "9field"}})
    dex = ShieldSwap(stub)
    dex._token_program = lambda tid: {"1field": "test_usdcx_stablecoin.aleo", "2field": "test_usad_stablecoin.aleo"}[tid]
    dex.claim_swap_output(_handle()).transact()
    fn, args = stub.last_call
    assert (stub.last_program, fn) == (ROUTER_ID, "claim_to_wrapped_refund_wrapped")
    roots = [r[0] for r in stub.mapping_reads if r[1] == "freeze_list_root"]
    assert roots == [SHIELD_SWAP_FREEZELIST, "test_usad_freezelist.aleo", "test_usdcx_freezelist.aleo"]
    assert args[-3:] == [default_merkle_proofs()] * 3
    with pytest.raises(ValueError, match="claim_swap_output on this route takes 3"):
        dex.claim_swap_output(_handle(), compliance_proofs=(LIT,))


def test_swap_wrapped_input_resolves_the_sender_bundle_from_the_named_program():
    stub = _stub(wrapped={"1field"})
    dex = ShieldSwap(stub)
    dex.swap(pool_key="5field", token_in_id="1field", amount_in=1_000_000, expected_out=1_000,
             token_in_program="test_usdcx_stablecoin.aleo", identity=None, nonce=1, track=False)
    fn, args = stub.last_call
    assert (stub.last_program, fn) == (ROUTER_ID, "swap_from_wrapped")
    assert args[1] == default_merkle_proofs()
    roots = [r[0] for r in stub.mapping_reads if r[1] == "freeze_list_root"]
    assert roots == ["test_usdcx_freezelist.aleo"]
    with pytest.raises(ValueError, match="swap on this route takes 1"):
        dex.swap(pool_key="5field", token_in_id="1field", amount_in=1_000_000, expected_out=1_000,
                 token_in_program="test_usdcx_stablecoin.aleo", nonce=1, track=False, compliance_proofs=())

