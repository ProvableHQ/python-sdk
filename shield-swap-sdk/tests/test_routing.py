"""Pure route table — program/function per token shape (guide §6)."""
from aleo_shield_swap._routing import (
    LP_ROUTER_ID,
    ROUTER_ID,
    claim_route,
    collect_route,
    increase_route,
    mint_route,
    swap_route,
)

CORE = "shield_swap.aleo"


def test_swap_route():
    assert swap_route(False) == (CORE, "swap")
    assert swap_route(True) == (ROUTER_ID, "swap_from_wrapped")


def test_claim_route_covers_all_shapes():
    assert claim_route(False, False) == (CORE, "claim_swap_output")
    assert claim_route(True, False) == (ROUTER_ID, "claim_to_wrapped_refund_arc20")
    assert claim_route(False, True) == (ROUTER_ID, "claim_to_arc20_refund_wrapped")
    assert claim_route(True, True) == (ROUTER_ID, "claim_to_wrapped_refund_wrapped")


def test_claim_route_no_refund_variants():
    """A fully consumed swap (amount_remaining == 0) takes the no-refund
    entrypoints, which never mint a zero-value refund record.  The refund
    leg's shape is irrelevant once there is nothing to refund, except that a
    wrapped refund token still forces the router (the core would try to pay
    a wrapper record it cannot mint)."""
    assert claim_route(False, False, no_refund=True) == (CORE, "claim_swap_output_no_refund")
    assert claim_route(True, False, no_refund=True) == (ROUTER_ID, "claim_to_wrapped_no_refund")
    assert claim_route(True, True, no_refund=True) == (ROUTER_ID, "claim_to_wrapped_no_refund")
    assert claim_route(False, True, no_refund=True) == (ROUTER_ID, "claim_to_arc20_no_refund")


def test_lp_routes():
    assert mint_route(False, False) == (CORE, "mint")
    assert mint_route(True, False) == (LP_ROUTER_ID, "mint_from_wrapped_arc20")
    assert mint_route(False, True) == (LP_ROUTER_ID, "mint_from_arc20_wrapped")
    assert mint_route(True, True) == (LP_ROUTER_ID, "mint_from_wrapped_wrapped")
    assert increase_route(True, False) == (LP_ROUTER_ID, "increase_from_wrapped_arc20")
    assert increase_route(False, True) == (LP_ROUTER_ID, "increase_from_arc20_wrapped")
    assert increase_route(False, False) == (CORE, "increase_liquidity")
    assert collect_route(False, True) == (LP_ROUTER_ID, "collect_to_arc20_wrapped")
    assert collect_route(True, True) == (LP_ROUTER_ID, "collect_to_wrapped_wrapped")
    assert collect_route(False, False) == (CORE, "collect")


def test_rebalance_route_covers_shape_times_funding():
    """14 entrypoints: shape (plain/wrapped per side) x funding mode.  Symmetric
    shapes collapse the one-sided modes into ``one``; mixed shapes keep
    ``fund0``/``fund1`` because the wrapped side also carries a sender proof."""
    from aleo_shield_swap._routing import REBALANCE_ROUTER_ID, rebalance_route
    R = REBALANCE_ROUTER_ID
    assert rebalance_route(False, False, False, False) == (R, "rebalance_plain_plain_none")
    assert rebalance_route(False, False, True, False) == (R, "rebalance_plain_plain_one")
    assert rebalance_route(False, False, False, True) == (R, "rebalance_plain_plain_one")
    assert rebalance_route(False, False, True, True) == (R, "rebalance_plain_plain_both")
    assert rebalance_route(True, False, False, False) == (R, "rebalance_wrapped_plain_none")
    assert rebalance_route(True, False, True, False) == (R, "rebalance_wrapped_plain_fund0")
    assert rebalance_route(True, False, False, True) == (R, "rebalance_wrapped_plain_fund1")
    assert rebalance_route(True, False, True, True) == (R, "rebalance_wrapped_plain_both")
    assert rebalance_route(False, True, False, False) == (R, "rebalance_plain_wrapped_none")
    assert rebalance_route(False, True, True, False) == (R, "rebalance_plain_wrapped_fund0")
    assert rebalance_route(False, True, False, True) == (R, "rebalance_plain_wrapped_fund1")
    assert rebalance_route(False, True, True, True) == (R, "rebalance_plain_wrapped_both")
    assert rebalance_route(True, True, False, False) == (R, "rebalance_wrapped_wrapped_none")
    assert rebalance_route(True, True, True, False) == (R, "rebalance_wrapped_wrapped_one")
    assert rebalance_route(True, True, False, True) == (R, "rebalance_wrapped_wrapped_one")
    assert rebalance_route(True, True, True, True) == (R, "rebalance_wrapped_wrapped_both")


def test_rebalance_input_counts_match_pinned_router_abi():
    """Slot rule shared by all 14 entries: [nft, nonce, each funded side's
    record (+ sender proof when wrapped), every wrapped side's receiver proof,
    request, assets, owner proofs, withdrawal proofs].  Derive the count from
    that rule for every (shape, funding) case and check it against the pinned
    ABI, so a router redeploy that reshapes a slot fails hermetically."""
    import json
    from pathlib import Path
    from itertools import product
    from aleo_shield_swap._routing import rebalance_route

    abi = json.loads((Path(__file__).parents[1] / "codegen"
                      / "shield_swap_rebalance_router.abi.json").read_text())
    pinned = {fn["name"]: len(fn["inputs"]) for fn in abi["functions"]}
    assert len(pinned) == 14
    for w0, w1, f0, f1 in product((False, True), repeat=4):
        _, fn = rebalance_route(w0, w1, f0, f1)
        funding = (1 + w0) * f0 + (1 + w1) * f1
        receivers = int(w0) + int(w1)
        assembled = 2 + funding + receivers + 4
        assert pinned[fn] == assembled, (fn, pinned[fn], assembled)


def test_client_input_counts_match_pinned_router_abis():
    """ABI ↔ client parity: the input counts the dispatch tests pin on the
    client side must equal the pinned router signatures.  A router redeploy
    that changes a signature fails here hermetically (and in the live drift
    test), instead of as an authorization error on a wrapped flow."""
    import json
    from pathlib import Path

    pinned: dict[str, int] = {}
    codegen = Path(__file__).parents[1] / "codegen"
    for name in ("shield_swap_router", "shield_swap_lp_router"):
        abi = json.loads((codegen / f"{name}.abi.json").read_text())
        pinned.update({fn["name"]: len(fn["inputs"]) for fn in abi["functions"]})

    # The exact input lists the client assembles (asserted in
    # test_swap/test_claim/test_liquidity against the stub recorder).
    assembled = {
        "swap_from_wrapped": 13,
        "claim_to_wrapped_refund_arc20": 9,
        "claim_to_arc20_refund_wrapped": 9,
        "claim_to_wrapped_refund_wrapped": 10,
        "claim_to_arc20_no_refund": 7,
        "claim_to_wrapped_no_refund": 8,
        "mint_from_wrapped_arc20": 12,
        "mint_from_arc20_wrapped": 12,
        "mint_from_wrapped_wrapped": 13,
        "increase_from_wrapped_arc20": 12,
        "increase_from_arc20_wrapped": 12,
        "increase_from_wrapped_wrapped": 13,
        "collect_to_wrapped_arc20": 8,
        "collect_to_arc20_wrapped": 8,
        "collect_to_wrapped_wrapped": 9,
    }
    for fn, count in assembled.items():
        assert pinned[fn] == count, (
            f"{fn}: pinned ABI takes {pinned[fn]} inputs, client assembles {count}")
