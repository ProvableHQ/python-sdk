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
