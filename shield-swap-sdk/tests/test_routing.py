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
