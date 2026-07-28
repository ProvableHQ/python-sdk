"""Pure routing decisions for the shield_swap stack (migration guide §6).

Wrapped assets must enter and leave the AMM through the routers; plain
ARC-20s call the core directly.  A plain-input swap that OUTPUTS a wrapped
token starts on the core and claims through the router — the claim route
depends only on the finalized SwapOutput's token shapes.
"""
from __future__ import annotations

from typing import NamedTuple

from ._generated import PROGRAM_ID

ROUTER_ID = "shield_swap_router.aleo"
LP_ROUTER_ID = "shield_swap_lp_router.aleo"


class Route(NamedTuple):
    program: str
    function: str


def swap_route(input_wrapped: bool) -> Route:
    return (Route(ROUTER_ID, "swap_from_wrapped") if input_wrapped
            else Route(PROGRAM_ID, "swap"))


def claim_route(output_wrapped: bool, refund_wrapped: bool) -> Route:
    if output_wrapped and refund_wrapped:
        return Route(ROUTER_ID, "claim_to_wrapped_refund_wrapped")
    if output_wrapped:
        return Route(ROUTER_ID, "claim_to_wrapped_refund_arc20")
    if refund_wrapped:
        return Route(ROUTER_ID, "claim_to_arc20_refund_wrapped")
    return Route(PROGRAM_ID, "claim_swap_output")


def _lp_route(w0: bool, w1: bool, stem: str, core_fn: str) -> Route:
    if w0 and w1:
        return Route(LP_ROUTER_ID, f"{stem}_wrapped_wrapped")
    if w0:
        return Route(LP_ROUTER_ID, f"{stem}_wrapped_arc20")
    if w1:
        return Route(LP_ROUTER_ID, f"{stem}_arc20_wrapped")
    return Route(PROGRAM_ID, core_fn)


def mint_route(w0: bool, w1: bool) -> Route:
    return _lp_route(w0, w1, "mint_from", "mint")


def increase_route(w0: bool, w1: bool) -> Route:
    return _lp_route(w0, w1, "increase_from", "increase_liquidity")


def collect_route(w0: bool, w1: bool) -> Route:
    return _lp_route(w0, w1, "collect_to", "collect")
