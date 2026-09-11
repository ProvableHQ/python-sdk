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
#: Close-and-remint in one transaction.  Deployed on testnet; not on mainnet
#: as of 2026-09 — a mainnet call fails at program registration.
REBALANCE_ROUTER_ID = "shield_swap_rebalance_router.aleo"


class Route(NamedTuple):
    """Which program and function a call must go through.

    Dispatch target only — it says nothing about the arguments, which differ
    between the core and router entrypoints.
    """

    program: str
    function: str


def swap_route(input_wrapped: bool) -> Route:
    """Route a swap by whether the INPUT token is wrapped.

    A wrapped input must enter through the router, which unwraps it; a plain
    ARC-20 calls the core directly. The output shape does not matter here — it is
    settled at claim time by :func:`claim_route`.
    """
    return (Route(ROUTER_ID, "swap_from_wrapped") if input_wrapped
            else Route(PROGRAM_ID, "swap"))


def claim_route(output_wrapped: bool, refund_wrapped: bool,
                no_refund: bool = False) -> Route:
    """Route a claim by the wrapped-ness of its output and refund legs.

    Either leg being wrapped forces the claim through the router, since only it
    can wrap on the way out; a claim with both legs plain goes to the core. The
    two flags come from the finalized ``SwapOutput``, not from the original swap.

    *no_refund* (``amount_remaining == 0``) selects the no-refund entrypoints,
    which take no ``amount_remaining`` input and mint no zero-value refund
    record.  With nothing to refund only the output leg's shape decides the
    proofs, but a wrapped refund token still keeps the claim on the router.
    """
    if no_refund:
        if output_wrapped:
            return Route(ROUTER_ID, "claim_to_wrapped_no_refund")
        if refund_wrapped:
            return Route(ROUTER_ID, "claim_to_arc20_no_refund")
        return Route(PROGRAM_ID, "claim_swap_output_no_refund")
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
    """Route a mint by which of the pool's two tokens are wrapped.

    Any wrapped side sends the call through the LP router; both plain goes to the
    core. *w0* and *w1* follow the pool's token order, so passing them reversed
    silently picks the wrong entrypoint.
    """
    return _lp_route(w0, w1, "mint_from", "mint")


def increase_route(w0: bool, w1: bool) -> Route:
    """Route a liquidity increase — same wrapped-ness rule as :func:`mint_route`."""
    return _lp_route(w0, w1, "increase_from", "increase_liquidity")


def collect_route(w0: bool, w1: bool) -> Route:
    """Route a fee collection by which pool tokens are wrapped.

    Mirrors :func:`mint_route`, but the router wraps on the way OUT rather than
    unwrapping on the way in.
    """
    return _lp_route(w0, w1, "collect_to", "collect")


def rebalance_route(w0: bool, w1: bool, funds0: bool, funds1: bool) -> Route:
    """Route a rebalance by each side's shape and whether it takes funding.

    The router deploys one entrypoint per (shape, funding) combination —
    ``rebalance_<side0>_<side1>_<mode>`` with sides ``plain``/``wrapped`` and
    mode ``none``/``both``, or for a single funded side ``one`` when both
    sides share a shape (the slots are symmetric) and ``fund0``/``fund1`` when
    they differ (a wrapped funded side also carries a sender proof).  Every
    rebalance goes through the router; the core's ``rebalance_position``
    accepts only the router as caller.
    """
    shape = f"{'wrapped' if w0 else 'plain'}_{'wrapped' if w1 else 'plain'}"
    if funds0 and funds1:
        mode = "both"
    elif funds0 or funds1:
        mode = "one" if w0 == w1 else ("fund0" if funds0 else "fund1")
    else:
        mode = "none"
    return Route(REBALANCE_ROUTER_ID, f"rebalance_{shape}_{mode}")
