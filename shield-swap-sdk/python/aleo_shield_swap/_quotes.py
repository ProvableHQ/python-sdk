"""Shared quote validation and multi-hop wire encoding."""
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

from ._core import _amount_to_base_units, resolve_swap_params
from .types import SwapQuote, SwapQuoteHop
from .tick_math import int_to_u256_plaintext, u256_to_int


def _units(amount: int, decimals: int) -> str:
    digits = str(amount).zfill(decimals + 1)
    return (digits[:-decimals] + "." + digits[-decimals:]).rstrip("0").rstrip(".") if decimals else digits


def _slippage(value):
    if type(value) is not int or not 0 <= value < 10000:
        raise ValueError("slippage_bps must be an integer within [0, 10000)")


def quote_request(tokens, token_in, token_out, amount, slippage_bps):
    from .api import _token_by_symbol
    _slippage(slippage_bps)
    source, target = _token_by_symbol(tokens, token_in), _token_by_symbol(tokens, token_out)
    if source.address == target.address:
        raise ValueError("Input and output tokens must differ")
    if not isinstance(amount, (str, Decimal)):
        raise TypeError("quote amount_in uses token units: pass a string or Decimal")
    atomic = _amount_to_base_units(amount, source.address, tokens)
    if atomic <= 0:
        raise ValueError("amount_in must be positive")
    return source, target, _units(atomic, source.decimals)


def build_quote(source, target, amount, slippage_bps, route, network, program):
    if (route.token_in, route.token_out) != (source.address, target.address):
        raise ValueError("Quoted route does not match requested tokens")
    if not route.estimated_amount_out:
        raise ValueError("No executable output quote for this pair")
    expected = _amount_to_base_units(route.estimated_amount_out, target.address, [target])
    minimum = expected * (10000 - slippage_bps) // 10000
    q = SwapQuote(amount, _units(expected, target.decimals), _units(minimum, target.decimals),
                  tuple(SwapQuoteHop(h.pool_key, h.token_in, h.token_out, h.zero_for_one) for h in route.hops),
                  source.address, target.address, network, program, slippage_bps,
                  source.decimals, target.decimals, route.protocol_revision)
    quote_amounts(q, network, program)
    return q


def quote_amounts(q, network, program):
    if not isinstance(q, SwapQuote):
        raise TypeError("Pass a SwapQuote returned by quote()")
    if (q.network, q.program) != (network, program):
        raise ValueError("Quote network/program does not match the client")
    _slippage(q.slippage_bps)
    if not 1 <= len(q.hops) <= 3:
        raise ValueError("An executable route requires one to three hops")
    current = q.token_in_id
    pools, tokens = set(), {current}
    for hop in q.hops:
        if (hop.token_in != current or hop.token_out in tokens or hop.pool_key in pools
                or type(hop.zero_for_one) is not bool):
            raise ValueError("Route must connect input to output without repeated pools or tokens")
        pools.add(hop.pool_key)
        tokens.add(hop.token_out)
        current = hop.token_out
    if current != q.token_out_id:
        raise ValueError("Route does not end at the quoted output token")
    metadata = [SimpleNamespace(id=q.token_in_id, address=q.token_in_id, decimals=q.token_in_decimals),
                SimpleNamespace(id=q.token_out_id, address=q.token_out_id, decimals=q.token_out_decimals)]
    amount = _amount_to_base_units(q.amount_in, q.token_in_id, metadata)
    expected = _amount_to_base_units(q.estimated_amount_out, q.token_out_id, metadata)
    minimum = _amount_to_base_units(q.minimum_amount_out, q.token_out_id, metadata)
    if amount <= 0 or expected <= 0 or minimum <= 0 or minimum != expected * (10000-q.slippage_bps)//10000:
        raise ValueError("Quote amounts or slippage floor are invalid")
    return amount, expected


def resolve_quote(q, pools, slots, amount, expected):
    literals = []
    for hop, pool, slot in zip(q.hops, pools, slots):
        if not pool.enabled:
            raise ValueError("Quoted pool is disabled")
        resolved = resolve_swap_params(pool=pool, slot=slot, token_in_id=hop.token_in,
                                       amount_in=amount, expected_out=expected,
                                       slippage_bps=q.slippage_bps)
        if resolved.token_out_id != hop.token_out or resolved.zero_for_one != hop.zero_for_one:
            raise ValueError("Quoted hop disagrees with on-chain pool tokens/direction")
        price = u256_to_int(slot.sqrt_price)
        if not (resolved.sqrt_price_limit < price if hop.zero_for_one else resolved.sqrt_price_limit > price):
            raise ValueError("Pool price leaves no executable range for the quoted hop")
        literals.append(f"{{ pool: {hop.pool_key}, zero_for_one: {str(hop.zero_for_one).lower()}, "
                        f"sqrt_price_limit: {int_to_u256_plaintext(resolved.sqrt_price_limit)} }}")
    return replace(resolved, token_out_id=q.token_out_id), literals


def multi_hop_inputs(q, literals, record, identity, amount, minimum, nonce, deadline, proofs, wrapped):
    padding = "{ pool: 0field, zero_for_one: false, sqrt_price_limit: { hi: 0u128, lo: 0u128 } }"
    tail = [q.token_in_id, q.token_out_id, f"{amount}u128", f"{minimum}u128",
            *literals, *([padding] * (3-len(literals))), f"{len(literals)}u8", f"{nonce}u64", f"{deadline}u32"]
    return ([record, proofs, identity.blinding_factor, identity.blinded_address, *tail] if wrapped else
            [identity.blinding_factor, identity.blinded_address, record, *tail])
