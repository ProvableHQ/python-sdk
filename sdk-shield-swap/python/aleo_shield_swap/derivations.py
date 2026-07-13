"""Pure derivations: pool keys, tick keys, and the blinded identity.

Ports of the TS SDK's ``utils/keys.ts`` and ``utils/blinding/identity.ts``
(which themselves mirror the Provable reference client in ``amm-v3-tests``).
Everything here is pure and local — no network I/O; the view key never
leaves the process.  Every function is verified against vectors generated
from the TS implementation.
"""
from __future__ import annotations

from typing import Any


def _net(network: str) -> Any:
    if network == "testnet":
        from aleo import testnet as net
    else:
        from aleo import mainnet as net
    return net


def _strip_suffix(literal: str, suffix: str) -> str:
    trimmed = literal.strip()
    return trimmed[: -len(suffix)] if trimmed.endswith(suffix) else trimmed


def _hash_struct(struct: str, network: str) -> str:
    """``BHP256::hash_to_field(<struct>)`` — the contract's key hash."""
    net = _net(network)
    pt = net.Plaintext.from_string(struct)
    return str(net.BHP256().hash(pt.to_bits_le()))


def derive_pool_key(token0: str, token1: str, fee: int, *, network: str = "testnet") -> str:
    """Pool key for a token pair and fee tier, without the network.

    Matches the program byte-for-byte: the pair is sorted ascending (as the
    contract does) and hashed as ``PoolKey { token0, token1, fee }`` with the
    fee as a u16 in pips (3000 = 0.30%).  Token ids may carry or omit the
    ``field`` suffix.
    """
    if not 0 <= fee <= 0xFFFF:
        raise ValueError(f"fee must be a u16 (0–65535 pips), got {fee}")
    a = int(_strip_suffix(token0, "field"))
    b = int(_strip_suffix(token1, "field"))
    lo, hi = (a, b) if a <= b else (b, a)
    return _hash_struct(
        f"{{ token0: {lo}field, token1: {hi}field, fee: {fee}u16 }}", network
    )


def derive_tick_key(pool: str, tick: int, *, network: str = "testnet") -> str:
    """Key into the ``ticks`` mapping for one tick of a pool.

    Matches the program's ``get_tick_key``:
    ``BHP256::hash_to_field(TickKey { pool, tick })`` with the tick as i32.
    """
    if not -(2**31) <= tick < 2**31:
        raise ValueError(f"tick must be an i32, got {tick}")
    p = _strip_suffix(pool, "field")
    return _hash_struct(f"{{ pool: {p}field, tick: {tick}i32 }}", network)
