"""Pure derivations: pool keys, tick keys, and the blinded identity.

Ports of the TS SDK's ``utils/keys.ts`` and ``utils/blinding/identity.ts``
(which themselves mirror the Provable reference client in ``amm-v3-tests``).
Everything here is pure and local — no network I/O; the view key never
leaves the process.  Every function is verified against vectors generated
from the TS implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
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


# ── Blinded identity ─────────────────────────────────────────────────────────
#
# Port of amm-v3-tests src/client/amm-client.ts (feat/q128) via the TS SDK's
# utils/blinding/identity.ts.  The domain separators are pinned from the
# reference client; CLAIM_OR_SWAP_DOMAIN must match the literal the program
# hashes in verify_blinded_address.

BLINDING_FACTOR_DOMAIN = "42815354924796718559205719970686750292466968495484257field"
CLAIM_OR_SWAP_DOMAIN = "11835072102227764468342786961086432175093421716844963782363567713633field"

DEFAULT_PROGRAM = "shield_swap_v3.aleo"


@dataclass(frozen=True)
class BlindedIdentity:
    """A single-use blinded identity for one private swap or claim.

    ``blinding_factor`` is secret — whoever holds it can claim the swap's
    output; treat it like a key.  ``blinded_address`` is public.
    """

    counter: int
    blinding_factor: str
    blinded_address: str


def _program_address_field(net: Any, program: str) -> Any:
    """``self.address as field`` — x-coordinate of the program address."""
    return net.Address.from_program_id(program).to_group().to_x_coordinate()


def derive_blinding_factor(
    view_key_scalar: str,
    counter: int,
    program: str = DEFAULT_PROGRAM,
    *,
    network: str = "testnet",
) -> str:
    """Blinding factor for one swap/claim, from the view key and a counter.

    ``Poseidon8::hash([program_address, DOMAIN, view_key as field,
    counter as field])`` — deterministic, so identities are re-derivable
    without storing them.  Pure and local; the view key never leaves the
    process.
    """
    net = _net(network)
    addr_field = _program_address_field(net, program)
    vk_field = net.Scalar.from_string(view_key_scalar).to_field()
    counter_field = net.U32.from_string(f"{counter}u32").to_scalar().to_field()
    preimage = [addr_field, net.Field.from_string(BLINDING_FACTOR_DOMAIN),
                vk_field, counter_field]
    return str(net.Poseidon8().hash(preimage))


def derive_blinded_address(
    blinding_factor: str,
    signer_address: str,
    program: str = DEFAULT_PROGRAM,
    *,
    network: str = "testnet",
) -> str:
    """Public blinded address for a blinding factor and signer.

    ``Poseidon8::hash_to_address_raw([program_address, CLAIM_OR_SWAP_DOMAIN,
    signer, blinding_factor])``.  The 252-bit little-endian repacking below
    emulates snarkVM's ``Plaintext::Array::to_fields_raw`` and is
    load-bearing — a one-bit deviation yields an address the program rejects.
    """
    net = _net(network)
    size_in_data_bits = 252

    contract_field = _program_address_field(net, program)
    signer_field = net.Address.from_string(signer_address).to_group().to_x_coordinate()

    input_bits: list[bool] = []
    for f in (contract_field, net.Field.from_string(CLAIM_OR_SWAP_DOMAIN),
              signer_field, net.Field.from_string(blinding_factor)):
        input_bits.extend(f.to_bits_le())

    preimage = [
        net.Field.from_bits_le(input_bits[i:i + size_in_data_bits])
        for i in range(0, len(input_bits), size_in_data_bits)
    ]
    blinded_group = net.Poseidon8().hash_to_group(preimage)
    return str(net.Address.from_group(blinded_group))


def next_blinded_identity(
    aleo: Any,
    account: Any,
    program: str = DEFAULT_PROGRAM,
    *,
    start_counter: int = 0,
    max_scan: int = 64,
) -> BlindedIdentity:
    """First unused single-use identity for *account*.

    Derives at ``start_counter, +1, …`` and probes the program's
    ``used_blinded_addresses`` mapping until one is free.  ``max_scan`` fails
    fast when something is systematically wrong (e.g. wrong program).
    """
    network = aleo.network_name
    scalar = str(account.view_key.to_scalar())
    signer = str(account.address)
    mapping = aleo.programs.get(program).mapping("used_blinded_addresses")
    for counter in range(start_counter, start_counter + max_scan):
        bf = derive_blinding_factor(scalar, counter, program, network=network)
        ba = derive_blinded_address(bf, signer, program, network=network)
        used = mapping.get(ba)
        if used in (None, "", "null", "false"):
            return BlindedIdentity(counter, bf, ba)
    raise ValueError(
        f"No unused blinded address in counters [{start_counter}, "
        f"{start_counter + max_scan}) for {program} — wrong program or scan range?"
    )
