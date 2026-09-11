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
# Port of the reference client in amm-v3 ts-tests/src/client/amm-client.ts
# (@ development) via the TS SDK's utils/blinding/identity.ts.  The domain
# separators are pinned from the reference client; CLAIM_OR_SWAP_DOMAIN must
# match the literal the program hashes in verify_blinded_address (confirmed
# unchanged in the deployed shield_swap.aleo bytecode).

BLINDING_FACTOR_DOMAIN = "42815354924796718559205719970686750292466968495484257field"
CLAIM_OR_SWAP_DOMAIN = "11835072102227764468342786961086432175093421716844963782363567713633field"

DEFAULT_PROGRAM = "shield_swap.aleo"


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
    gallop: bool = True,
) -> BlindedIdentity:
    """An unused single-use identity for *account*.

    Derives at ``start_counter, +1, …`` and probes the program's
    ``used_blinded_addresses`` mapping until one is free.  When the whole
    linear window is used — an account that has swapped more than *max_scan*
    times without a journal — *gallop* extends the search in O(log n) probes:
    double the stride past the window until a free counter appears, then
    bisect back to the lowest free one in that span.  Any free counter is a
    valid identity (a gap left by a failed swap is fine), so the search only
    needs SOME unused address, not the exact end of the used run.

    With ``gallop=False`` the linear window is the whole search and
    exhausting it raises — the fail-fast for a systematically wrong program.
    """
    from ._core import mapping_flag_set, read_mapping_value

    identities = BlindedIdentityCache(aleo.network_name, account, program)

    def is_used(counter: int) -> bool:
        # Straight to the node's mapping endpoint: no program download per
        # probe (this is the path every journal-less swap takes).
        return mapping_flag_set(read_mapping_value(
            aleo, program, "used_blinded_addresses", identities.at(counter).blinded_address))

    try:
        counter = find_unused_counter(is_used, start_counter=start_counter,
                                      max_scan=max_scan, gallop=gallop)
    except LookupError as exc:
        raise ValueError(f"{exc} for {program} — wrong program or scan range?") from None
    return identities.at(counter)


class BlindedIdentityCache:
    """Derive-once identities for one (account, program) during a search."""

    def __init__(self, network: str, account: Any, program: str) -> None:
        self._network = network
        self._scalar = str(account.view_key.to_scalar())
        self._signer = str(account.address)
        self._program = program
        self._cache: dict[int, BlindedIdentity] = {}

    def at(self, counter: int) -> BlindedIdentity:
        if counter not in self._cache:
            bf = derive_blinding_factor(self._scalar, counter, self._program,
                                        network=self._network)
            self._cache[counter] = BlindedIdentity(
                counter, bf,
                derive_blinded_address(bf, self._signer, self._program, network=self._network))
        return self._cache[counter]


def _unused_counter_search(start_counter: int, max_scan: int, gallop: bool):
    """The search as a generator: yields the counter to probe, receives
    whether it is used, and returns the chosen counter.  One algorithm drives
    both :func:`find_unused_counter` and :func:`find_unused_counter_async`.

    Linear over ``[start_counter, start_counter + max_scan)`` first; when that
    whole window is used and *gallop* is on, doubles the stride past it until
    a free counter appears, then bisects back to the lowest free counter in
    that span — O(log n) probes for an account with a long swap history.  Any
    free counter is acceptable (a gap left by a failed swap included), so the
    result is SOME unused counter, not necessarily the end of the used run.
    """
    for counter in range(start_counter, start_counter + max_scan):
        if not (yield counter):
            return counter
    if not gallop:
        raise LookupError(
            f"No unused blinded address in counters [{start_counter}, "
            f"{start_counter + max_scan})")
    lo = start_counter + max_scan - 1              # known used
    stride = max(max_scan, 1)
    while True:
        hi = lo + stride
        if hi - start_counter > 1 << 24:
            raise LookupError(
                f"No unused blinded address in counters [{start_counter}, {hi})")
        if not (yield hi):
            break
        lo, stride = hi, stride * 2
    # Bisect (lo used, hi free) down to the lowest free counter in the span.
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if (yield mid):
            lo = mid
        else:
            hi = mid
    return hi


def find_unused_counter(is_used: Any, *, start_counter: int = 0,
                        max_scan: int = 64, gallop: bool = True) -> int:
    """The lowest-effort unused counter given an ``is_used(counter)`` probe.

    See :func:`_unused_counter_search` for the strategy.

    Raises:
        LookupError: If the window is exhausted with *gallop* off, or the
            gallop exceeds 2^24 counters (a systematically wrong probe).
    """
    search = _unused_counter_search(start_counter, max_scan, gallop)
    try:
        counter = next(search)
        while True:
            counter = search.send(bool(is_used(counter)))
    except StopIteration as done:
        return done.value


async def find_unused_counter_async(is_used: Any, *, start_counter: int = 0,
                                    max_scan: int = 64, gallop: bool = True) -> int:
    """:func:`find_unused_counter` for an ``async def is_used(counter)`` probe —
    the same search, so the sync and async clients cannot drift apart."""
    search = _unused_counter_search(start_counter, max_scan, gallop)
    try:
        counter = next(search)
        while True:
            counter = search.send(bool(await is_used(counter)))
    except StopIteration as done:
        return done.value


def blinded_identity_at(
    aleo: Any,
    account: Any,
    program: str,
    counter: int,
) -> BlindedIdentity:
    """The identity at an exact *counter* — no on-chain probing.

    Use with journal-reserved counters for concurrent swaps;
    :func:`next_blinded_identity` (probe-based) remains the recovery path
    when no journal exists.
    """
    network = aleo.network_name
    scalar = str(account.view_key.to_scalar())
    signer = str(account.address)
    bf = derive_blinding_factor(scalar, counter, program, network=network)
    ba = derive_blinded_address(bf, signer, program, network=network)
    return BlindedIdentity(counter, bf, ba)
