"""Pure Hyperlane sealevel (Solana) layouts for the SOL warp route.

Nothing here imports solders or solana-py: inputs and outputs are ``bytes``,
``int`` and base58 ``str`` so every layout is testable on an Aleo-only
install. Sources: veil ``src/solana/SEALEVEL_NOTES.md`` (primary-source
derivations against hyperlane-monorepo 45c0988), ``src/solana/transferRemote.ts``,
``src/solana/igp.ts``, ``src/protocols/hyperlane/solanaMetadata.ts``, and the
recorded mainnet deposit in ``tests/fixtures/sealevel-transfer-remote.json``.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from ._base58 import b58decode, b58encode
from .errors import BridgeError, ConfigurationError, InvalidAmountError, InvalidRecipientError, RouteUnavailableError
from .registry import Route

# SEALEVEL_NOTES §1: every Sealevel Hyperlane instruction is prefixed with this
# fixed 8-byte discriminator; TransferRemote is Borsh enum variant 1.
PROGRAM_INSTRUCTION_DISCRIMINATOR = bytes([1] * 8)
TRANSFER_REMOTE_VARIANT_TAG = 1
INSTRUCTION_DATA_BYTES = 77          # 8 + 1 + 4 + 32 + 32
U256_BYTES = 32
ALEO_MAINNET_HYPERLANE_DOMAIN = 1634493807


def build_transfer_remote_instruction_data(destination_domain: int, recipient32: bytes, amount: int) -> bytes:
    """``[8B 0x01×8][1B 0x01][u32 LE domain][32B recipient][u256 LE amount]`` — 77 bytes.

    ``recipient32`` is the raw bech32m payload from ``encoding.aleo_address_to_bytes32``
    (no byte reversal); ``amount`` is lamports.
    """
    if not 0 <= destination_domain <= 0xFFFF_FFFF:
        raise ConfigurationError(f"destination domain {destination_domain} does not fit in a u32")
    if len(recipient32) != 32:
        raise InvalidRecipientError(f"recipient must be exactly 32 bytes, got {len(recipient32)}")
    if not 0 <= amount < (1 << (U256_BYTES * 8)):
        raise InvalidAmountError("amount does not fit in a 32-byte unsigned integer")
    data = (
        PROGRAM_INSTRUCTION_DISCRIMINATOR
        + bytes([TRANSFER_REMOTE_VARIANT_TAG])
        + destination_domain.to_bytes(4, "little")
        + bytes(recipient32)
        + amount.to_bytes(U256_BYTES, "little")
    )
    assert len(data) == INSTRUCTION_DATA_BYTES
    return data


# SEALEVEL_NOTES §4: AccountData<DiscriminatorPrefixed<Igp>> layout and compute_gas_fee constants.
IGP_DISCRIMINATOR = b"IGP_____"
TOKEN_EXCHANGE_RATE_SCALE = 10 ** 19     # exchange rate 1.0 is stored as 10^19
SOL_DECIMALS = 9
GAS_ORACLE_ENTRY_BYTES = 38              # [4B domain][1B tag][16B exchange rate][16B gas price][1B decimals]
REMOTE_GAS_DATA_TAG = 0                  # the only GasOracle variant defined today


@dataclass(frozen=True)
class GasOracle:
    token_exchange_rate: int
    gas_price: int
    token_decimals: int


@dataclass(frozen=True)
class IgpAccount:
    bump: int
    salt: bytes
    owner: str | None
    beneficiary: str
    gas_oracles: dict[int, GasOracle]
    unsupported_oracles: dict[int, int]   # domain -> variant tag, for entries that are not RemoteGasData


class _Cursor:
    """Little-endian Borsh reader over immutable bytes."""

    def __init__(self, data: bytes) -> None:
        self._data = bytes(data)
        self._offset = 0

    def take(self, size: int) -> bytes:
        end = self._offset + size
        if end > len(self._data):
            raise ConfigurationError("malformed Sealevel IGP account data: declared layout exceeds the supplied bytes")
        chunk = self._data[self._offset:end]
        self._offset = end
        return chunk

    def u8(self) -> int:
        return self.take(1)[0]

    def u32(self) -> int:
        return int.from_bytes(self.take(4), "little")

    def u128(self) -> int:
        return int.from_bytes(self.take(16), "little")

    def pubkey(self) -> str:
        return b58encode(self.take(32))


def decode_igp_account(data: bytes) -> IgpAccount:
    """Decode the terminal ``Igp`` account (the ``inner`` of an OverheadIgp), SEALEVEL_NOTES §4."""
    cursor = _Cursor(data)
    if cursor.u8() != 1:
        raise ConfigurationError("Sealevel IGP account is not initialized")
    discriminator = cursor.take(8)
    if discriminator != IGP_DISCRIMINATOR:
        raise ConfigurationError(
            f"Sealevel IGP account has an unexpected discriminator {discriminator!r}; expected {IGP_DISCRIMINATOR!r}"
        )
    bump = cursor.u8()
    salt = cursor.take(32)
    owner_tag = cursor.u8()
    if owner_tag not in (0, 1):
        raise ConfigurationError(f"malformed Sealevel IGP account data: unsupported owner option tag {owner_tag}")
    owner = cursor.pubkey() if owner_tag == 1 else None
    beneficiary = cursor.pubkey()
    count = cursor.u32()
    oracles: dict[int, GasOracle] = {}
    unsupported: dict[int, int] = {}
    for _ in range(count):
        domain = cursor.u32()
        tag = cursor.u8()
        exchange_rate = cursor.u128()
        gas_price = cursor.u128()
        decimals = cursor.u8()
        if tag == REMOTE_GAS_DATA_TAG:
            oracles[domain] = GasOracle(exchange_rate, gas_price, decimals)
        else:
            unsupported[domain] = tag
    return IgpAccount(bump, salt, owner, beneficiary, oracles, unsupported)


def igp_lamports(oracle: GasOracle, gas_amount: int) -> int:
    """``compute_gas_fee`` + ``convert_decimals`` (SEALEVEL_NOTES §4), exact integer arithmetic."""
    destination_cost = gas_amount * oracle.gas_price
    origin_cost = destination_cost * oracle.token_exchange_rate // TOKEN_EXCHANGE_RATE_SCALE
    if oracle.token_decimals <= SOL_DECIMALS:
        return origin_cost * 10 ** (SOL_DECIMALS - oracle.token_decimals)
    return origin_cost // 10 ** (oracle.token_decimals - SOL_DECIMALS)


def quote_igp_lamports(igp_account_data: bytes, destination_domain: int, gas_amount: int) -> int:
    """Lamports the IGP charges to deliver ``gas_amount`` destination gas to ``destination_domain``.

    ``gas_amount`` is the warp token's ``destination_gas`` for the domain (route metadata
    ``destinationGasAmount``, 464000 for Aleo), not derived from the message.
    """
    account = decode_igp_account(igp_account_data)
    if destination_domain in account.unsupported_oracles:
        tag = account.unsupported_oracles[destination_domain]
        raise ConfigurationError(
            f"Sealevel IGP account has an unexpected GasOracle variant tag {tag} for domain {destination_domain}; "
            "only variant 0 (RemoteGasData) is decoded"
        )
    oracle = account.gas_oracles.get(destination_domain)
    if oracle is None:
        raise ConfigurationError(f"Sealevel IGP account has no gas-oracle entry for destination domain {destination_domain}")
    return igp_lamports(oracle, gas_amount)


# --- Route metadata ----------------------------------------------------------------------------

SOLANA_ROUTE_ID = "hyperlane:solana/sol->aleo/sol"
SYSTEM_PROGRAM_ADDRESS = "11111111111111111111111111111111"
SOLANA_PUBKEY_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

# SEALEVEL_NOTES §3: seeds are separate byte strings (separators are their own seed).
DISPATCHED_MESSAGE_SEED_PREFIX = (b"hyperlane", b"-", b"dispatched_message", b"-")
GAS_PAYMENT_SEED_PREFIX = (b"hyperlane_igp", b"-", b"gas_payment", b"-")
PDA_MARKER = b"ProgramDerivedAddress"
MAX_SEEDS = 16
MAX_SEED_LENGTH = 32

# veil protocols/hyperlane/solana.ts: rent for the two accounts a transfer creates (gas-payment PDA,
# dispatched-message PDA) plus the sender's own rent floor; compute-unit limit set on every transfer.
GAS_PAYMENT_ACCOUNT_DATA_LENGTH = 141
DISPATCHED_MESSAGE_ACCOUNT_DATA_LENGTH = 194
COMPUTE_UNIT_LIMIT = 400_000


@dataclass(frozen=True)
class SolanaRouteMetadata:
    warp_program_address: str
    token_pda: str
    native_collateral_pda: str
    dispatch_authority_pda: str
    mailbox_program_address: str
    mailbox_outbox_pda: str
    igp_program_address: str
    igp_program_data_pda: str
    igp_account: str
    igp_overhead_account: str | None
    spl_noop_program_address: str
    destination_domain: int
    destination_gas_amount: int
    registry_commit: str
    solana_reviewed_at: str
    solana_config_source: str


_PUBKEY_FIELDS = (
    ("warpProgramAddress", "warp_program_address"),
    ("tokenPda", "token_pda"),
    ("nativeCollateralPda", "native_collateral_pda"),
    ("dispatchAuthorityPda", "dispatch_authority_pda"),
    ("mailboxProgramAddress", "mailbox_program_address"),
    ("mailboxOutboxPda", "mailbox_outbox_pda"),
    ("igpProgramAddress", "igp_program_address"),
    ("igpProgramDataPda", "igp_program_data_pda"),
    ("igpAccount", "igp_account"),
    ("splNoopProgramAddress", "spl_noop_program_address"),
)


def solana_route_metadata(route: Route) -> SolanaRouteMetadata:
    """Validate and return the reviewed Solana deployment metadata (veil ``solanaMetadata.ts``).

    Every address participates in instruction account ordering, so the whole route is
    refused when one field is missing or malformed rather than letting a bad key through.
    """
    if route.protocol != "hyperlane":
        raise RouteUnavailableError(f"Solana Hyperlane actions require a Hyperlane route, got {route.protocol}: {route.id}")
    if route.availability != "active":
        raise RouteUnavailableError(f"Hyperlane route is not executable: {route.id} ({route.availability})")
    metadata: Mapping[str, object] = route.metadata or {}

    def pubkey(key: str) -> str:
        value = metadata.get(key)
        if not isinstance(value, str) or not SOLANA_PUBKEY_RE.match(value):
            raise RouteUnavailableError(f"Solana Hyperlane route has an invalid {key}: {route.id}")
        return value

    fields = {attr: pubkey(key) for key, attr in _PUBKEY_FIELDS}
    overhead = metadata.get("igpOverheadAccount")
    fields["igp_overhead_account"] = None if overhead is None else pubkey("igpOverheadAccount")

    domain = metadata.get("destinationDomain")
    if isinstance(domain, bool) or not isinstance(domain, int) or not 0 <= domain <= 0xFFFF_FFFF:
        raise RouteUnavailableError(f"Solana Hyperlane route has an invalid destinationDomain: {route.id}")
    gas = metadata.get("destinationGasAmount")
    if isinstance(gas, bool) or not (isinstance(gas, int) or (isinstance(gas, str) and gas.isdigit())):
        raise RouteUnavailableError(f"Solana Hyperlane route has an invalid destinationGasAmount: {route.id}")
    commit = metadata.get("registryCommit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise RouteUnavailableError(f"Solana Hyperlane route has an invalid registryCommit: {route.id}")
    reviewed = metadata.get("solanaReviewedAt")
    if not isinstance(reviewed, str) or not re.match(r"^\d{4}-\d{2}-\d{2}", reviewed):
        raise RouteUnavailableError(f"Solana Hyperlane route has an invalid solanaReviewedAt: {route.id}")
    source = metadata.get("solanaConfigSource")
    if not isinstance(source, str) or not source:
        raise RouteUnavailableError(f"Solana Hyperlane route has an invalid solanaConfigSource: {route.id}")
    return SolanaRouteMetadata(
        destination_domain=domain,
        destination_gas_amount=int(gas),
        registry_commit=commit,
        solana_reviewed_at=reviewed,
        solana_config_source=source,
        **fields,
    )


# --- PDAs -------------------------------------------------------------------------------------

_ED25519_P = 2 ** 255 - 19
_ED25519_D = (-121665 * pow(121666, -1, _ED25519_P)) % _ED25519_P


def _is_on_curve(point: bytes) -> bool:
    """Whether a compressed Edwards Y coordinate decompresses (curve25519-dalek ``decompress``):
    x² = (y² − 1) / (d·y² + 1) must have a square root (or be zero)."""
    y = (int.from_bytes(point, "little") & ((1 << 255) - 1)) % _ED25519_P
    y2 = y * y % _ED25519_P
    u = (y2 - 1) % _ED25519_P
    v = (_ED25519_D * y2 + 1) % _ED25519_P
    if v == 0:
        return u == 0
    x2 = u * pow(v, -1, _ED25519_P) % _ED25519_P
    return x2 == 0 or pow(x2, (_ED25519_P - 1) // 2, _ED25519_P) == 1


def create_program_address(seeds: Sequence[bytes], program_id: str) -> str | None:
    """``Pubkey::create_program_address``: sha256(seeds ‖ program_id ‖ marker); None when on-curve."""
    if len(seeds) > MAX_SEEDS:
        raise BridgeError(f"program address derivation accepts at most {MAX_SEEDS} seeds")
    digest = hashlib.sha256()
    for seed in seeds:
        if len(seed) > MAX_SEED_LENGTH:
            raise BridgeError(f"each program address seed must be at most {MAX_SEED_LENGTH} bytes")
        digest.update(bytes(seed))
    digest.update(b58decode(program_id))
    digest.update(PDA_MARKER)
    candidate = digest.digest()
    return None if _is_on_curve(candidate) else b58encode(candidate)


def find_program_address(seeds: Sequence[bytes], program_id: str) -> tuple[str, int]:
    """``Pubkey::find_program_address``: try bump seeds 255 → 0, return the first off-curve address."""
    for bump in range(255, -1, -1):
        address = create_program_address([*seeds, bytes([bump])], program_id)
        if address is not None:
            return address, bump
    raise BridgeError("unable to find a viable program address bump seed")


def derive_dispatched_message_pda(mailbox_program_address: str, unique_message_address: str) -> str:
    """Mailbox ``["hyperlane","-","dispatched_message","-", unique_message_pubkey]`` (SEALEVEL_NOTES §3)."""
    return find_program_address([*DISPATCHED_MESSAGE_SEED_PREFIX, b58decode(unique_message_address)], mailbox_program_address)[0]


def derive_gas_payment_pda(igp_program_address: str, unique_message_address: str) -> str:
    """IGP ``["hyperlane_igp","-","gas_payment","-", unique_message_pubkey]`` — same unique key as the message PDA."""
    return find_program_address([*GAS_PAYMENT_SEED_PREFIX, b58decode(unique_message_address)], igp_program_address)[0]


# --- Account table ----------------------------------------------------------------------------

@dataclass(frozen=True)
class SolanaAccountMeta:
    address: str
    signer: bool
    writable: bool

    def to_dict(self) -> dict[str, str | bool]:
        return {"address": self.address, "signer": self.signer, "writable": self.writable}


def account_metas(metadata: SolanaRouteMetadata, sender: str, unique_message: str) -> list[SolanaAccountMeta]:
    """The native-collateral ``TransferRemote`` account list, SEALEVEL_NOTES §2 rows 0–15.

    Row 12 (``igpOverheadAccount``) is present only when the route wraps its IGP in an
    OverheadIgp; the list then has 16 entries, otherwise 15. The sender compiles writable
    (the native-collateral ``transfer_in`` CPI debits it) and the unique-message account is
    a read-only signer.
    """
    def ro(address: str) -> SolanaAccountMeta:
        return SolanaAccountMeta(address, False, False)

    def rw(address: str) -> SolanaAccountMeta:
        return SolanaAccountMeta(address, False, True)

    dispatched_message = derive_dispatched_message_pda(metadata.mailbox_program_address, unique_message)
    gas_payment = derive_gas_payment_pda(metadata.igp_program_address, unique_message)
    metas = [
        ro(SYSTEM_PROGRAM_ADDRESS),                       # 0
        ro(metadata.spl_noop_program_address),            # 1
        ro(metadata.token_pda),                           # 2
        ro(metadata.mailbox_program_address),             # 3
        rw(metadata.mailbox_outbox_pda),                  # 4
        ro(metadata.dispatch_authority_pda),              # 5
        SolanaAccountMeta(sender, True, True),            # 6 sender / fee payer
        SolanaAccountMeta(unique_message, True, False),   # 7 unique message (readonly signer)
        rw(dispatched_message),                           # 8
        ro(metadata.igp_program_address),                 # 9
        rw(metadata.igp_program_data_pda),                # 10
        rw(gas_payment),                                  # 11
    ]
    if metadata.igp_overhead_account is not None:
        metas.append(ro(metadata.igp_overhead_account))   # 12 (optional)
    metas.extend([
        rw(metadata.igp_account),                         # 13
        ro(SYSTEM_PROGRAM_ADDRESS),                       # 14
        rw(metadata.native_collateral_pda),               # 15
    ])
    return metas


# --- Program logs ------------------------------------------------------------------------------

# SEALEVEL_NOTES §5: only the Mailbox dispatch line carries the full id; the IGP and warp-completion
# lines print H256 with Display (truncated "0xffe0…7805") and must never be parsed.
DISPATCHED_MESSAGE_LOG_PATTERN = re.compile(r"Dispatched message to \d+, ID (0x[0-9a-fA-F]{64})")


def extract_hyperlane_message_id(logs: "list[str] | None") -> str | None:
    """The 32-byte Hyperlane message id from confirmed program logs, or None when absent."""
    if not logs:
        return None
    for line in logs:
        match = DISPATCHED_MESSAGE_LOG_PATTERN.search(line)
        if match:
            return match.group(1)
    return None
