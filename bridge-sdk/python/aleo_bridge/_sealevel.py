"""Pure Hyperlane sealevel (Solana) layouts for the SOL warp route.

Nothing here imports solders or solana-py: inputs and outputs are ``bytes``,
``int`` and base58 ``str`` so every layout is testable on an Aleo-only
install. Sources: veil ``src/solana/SEALEVEL_NOTES.md`` (primary-source
derivations against hyperlane-monorepo 45c0988), ``src/solana/transferRemote.ts``,
``src/solana/igp.ts``, ``src/protocols/hyperlane/solanaMetadata.ts``, and the
recorded mainnet deposit in ``tests/fixtures/sealevel-transfer-remote.json``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ._base58 import b58encode
from .errors import ConfigurationError, InvalidAmountError, InvalidRecipientError

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
