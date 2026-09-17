"""Pure Hyperlane sealevel (Solana) layouts for the SOL warp route.

Nothing here imports solders or solana-py: inputs and outputs are ``bytes``,
``int`` and base58 ``str`` so every layout is testable on an Aleo-only
install. Sources: veil ``src/solana/SEALEVEL_NOTES.md`` (primary-source
derivations against hyperlane-monorepo 45c0988), ``src/solana/transferRemote.ts``,
``src/solana/igp.ts``, ``src/protocols/hyperlane/solanaMetadata.ts``, and the
recorded mainnet deposit in ``tests/fixtures/sealevel-transfer-remote.json``.
"""
from __future__ import annotations

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
