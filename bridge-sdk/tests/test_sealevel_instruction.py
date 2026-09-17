import base64

import pytest

from aleo_bridge import _sealevel as sl
from aleo_bridge.encoding import aleo_address_to_bytes32
from aleo_bridge.errors import BridgeError
from tests.fakes.sealevel_fixtures import ALEO_MAINNET_DOMAIN, TRANSFER

RECIPIENT32_HEX = "1c3496991e7c611ced5ee5cd0cdee969c53efc8a5497ae050819b1ef00ed2912"


def test_instruction_data_matches_mainnet_fixture_byte_for_byte():
    data = sl.build_transfer_remote_instruction_data(
        ALEO_MAINNET_DOMAIN,
        aleo_address_to_bytes32(TRANSFER["recipientAleoAddress"]),
        TRANSFER["amountLamports"],
    )
    assert len(data) == sl.INSTRUCTION_DATA_BYTES == 77
    assert base64.b64encode(data).decode() == TRANSFER["instructionDataBase64"]


def test_instruction_data_layout_offsets():
    data = sl.build_transfer_remote_instruction_data(ALEO_MAINNET_DOMAIN, bytes.fromhex(RECIPIENT32_HEX), 676_200_000_000)
    assert data[0:8] == bytes([1] * 8) == sl.PROGRAM_INSTRUCTION_DISCRIMINATOR
    assert data[8] == sl.TRANSFER_REMOTE_VARIANT_TAG == 1
    assert data[9:13] == bytes.fromhex("6f656c61")            # 0x616c656f little-endian
    assert data[13:45] == bytes.fromhex(RECIPIENT32_HEX)      # raw bech32m payload, no reversal
    assert data[45:77] == (676_200_000_000).to_bytes(32, "little")
    assert data[45:49] == bytes.fromhex("002aa970")


def test_instruction_data_rejects_bad_inputs():
    with pytest.raises(BridgeError, match="32 bytes"):
        sl.build_transfer_remote_instruction_data(ALEO_MAINNET_DOMAIN, bytes(31), 1)
    with pytest.raises(BridgeError, match="32-byte unsigned"):
        sl.build_transfer_remote_instruction_data(ALEO_MAINNET_DOMAIN, bytes(32), 1 << 256)
    with pytest.raises(BridgeError, match="32-byte unsigned"):
        sl.build_transfer_remote_instruction_data(ALEO_MAINNET_DOMAIN, bytes(32), -1)
    with pytest.raises(BridgeError, match="destination domain"):
        sl.build_transfer_remote_instruction_data(1 << 32, bytes(32), 1)
