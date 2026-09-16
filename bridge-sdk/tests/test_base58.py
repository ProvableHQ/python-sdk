import pytest

from aleo_bridge._base58 import b58decode, b58encode

SYSTEM_PROGRAM = "11111111111111111111111111111111"
WARP_PROGRAM = "8YGT2pZwyZe94qBpGzWfY2TMEVcwaQ1bXAE7YAgpUaM7"


def test_decode_known_solana_keys():
    assert b58decode(SYSTEM_PROGRAM) == bytes(32)
    raw = b58decode(WARP_PROGRAM)
    assert len(raw) == 32
    # First four bytes match the aleoRemoteRouterRecipient literal of hyperlane:aleo/sol->solana/sol
    assert list(raw[:4]) == [112, 4, 72, 22]
    assert list(raw[-4:]) == [7, 6, 211, 122]


def test_round_trip_and_leading_zero_handling():
    for value in (bytes(32), b"\x00\x00\x01", b"\x01\x00\x00", bytes(range(1, 33)), b""):
        assert b58decode(b58encode(value)) == value
    assert b58encode(b"") == ""
    assert b58encode(bytes(32)) == SYSTEM_PROGRAM
    assert b58encode(b58decode(WARP_PROGRAM)) == WARP_PROGRAM


def test_decode_rejects_bad_characters():
    for bad in ("not-base58!", "0OIl", "8YGT2pZwyZe94qBpGzWfY2TMEVcwaQ1bXAE7YAgpUaM7 "):
        with pytest.raises(ValueError):
            b58decode(bad)
