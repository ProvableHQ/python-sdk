import pytest

from aleo_bridge._keccak import keccak256


def test_keccak_known_answers():
    assert keccak256(b"").hex() == "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    assert keccak256(b"abc").hex() == "4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"
    # Hyperlane Mailbox event topic — pinned by the whole ecosystem.
    assert keccak256(b"DispatchId(bytes32)").hex() == \
        "788dbc1b7152732178210e7f4d9d010ef016f9eafbe66786bd7169f56e0c353a"


def test_keccak_accepts_bytearray_and_memoryview():
    assert keccak256(bytearray(b"abc")) == keccak256(b"abc")
    assert keccak256(memoryview(b"abc")) == keccak256(b"abc")


def test_keccak_long_input_spans_blocks_and_agrees_with_web3():
    web3 = pytest.importorskip("web3")  # dev extra; the primitive still ships without it
    for data in (bytes(range(256)) * 3, bytes(135), bytes(136), bytes(137), b"x" * 1000):
        assert keccak256(data) == bytes(web3.Web3.keccak(data))
