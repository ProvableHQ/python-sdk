"""Tests for W4d2 wasm-parity additions."""
import pytest
from aleo.mainnet import Proof


def test_proof_from_invalid_string():
    with pytest.raises(Exception):
        Proof.from_string("not_a_proof")


def test_proof_from_invalid_bytes():
    with pytest.raises(Exception):
        Proof.from_bytes(bytes(10))
