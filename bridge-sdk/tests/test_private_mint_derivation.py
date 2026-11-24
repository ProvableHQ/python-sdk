"""Shield-compatible private mint derivation (veil PR #166, shield-core cc29a60).

The six vectors in ``tests/fixtures/shield-core-private-mint.json`` were produced by Shield's own
``deriveShieldedUsdcxScalar`` / ``deriveShieldedUsdcxAddressCommitment``; the bridge must reproduce
them byte for byte so an identity minted here is one Shield could reproduce.
"""
import json
from pathlib import Path

import pytest

from aleo_bridge import private_mint as pm
from aleo_bridge.encoding import xreserve_hook_data
from aleo_bridge.errors import ConfigurationError

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "shield-core-private-mint.json").read_text())
VIEW_KEY = "AViewKey1sqm952gJj1tmAWySYDQvSv2NmfnyEMvU6a9ZBCuyG7PN"
RECIPIENT = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"


@pytest.mark.parametrize("vector", FIXTURE["vectors"], ids=lambda v: f"{v['network']}-{v['counter']}")
def test_shield_vectors_reproduce(vector):
    nonce = pm.derive_secret_nonce(VIEW_KEY, vector["counter"], vector["network"])
    assert nonce == vector["nonce"]
    commitment = pm.derive_address_commitment(RECIPIENT, nonce, vector["network"])
    assert commitment == vector["commitment"]
    hook = xreserve_hook_data("private", RECIPIENT, vector["network"], nonce)
    assert hook == b"\x02" + bytes.fromhex(commitment) + b"\x00" * 32
    assert pm.commitment_from_hook_data(hook) == commitment


def test_domain_fields_are_little_endian_ascii_labels():
    for network, expected in (("mainnet", pm.SHIELDED_USDCX_DOMAIN_MAINNET),
                              ("testnet", pm.SHIELDED_USDCX_DOMAIN_TESTNET)):
        n = int.from_bytes(f"shielded_usdcx_{network}".encode(), "little")
        assert pm.domain_field(network) == f"{n}field" == expected
    assert pm.SHIELDED_USDCX_DOMAIN_MAINNET != pm.SHIELDED_USDCX_DOMAIN_TESTNET


def test_view_key_object_and_string_derive_the_same_nonce():
    import aleo
    vk = aleo.mainnet.ViewKey.from_string(VIEW_KEY)
    assert pm.derive_secret_nonce(vk, 7, "mainnet") == pm.derive_secret_nonce(VIEW_KEY, 7, "mainnet")


@pytest.mark.parametrize("counter", [-1, 0x1_0000_0000, 1.5, True])
def test_counter_must_be_u32(counter):
    with pytest.raises(ConfigurationError, match="u32"):
        pm.derive_secret_nonce(VIEW_KEY, counter, "mainnet")


def test_unknown_network_and_bad_view_key_are_configuration_errors():
    with pytest.raises(ConfigurationError):
        pm.derive_secret_nonce(VIEW_KEY, 0, "devnet")
    with pytest.raises(ConfigurationError, match="view key"):
        pm.derive_secret_nonce("AViewKey1nope", 0, "mainnet")


@pytest.mark.parametrize("hook", [
    b"\x02" + b"\x11" * 32 + b"\x01" + b"\x00" * 31,     # reserved tail is not zero
    b"\x00" * 65,                                        # public hook, not a private commitment
    b"\x02" + b"\x11" * 32,                              # wrong width
])
def test_commitment_from_hook_data_rejects_non_private_or_malformed_hooks(hook):
    with pytest.raises(ConfigurationError):
        pm.commitment_from_hook_data(hook)


def test_identity_repr_never_shows_the_scalar():
    identity = pm.PrivateMintIdentity(counter=3, recipient=RECIPIENT, secret_nonce="7scalar",
                                      address_commitment="ab" * 32)
    text = repr(identity)
    assert "7scalar" not in text and "ab" * 32 in text and "3" in text
