"""TestnetV0 bindings tests.

These require BOTH the mainnet and testnet extension modules to be installed
into the environment (see sdk/build-both.sh). They confirm that:

  * `aleo.testnet` imports and exposes the same class surface as `aleo.mainnet`,
  * the vendored key-derivation triples (network-agnostic encodings) reproduce
    identically under both networks,
  * `Network.id()` distinguishes the two (mainnet 0 / testnet 1).
"""

import aleo.mainnet as mainnet
import aleo.testnet as testnet
from conftest import load_vectors


# ---------------------------------------------------------------------------
# Import / surface
# ---------------------------------------------------------------------------

def test_testnet_importable():
    from aleo.testnet import PrivateKey, Address, ViewKey  # noqa: F401


def test_module_parity_same_classes():
    """Both networks expose the same public class names."""
    def public_classes(mod):
        return {
            name
            for name in dir(mod)
            if not name.startswith("_") and isinstance(getattr(mod, name), type)
        }

    m = public_classes(mainnet)
    t = public_classes(testnet)
    assert m == t, f"class surface differs: only mainnet={m - t}, only testnet={t - m}"


def test_common_classes_present():
    for name in ("PrivateKey", "ViewKey", "Address", "Signature", "Network"):
        assert hasattr(mainnet, name)
        assert hasattr(testnet, name)


# ---------------------------------------------------------------------------
# Key derivation is network-agnostic: the vendored triples reproduce on both.
# ---------------------------------------------------------------------------

def test_key_derivation_triples_parity():
    triples = load_vectors("accounts.json")["triples"]
    for t in triples:
        for mod in (mainnet, testnet):
            pk = mod.PrivateKey.from_string(t["private_key"])
            assert str(pk.view_key) == t["view_key"]
            assert str(pk.address) == t["address"]


def test_mainnet_testnet_derive_identically():
    triples = load_vectors("accounts.json")["triples"]
    for t in triples:
        m_pk = mainnet.PrivateKey.from_string(t["private_key"])
        t_pk = testnet.PrivateKey.from_string(t["private_key"])
        assert str(m_pk.address) == str(t_pk.address)
        assert str(m_pk.view_key) == str(t_pk.view_key)


# ---------------------------------------------------------------------------
# Network identity differs across networks.
# ---------------------------------------------------------------------------

def test_network_id_differs():
    assert mainnet.Network.id() == 0
    assert testnet.Network.id() == 1
