"""Frozen addresses against the REAL testnet USDCx list (``test_usdcx_freezelist.aleo``).

``FROZEN_ADDRESS_1`` / ``FROZEN_ADDRESS_2`` hold the private keys of two accounts that are (or
will be) frozen there.  While the list does not contain them the matching checks skip and say so;
once it does they prove that the resolver refuses to build a proof for a frozen address and that a
USDCx unshield or private xReserve burn signed by a frozen account is refused before anything is
proved or paid, and that the SDK-built bundle authorizes locally.  The program compares the
reconstructed root with its stored root in the *finalize*, on chain, so a bad bundle passed through
``compliance_proofs=`` is only rejected after the fee is paid — which is why the SDK verifies every
witness it builds against the root before a call is prepared.

No test here submits a transaction.  Gates: the frozen keys; ``ALEO_E2E_PRIVATE_KEY`` for the
simulate check (that account holds testnet USDCx records and shares its view key with the hosted
scanner already).
"""
from __future__ import annotations

import os

import pytest

from aleo import Aleo, HTTPProvider, testnet
from aleo.compliance import address_to_field_int
from aleo_bridge import Bridge
from aleo_bridge.errors import ConfigurationError

pytestmark = pytest.mark.live

ENDPOINT = os.environ.get("ALEO_ENDPOINT", "https://edge.provable.com/api")
ASSET = "aleo-testnet/usdcx"
TOKEN = "test_usdcx_stablecoin.aleo"
LIST = "test_usdcx_freezelist.aleo"
SEPOLIA_RECIPIENT = "0x" + "11" * 20


def _frozen_keys() -> list[tuple[str, str]]:
    found = [(name, os.environ[name]) for name in ("FROZEN_ADDRESS_1", "FROZEN_ADDRESS_2") if os.environ.get(name)]
    if not found:
        pytest.skip("set FROZEN_ADDRESS_1 / FROZEN_ADDRESS_2 (private keys of the frozen test accounts)")
    return found


def _bridge(private_key: str) -> Bridge:
    aleo = Aleo(HTTPProvider(ENDPOINT, network="testnet"))
    aleo.default_account = aleo.account.from_private_key(private_key)
    return Bridge(aleo)


def _is_leaf(bridge: Bridge, address: str) -> bool:
    tree = bridge.aleo.network.get_program_freeze_list(LIST)
    assert tree is not None
    return address_to_field_int(address) in tree[: (len(tree) + 1) // 2]


def test_frozen_addresses_are_listed_and_refused():
    listed_any = False
    for name, key in _frozen_keys():
        bridge = _bridge(key)
        address = bridge.aleo_address()
        listed = _is_leaf(bridge, address)
        listed_any |= listed
        print(f"  {name} {address[:12]}… on {LIST}: {'FROZEN' if listed else 'not listed'}")
        if listed:
            with pytest.raises(ConfigurationError, match="is on the freeze list"):
                bridge.freezelist.exclusion_proof(address, TOKEN)
        else:
            assert bridge.freezelist.exclusion_proof(address, TOKEN).count("leaf_index") == 2
    if not listed_any:
        pytest.skip(f"{LIST} does not contain the frozen test accounts yet")


def test_frozen_signer_unshield_and_private_burn_are_refused_before_proving():
    for _, key in _frozen_keys():
        bridge = _bridge(key)
        address = bridge.aleo_address()
        if not _is_leaf(bridge, address):
            continue
        # A record literal stands in for a scan: the refusal comes before the record is ever used.
        record = f"{{ owner: {address}.private, amount: 5000000u128.private, _nonce: 0group.public }}"
        with pytest.raises(ConfigurationError, match="is on the freeze list"):
            bridge.unshield(ASSET, amount="1", record=record)
        with pytest.raises(ConfigurationError, match="is on the freeze list"):
            bridge.xreserve.burn(SEPOLIA_RECIPIENT, amount="3", record=record)
        return
    pytest.skip(f"{LIST} does not contain the frozen test accounts yet")


def test_sdk_built_bundle_authorizes_locally():
    key = os.environ.get("ALEO_E2E_PRIVATE_KEY")
    if not key:
        pytest.skip("ALEO_E2E_PRIVATE_KEY (testnet account holding USDCx records) is required")
    bridge = _bridge(key)
    bridge.aleo.records.register(bridge.aleo.default_account)
    call = bridge.unshield(ASSET, amount="0.01")
    assert call.function_name == "transfer_private_to_public"
    assert call.inputs[-1] == bridge.freezelist.exclusion_proof(bridge.aleo_address(), TOKEN)
    call.simulate()
