import dataclasses

import pytest

from aleo_bridge import _sealevel as sl
from aleo_bridge.errors import RouteUnavailableError
from aleo_bridge.registry import DEFAULT_REGISTRY, Route
from tests.fakes.sealevel_fixtures import TRANSFER, WARP_PROGRAM_ADDRESS, metadata_from_fixture

ACCOUNTS = TRANSFER["accounts"]


def route_with(metadata: dict) -> Route:
    base = DEFAULT_REGISTRY.route(sl.SOLANA_ROUTE_ID)
    return dataclasses.replace(base, availability="active", metadata=metadata)


def test_pdas_match_the_recorded_transaction():
    unique = TRANSFER["uniqueMessageAddress"]
    assert sl.derive_dispatched_message_pda(ACCOUNTS[3]["address"], unique) == ACCOUNTS[8]["address"]
    assert sl.derive_gas_payment_pda(ACCOUNTS[9]["address"], unique) == ACCOUNTS[11]["address"]
    dispatched, bump_dispatched = sl.find_program_address(
        [*sl.DISPATCHED_MESSAGE_SEED_PREFIX, sl.b58decode(unique)], ACCOUNTS[3]["address"])
    gas_payment, bump_gas = sl.find_program_address(
        [*sl.GAS_PAYMENT_SEED_PREFIX, sl.b58decode(unique)], ACCOUNTS[9]["address"])
    assert (dispatched, bump_dispatched) == (ACCOUNTS[8]["address"], 253)   # SEALEVEL_NOTES §2 bump seeds
    assert (gas_payment, bump_gas) == (ACCOUNTS[11]["address"], 255)


def test_route_static_pdas_recompute_from_the_warp_program():
    # The registry's tokenPda / dispatchAuthorityPda / nativeCollateralPda / outbox / IGP program-data are
    # PDAs of the warp, mailbox and IGP programs (SEALEVEL_NOTES §2-3); bumps 255, 254, 255, 255, 254.
    assert sl.find_program_address([b"hyperlane_message_recipient", b"-", b"handle", b"-", b"account_metas"], WARP_PROGRAM_ADDRESS) == (ACCOUNTS[2]["address"], 255)
    assert sl.find_program_address([b"hyperlane_dispatcher", b"-", b"dispatch_authority"], WARP_PROGRAM_ADDRESS) == (ACCOUNTS[5]["address"], 254)
    assert sl.find_program_address([b"hyperlane_token", b"-", b"native_collateral"], WARP_PROGRAM_ADDRESS) == (ACCOUNTS[15]["address"], 255)
    assert sl.find_program_address([b"hyperlane", b"-", b"outbox"], ACCOUNTS[3]["address"]) == (ACCOUNTS[4]["address"], 255)
    assert sl.find_program_address([b"hyperlane_igp", b"-", b"program_data"], ACCOUNTS[9]["address"]) == (ACCOUNTS[10]["address"], 254)


def test_pda_derivation_agrees_with_solders_on_random_keys():
    solders_keypair = pytest.importorskip("solders.keypair")
    from solders.pubkey import Pubkey

    mailbox = Pubkey.from_string(ACCOUNTS[3]["address"])
    for _ in range(16):
        unique = solders_keypair.Keypair().pubkey()
        seeds = [*sl.DISPATCHED_MESSAGE_SEED_PREFIX, bytes(unique)]
        expected, expected_bump = Pubkey.find_program_address(seeds, mailbox)
        assert sl.find_program_address(seeds, str(mailbox)) == (str(expected), expected_bump)


def test_find_program_address_rejects_oversized_seeds():
    with pytest.raises(sl.BridgeError, match="32 bytes"):
        sl.find_program_address([bytes(33)], WARP_PROGRAM_ADDRESS)
    with pytest.raises(sl.BridgeError, match="16 seeds"):
        sl.find_program_address([b"x"] * 17, WARP_PROGRAM_ADDRESS)


def test_account_metas_reproduce_the_recorded_16_accounts():
    metadata = sl.solana_route_metadata(route_with(metadata_from_fixture()))
    metas = sl.account_metas(metadata, TRANSFER["senderAddress"], TRANSFER["uniqueMessageAddress"])
    assert len(metas) == 16
    assert [m.to_dict() for m in metas] == ACCOUNTS


def test_account_metas_omit_the_optional_overhead_slot():
    metadata = sl.solana_route_metadata(route_with(metadata_from_fixture(overhead=False)))
    metas = sl.account_metas(metadata, TRANSFER["senderAddress"], TRANSFER["uniqueMessageAddress"])
    overhead = ACCOUNTS[12]["address"]
    assert len(metas) == 15
    assert overhead not in [m.address for m in metas]
    assert [m.to_dict() for m in metas] == [a for a in ACCOUNTS if a["address"] != overhead]


def test_default_registry_route_carries_the_recorded_deployment():
    live = sl.solana_route_metadata(DEFAULT_REGISTRY.route(sl.SOLANA_ROUTE_ID))
    recorded = sl.solana_route_metadata(route_with(metadata_from_fixture()))
    for field in ("warp_program_address", "token_pda", "native_collateral_pda", "dispatch_authority_pda",
                  "mailbox_program_address", "mailbox_outbox_pda", "igp_program_address", "igp_program_data_pda",
                  "igp_account", "igp_overhead_account", "spl_noop_program_address", "destination_domain",
                  "destination_gas_amount", "registry_commit"):
        assert getattr(live, field) == getattr(recorded, field), field
    assert live.destination_gas_amount == 464_000
    assert live.igp_overhead_account == "AkeHBbE5JkwVppujCQQ6WuxsVsJtruBAjUo6fDCFp6fF"


def test_route_metadata_validation():
    inactive = dataclasses.replace(DEFAULT_REGISTRY.route(sl.SOLANA_ROUTE_ID), availability="metadata-required")
    with pytest.raises(RouteUnavailableError, match="not executable"):
        sl.solana_route_metadata(inactive)
    with pytest.raises(RouteUnavailableError, match="invalid igpAccount"):
        sl.solana_route_metadata(route_with({**metadata_from_fixture(), "igpAccount": "not-a-solana-address"}))
    with pytest.raises(RouteUnavailableError, match="invalid destinationDomain"):
        sl.solana_route_metadata(route_with({**metadata_from_fixture(), "destinationDomain": "1634493807"}))
    with pytest.raises(RouteUnavailableError, match="invalid destinationGasAmount"):
        sl.solana_route_metadata(route_with({**metadata_from_fixture(), "destinationGasAmount": "lots"}))
    with pytest.raises(RouteUnavailableError, match="invalid registryCommit"):
        sl.solana_route_metadata(route_with({**metadata_from_fixture(), "registryCommit": "418056e2"}))
    missing = metadata_from_fixture()
    del missing["mailboxOutboxPda"]
    with pytest.raises(RouteUnavailableError, match="invalid mailboxOutboxPda"):
        sl.solana_route_metadata(route_with(missing))
    xreserve = DEFAULT_REGISTRY.route("xreserve:ethereum/usdc->aleo/usdcx")
    with pytest.raises(RouteUnavailableError, match="Hyperlane"):
        sl.solana_route_metadata(xreserve)
