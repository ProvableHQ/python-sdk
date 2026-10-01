"""SPL-collateral ``TransferRemote``: the BAT/USDG/ZEC warp routes replace the native plugin's final two
accounts with the token program, mint, sender associated token account and escrow PDA.

Golden fixture: the successful ZEC Solana → Aleo transfer pinned by veil PR #169
(``tests/fixtures/sealevel-spl-collateral-transfer-remote.json``).
"""
import base64
import dataclasses

import pytest

from aleo_bridge import _sealevel as sl
from aleo_bridge.encoding import aleo_address_to_bytes32
from aleo_bridge.errors import BridgeError, RouteUnavailableError
from aleo_bridge.registry import DEFAULT_REGISTRY, Route
from tests.fakes.sealevel_fixtures import SPL_TRANSFER, metadata_from_fixture, spl_metadata_from_fixture

ACCOUNTS = SPL_TRANSFER["accounts"]
ZEC_ROUTE_ID = "hyperlane:solana/zec->aleo/zec"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
ZEC_MINT = "A7bdiYdS5GjqGFtxf17ppRHtDKPkkRqbKtR27dxvQXaS"


def route_with(metadata: dict, route_id: str = ZEC_ROUTE_ID) -> Route:
    return dataclasses.replace(DEFAULT_REGISTRY.route(route_id), availability="active", metadata=metadata)


def token_account_data(amount: int, length: int = 165) -> bytes:
    data = bytearray(length)
    data[64:72] = amount.to_bytes(8, "little")
    return bytes(data)


def test_instruction_data_matches_the_zec_fixture_byte_for_byte():
    data = sl.build_transfer_remote_instruction_data(
        1634493807, aleo_address_to_bytes32(SPL_TRANSFER["recipientAleoAddress"]), SPL_TRANSFER["amountAtomic"])
    assert base64.b64encode(data).decode() == SPL_TRANSFER["instructionDataBase64"]


def test_associated_token_address_matches_the_recorded_sender_account():
    ata = sl.derive_associated_token_address(SPL_TRANSFER["senderAddress"], ZEC_MINT, TOKEN_PROGRAM)
    assert ata == ACCOUNTS[16]["address"] == "FPAwNBT635S69zX3d8XLFDRBin4yHfVRkPhAvZN1kQ2K"


def test_associated_token_address_agrees_with_solders_for_both_token_programs():
    spl_token = pytest.importorskip("spl.token.instructions")
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey

    for program in (TOKEN_PROGRAM, TOKEN_2022_PROGRAM):
        for _ in range(4):
            owner, mint = Keypair().pubkey(), Keypair().pubkey()
            expected = spl_token.get_associated_token_address(owner, mint, Pubkey.from_string(program))
            assert sl.derive_associated_token_address(str(owner), str(mint), program) == str(expected)


def test_decode_spl_token_account_amount():
    assert sl.decode_spl_token_account_amount(None, "x") == 0           # uncreated account → zero balance
    assert sl.decode_spl_token_account_amount(token_account_data(166_575), "x") == 166_575
    assert sl.decode_spl_token_account_amount(token_account_data(2**64 - 1, length=72), "x") == 2**64 - 1
    assert sl.decode_spl_token_account_amount(token_account_data(7, length=300), "x") == 7   # Token-2022 extensions
    with pytest.raises(BridgeError, match="invalid data: FPAw"):
        sl.decode_spl_token_account_amount(bytes(71), "FPAwNBT635S69zX3d8XLFDRBin4yHfVRkPhAvZN1kQ2K")


def test_spl_route_metadata_parses_the_collateral_fields():
    metadata = sl.solana_route_metadata(route_with(spl_metadata_from_fixture()))
    assert metadata.router_type == "spl-collateral"
    assert metadata.native_collateral_pda is None
    assert (metadata.spl_token_program_address, metadata.collateral_mint_address, metadata.escrow_pda) == (
        TOKEN_PROGRAM, ZEC_MINT, ACCOUNTS[17]["address"])
    assert metadata.igp_overhead_account == ACCOUNTS[12]["address"]
    assert metadata.destination_gas_amount == 460_000


def test_native_route_metadata_is_unchanged_and_accepts_an_explicit_router_type():
    native = sl.solana_route_metadata(route_with(metadata_from_fixture(), sl.SOLANA_ROUTE_ID))
    assert native.router_type == "native" and native.native_collateral_pda is not None
    assert native.spl_token_program_address is None and native.collateral_mint_address is None and native.escrow_pda is None
    explicit = sl.solana_route_metadata(route_with({**metadata_from_fixture(), "routerType": "native"}, sl.SOLANA_ROUTE_ID))
    assert explicit == native


def test_spl_route_metadata_validation():
    for field in ("splTokenProgramAddress", "collateralMintAddress", "escrowPda"):
        broken = {**spl_metadata_from_fixture(), field: "not-a-solana-address"}
        with pytest.raises(RouteUnavailableError, match=f"invalid {field}"):
            sl.solana_route_metadata(route_with(broken))
        missing = spl_metadata_from_fixture()
        del missing[field]
        with pytest.raises(RouteUnavailableError, match=f"invalid {field}"):
            sl.solana_route_metadata(route_with(missing))
    with pytest.raises(RouteUnavailableError, match="invalid routerType"):
        sl.solana_route_metadata(route_with({**metadata_from_fixture(), "routerType": "synthetic"}, sl.SOLANA_ROUTE_ID))
    # a native route still needs its collateral PDA
    native = metadata_from_fixture()
    del native["nativeCollateralPda"]
    with pytest.raises(RouteUnavailableError, match="invalid nativeCollateralPda"):
        sl.solana_route_metadata(route_with(native, sl.SOLANA_ROUTE_ID))


def test_account_metas_reproduce_the_recorded_zec_transfer():
    metadata = sl.solana_route_metadata(route_with(spl_metadata_from_fixture()))
    metas = sl.account_metas(metadata, SPL_TRANSFER["senderAddress"], SPL_TRANSFER["uniqueMessageAddress"])
    assert len(metas) == 18
    assert [m.to_dict() for m in metas] == ACCOUNTS


def test_default_registry_zec_route_carries_the_recorded_deployment():
    live = sl.solana_route_metadata(DEFAULT_REGISTRY.route(ZEC_ROUTE_ID))
    recorded = sl.solana_route_metadata(route_with(spl_metadata_from_fixture()))
    for field in ("router_type", "warp_program_address", "token_pda", "dispatch_authority_pda", "mailbox_program_address",
                  "mailbox_outbox_pda", "igp_program_address", "igp_program_data_pda", "igp_account", "igp_overhead_account",
                  "spl_noop_program_address", "spl_token_program_address", "collateral_mint_address", "escrow_pda",
                  "destination_domain", "destination_gas_amount", "registry_commit"):
        assert getattr(live, field) == getattr(recorded, field), field
    metas = sl.account_metas(live, SPL_TRANSFER["senderAddress"], SPL_TRANSFER["uniqueMessageAddress"])
    assert [m.to_dict() for m in metas] == ACCOUNTS


@pytest.mark.parametrize("asset,token_program", [("bat", TOKEN_PROGRAM), ("usdg", TOKEN_2022_PROGRAM), ("zec", TOKEN_PROGRAM)])
def test_default_registry_spl_routes_bind_their_mint_and_token_program(asset, token_program):
    route = DEFAULT_REGISTRY.route(f"hyperlane:solana/{asset}->aleo/{asset}")
    metadata = sl.solana_route_metadata(route)
    assert metadata.router_type == "spl-collateral"
    assert metadata.spl_token_program_address == token_program
    assert metadata.collateral_mint_address == DEFAULT_REGISTRY.asset(route.source_asset_id).locator.value
    # the escrow is the warp program's collateral PDA for its mint (hyperlane-monorepo: ["hyperlane_token","-","escrow"])
    assert sl.find_program_address([b"hyperlane_token", b"-", b"escrow"], metadata.warp_program_address)[0] == metadata.escrow_pda
