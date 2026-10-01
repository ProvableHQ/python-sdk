import dataclasses

import pytest

pytest.importorskip("solders")
from solders.keypair import Keypair

from aleo_bridge import _sealevel as sl
from aleo_bridge.errors import (
    BridgeError,
    ConfigurationError,
    InvalidAmountError,
    InvalidRecipientError,
    RegistryVersionMismatchError,
    RouteNotFoundError,
    RouteUnavailableError,
    UnsupportedRouteError,
)
from aleo_bridge.sol import SolModule, Solana
from aleo_bridge.types import SolanaHyperlaneQuote
from tests.fakes.fake_solana import FakeSolanaClient, stub_bridge
from tests.fakes.sealevel_fixtures import (
    DISPATCHED_MESSAGE_RENT_LAMPORTS,
    EXPECTED_IGP_PAYMENT_LAMPORTS,
    FEE_PAYER_RENT_LAMPORTS,
    GAS_PAYMENT_RENT_LAMPORTS,
    NETWORK_FEE_LAMPORTS,
    TRANSFER,
    WARP_PROGRAM_ADDRESS,
)

RECIPIENT = TRANSFER["recipientAleoAddress"]
SENDER = TRANSFER["senderAddress"]
AMOUNT = TRANSFER["amountLamports"]
RENT = GAS_PAYMENT_RENT_LAMPORTS + DISPATCHED_MESSAGE_RENT_LAMPORTS + FEE_PAYER_RENT_LAMPORTS


def module(fake=None, *, signer=None, environment="mainnet") -> tuple[SolModule, FakeSolanaClient]:
    fake = fake or FakeSolanaClient()
    return SolModule(stub_bridge(environment), Solana(client=fake, signer=signer)), fake


def test_quote_with_pinned_sender_on_read_only_connection():
    mod, fake = module()
    quote = mod.quote_transfer_remote(RECIPIENT, amount="676.2", sender=SENDER)
    assert isinstance(quote, SolanaHyperlaneQuote) and quote.kind == "solana-hyperlane"
    assert quote.igp_lamports == EXPECTED_IGP_PAYMENT_LAMPORTS
    assert quote.network_fee_lamports == NETWORK_FEE_LAMPORTS
    assert quote.rent_lamports == RENT == 5_004_240
    assert quote.total_lamports == AMOUNT + EXPECTED_IGP_PAYMENT_LAMPORTS + NETWORK_FEE_LAMPORTS + RENT == 676_207_914_240
    assert quote.plan.route_id == sl.SOLANA_ROUTE_ID
    assert quote.plan.sender == SENDER and quote.plan.recipient == RECIPIENT
    assert quote.plan.amount == "676.2" and quote.plan.amount_atomic == AMOUNT
    assert quote.plan.mint_mode == "public" and [s.id for s in quote.plan.steps] == ["source-dispatch", "message-delivery", "destination-confirmation"]
    assert quote.amount_out == "676.2"
    assert sl.SOLANA_PUBKEY_RE.match(quote.unique_message_address)
    assert [(f.kind, f.amount) for f in quote.fees] == [("interchain-gas", "0.0029"), ("network", "0.00001"), ("rent", "0.00500424")]
    assert "send_raw_transaction" not in fake.calls and "get_balance" not in fake.calls


def test_quote_fee_message_is_a_v0_message_with_compute_budget_and_transfer_remote():
    mod, fake = module(signer=Keypair())
    quote = mod.quote_transfer_remote(RECIPIENT, amount_atomic=1)
    assert quote.plan.sender == mod.conn.address
    message = fake.fee_messages[0]
    assert message.header.num_required_signatures == 2
    assert message.header.num_readonly_signed_accounts == 1
    programs = [str(message.account_keys[ix.program_id_index]) for ix in message.instructions]
    assert programs == ["ComputeBudget111111111111111111111111111111", WARP_PROGRAM_ADDRESS]
    assert bytes(message.instructions[0].data) == bytes.fromhex("02801a0600")      # SetComputeUnitLimit(400_000)
    assert len(bytes(message.instructions[1].data)) == 77
    assert str(message.recent_blockhash) == WARP_PROGRAM_ADDRESS
    assert fake.calls.count("get_minimum_balance_for_rent_exemption") == 3


def test_quote_uses_amount_atomic_and_rejects_ambiguous_amounts():
    mod, _ = module()
    assert mod.quote_transfer_remote(RECIPIENT, amount_atomic=1, sender=SENDER).plan.amount == "0.000000001"
    with pytest.raises(InvalidAmountError):
        mod.quote_transfer_remote(RECIPIENT, sender=SENDER)
    with pytest.raises(InvalidAmountError):
        mod.quote_transfer_remote(RECIPIENT, amount="1", amount_atomic=1, sender=SENDER)
    with pytest.raises(InvalidAmountError):
        mod.quote_transfer_remote(RECIPIENT, amount_atomic=0, sender=SENDER)


def test_quote_error_paths():
    mod, _ = module()
    with pytest.raises(ConfigurationError, match="sender"):
        mod.quote_transfer_remote(RECIPIENT, amount_atomic=1)
    with pytest.raises(InvalidRecipientError):
        mod.quote_transfer_remote("aleo1notanaddress", amount_atomic=1, sender=SENDER)
    missing_igp, _ = module(FakeSolanaClient(accounts={}))
    with pytest.raises(BridgeError, match="IGP account does not exist"):
        missing_igp.quote_transfer_remote(RECIPIENT, amount_atomic=1, sender=SENDER)
    testnet, _ = module(environment="testnet")
    with pytest.raises(RouteNotFoundError):
        testnet.quote_transfer_remote(RECIPIENT, amount_atomic=1, sender=SENDER)


def test_quote_with_plan_checks_registry_version_and_reuses_the_plan():
    mod, _ = module()
    plan = mod.quote_transfer_remote(RECIPIENT, amount_atomic=5, sender=SENDER).plan
    quoted = mod.quote_transfer_remote(RECIPIENT, plan=plan)
    assert quoted.plan is plan and quoted.total_lamports == 5 + EXPECTED_IGP_PAYMENT_LAMPORTS + NETWORK_FEE_LAMPORTS + RENT
    stale = dataclasses.replace(plan, registry_version="2020-01-01.stale.0")
    with pytest.raises(RegistryVersionMismatchError):
        mod.quote_transfer_remote(RECIPIENT, plan=stale)


def test_a_plan_alone_supplies_recipient_amount_and_sender():
    mod, _ = module()
    plan = mod.quote_transfer_remote(RECIPIENT, amount_atomic=5, sender=SENDER).plan
    quoted = mod.quote_transfer_remote(plan=plan)                  # no positional recipient
    assert quoted.plan is plan and quoted.plan.sender == SENDER
    assert quoted.total_lamports == 5 + EXPECTED_IGP_PAYMENT_LAMPORTS + NETWORK_FEE_LAMPORTS + RENT


def test_a_plan_is_mutually_exclusive_with_sender_and_a_differing_amount():
    mod, _ = module()
    plan = mod.quote_transfer_remote(RECIPIENT, amount_atomic=5, sender=SENDER).plan
    with pytest.raises(ValueError, match="plan"):
        mod.quote_transfer_remote(plan=plan, sender=SENDER)
    with pytest.raises(ValueError, match="plan"):
        mod.quote_transfer_remote(plan=plan, amount_atomic=6)
    assert mod.quote_transfer_remote(plan=plan, amount_atomic=5).plan is plan       # identical is fine


def test_neither_a_plan_nor_a_recipient_is_refused():
    mod, fake = module()
    with pytest.raises(InvalidRecipientError, match="recipient is required when no plan is given"):
        mod.quote_transfer_remote(amount_atomic=1, sender=SENDER)
    with pytest.raises(InvalidRecipientError, match="recipient is required when no plan is given"):
        mod.transfer_remote(amount_atomic=1)
    assert fake.sent == []


def test_a_malformed_json_private_key_never_carries_the_secret_into_the_traceback():
    """A chained JSONDecodeError keeps the whole document in .doc — which IS the private key."""
    from aleo_bridge.sol import keypair_from_private_key

    secret = "[17,42,99,128"                                       # a truncated solana-cli id.json
    with pytest.raises(ConfigurationError) as excinfo:
        keypair_from_private_key(secret)
    exc = excinfo.value
    assert exc.__cause__ is None and exc.__context__ is None
    assert "17" not in repr(exc) and secret not in repr(exc)


def test_a_malformed_base58_private_key_never_carries_the_secret_into_the_traceback():
    from aleo_bridge.sol import keypair_from_private_key

    secret = "5JueXBoJHvOoPeKeYsEcReT"
    with pytest.raises(ConfigurationError) as excinfo:
        keypair_from_private_key(secret)
    exc = excinfo.value
    assert exc.__cause__ is None and exc.__context__ is None
    assert secret not in repr(exc)


def test_balance_reads_the_connected_wallet():
    keypair = Keypair()
    mod, fake = module(FakeSolanaClient(balance=42), signer=keypair)
    assert mod.balance() == 42
    read_only, _ = module()
    with pytest.raises(ConfigurationError, match="read-only"):
        read_only.balance()


def test_balance_reads_public_recipient_without_signer():
    from unittest.mock import Mock
    mod, _ = module()
    mod._balance_of = Mock(return_value=42)
    recipient = str(Keypair().pubkey())
    assert mod.balance(address=recipient) == 42
    mod._balance_of.assert_called_once_with(recipient)


# --- SPL collateral (veil PR #169) ---------------------------------------------------------------

ZEC_ROUTE_ID = "hyperlane:solana/zec->aleo/zec"
ZEC_MINT = "A7bdiYdS5GjqGFtxf17ppRHtDKPkkRqbKtR27dxvQXaS"
ZEC_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
ZEC_SENDER = "D4jZ2sNktKgTrhWVMnjZb5BXP7MMh9N3y5ZLwkyKfozb"


def test_spl_quote_reports_the_token_amount_and_excludes_it_from_the_sol_total():
    """veil quoteSolanaHyperlaneTransfer.test.ts: 'quotes SPL collateral in token atomic units without adding
    it to the SOL requirement'. Every fee is SOL; the amount is ZEC."""
    mod, fake = module()
    quote = mod.quote_transfer_remote(RECIPIENT, amount="0.0001", sender=ZEC_SENDER, asset="solana/zec")
    assert quote.plan.route_id == ZEC_ROUTE_ID and quote.plan.amount_atomic == 10_000 and quote.plan.amount == "0.0001"
    assert quote.plan.source_asset_id == "solana/zec" and quote.plan.destination_asset_id == "aleo/zec"
    assert quote.igp_lamports > 0 and quote.network_fee_lamports == NETWORK_FEE_LAMPORTS and quote.rent_lamports == RENT
    assert quote.total_lamports == quote.igp_lamports + quote.network_fee_lamports + quote.rent_lamports
    assert all(f.asset_id == "solana/sol" and f.chain_id == "solana" for f in quote.fees)
    assert [f.kind for f in quote.fees] == ["interchain-gas", "network", "rent"]
    assert quote.amount_out == "0.0001"
    assert [s.kind for s in quote.plan.steps] == ["dispatch", "wait-delivery", "confirm-delivery"]
    # the fee-estimate message compiles the 18-account SPL instruction against the ZEC warp program
    message = fake.fee_messages[0]
    programs = [str(message.account_keys[ix.program_id_index]) for ix in message.instructions]
    assert programs[-1] == "2RBzic8nUNJ8KngRRbsCEjkeM9CtpQN2CCqU1cs1n2y5"
    assert len(message.instructions[1].accounts) == 18


def test_spl_quote_igp_uses_the_route_gas_amount():
    mod, _ = module()
    sol_quote = mod.quote_transfer_remote(RECIPIENT, amount_atomic=1, sender=SENDER)
    zec_quote = mod.quote_transfer_remote(RECIPIENT, amount_atomic=1, sender=ZEC_SENDER, asset="zec")
    # same oracle, 460_000 gas (OverheadIgp 160_000 + 300_000) vs the SOL route's 464_000
    assert zec_quote.igp_lamports == sl.igp_lamports(
        sl.decode_igp_account(__import__("tests.fakes.sealevel_fixtures", fromlist=["igp_account_data"]).igp_account_data()).gas_oracles[1634493807],
        460_000)
    assert zec_quote.igp_lamports < sol_quote.igp_lamports


def test_asset_selects_the_route_and_is_exclusive_with_a_plan():
    mod, _ = module()
    assert mod.outbound_route().id == sl.SOLANA_ROUTE_ID
    assert mod.outbound_route("zec").id == mod.outbound_route("solana/zec").id == mod.outbound_route(("solana", "ZEC")).id == ZEC_ROUTE_ID
    assert mod.outbound_route("usdg").id == "hyperlane:solana/usdg->aleo/usdg" and mod.outbound_route("BAT").id == "hyperlane:solana/bat->aleo/bat"
    with pytest.raises(RouteNotFoundError):
        mod.outbound_route("aleo/zec")                        # not a Solana asset
    with pytest.raises(RouteUnavailableError, match="metadata-required"):
        mod.outbound_route("solana/aleo")
    plan = mod.quote_transfer_remote(RECIPIENT, amount_atomic=10_000, sender=ZEC_SENDER, asset="solana/zec").plan
    with pytest.raises(ValueError, match="plan"):
        mod.quote_transfer_remote(plan=plan, asset="solana/zec")
    requoted = mod.quote_transfer_remote(plan=plan)
    assert requoted.plan is plan and requoted.total_lamports == requoted.igp_lamports + NETWORK_FEE_LAMPORTS + RENT
    foreign = dataclasses.replace(plan, route_id="hyperlane:aleo/zec->solana/zec", source_asset_id="aleo/zec", destination_asset_id="solana/zec")
    with pytest.raises(UnsupportedRouteError, match="Solana Hyperlane"):
        mod.quote_transfer_remote(plan=foreign)


def test_spl_balance_reads_the_associated_token_account_and_defaults_to_sol():
    keypair = Keypair()
    owner = str(keypair.pubkey())
    ata = sl.derive_associated_token_address(owner, ZEC_MINT, ZEC_TOKEN_PROGRAM)
    data = bytearray(165)
    data[64:72] = (166_575).to_bytes(8, "little")
    fake = FakeSolanaClient(balance=7, accounts={ata: bytes(data)})
    mod, _ = module(fake, signer=keypair)
    assert mod.balance() == 7 and mod.balance("solana/sol") == 7
    assert mod.balance("solana/zec") == 166_575 and mod.balance("zec", address=owner) == 166_575
    assert mod.balance("solana/usdg") == 0                     # uncreated Token-2022 account → zero
    assert mod.balance("solana/bat", address=SENDER) == 0
    with pytest.raises(UnsupportedRouteError, match="locator"):
        mod.balance("solana/aleo")
