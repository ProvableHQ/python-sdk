import dataclasses

import pytest

pytest.importorskip("solders")
from solders.keypair import Keypair
from solders.message import to_bytes_versioned
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.transaction import VersionedTransaction

from aleo_bridge import _sealevel as sl
from aleo_bridge import sol
from aleo_bridge._calls import SolCall
from aleo_bridge.errors import (
    BridgeError,
    ConfigurationError,
    InsufficientBalanceError,
    RegistryVersionMismatchError,
    UnsupportedRouteError,
)
from aleo_bridge.sol import SolModule, Solana
from aleo_bridge.types import DispatchReceipt, Status
from tests.fakes.fake_solana import BLOCKHASH, STUB_SIGNATURE, FakeSignatureStatus, FakeSolanaClient, stub_bridge
from tests.fakes.sealevel_fixtures import EXPECTED_MESSAGE_ID, TRANSFER, WARP_PROGRAM_ADDRESS

RECIPIENT = TRANSFER["recipientAleoAddress"]
AMOUNT = TRANSFER["amountLamports"]
SIGNATURE = str(STUB_SIGNATURE)
CHECKPOINT_SOURCE_KEYS = {"transactionId", "blockhash", "lastValidBlockHeight"}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(sol.time, "sleep", lambda seconds: None)


def module(fake=None, *, signer=None, checkpoints=None):
    fake = fake or FakeSolanaClient()
    keypair = signer or Keypair()
    bridge = stub_bridge(checkpoints=checkpoints)
    return SolModule(bridge, Solana(client=fake, signer=keypair)), fake, keypair


class ExplodingStore:
    """A checkpoint store whose disk is full / read-only."""

    def __init__(self):
        self.attempts = []

    def save(self, checkpoint):
        self.attempts.append(checkpoint)
        raise OSError("read-only file system")

    def load(self, checkpoint_id):          # pragma: no cover - never reached
        return None

    def list(self):                          # pragma: no cover - never reached
        return []

    def delete(self, checkpoint_id):         # pragma: no cover - never reached
        return None


def test_extract_message_id_reads_only_the_mailbox_dispatch_line():
    assert sl.extract_hyperlane_message_id(TRANSFER["logMessages"]) == EXPECTED_MESSAGE_ID
    truncated = [line for line in TRANSFER["logMessages"] if "Dispatched message" not in line]
    assert any("Paid IGP" in line for line in truncated)
    assert sl.extract_hyperlane_message_id(truncated) is None
    assert sl.extract_hyperlane_message_id(None) is None
    assert sl.extract_hyperlane_message_id([]) is None


def test_build_returns_a_transaction_signed_only_by_the_unique_message_key():
    mod, fake, keypair = module()
    call = mod.transfer_remote(RECIPIENT, amount_atomic=AMOUNT)
    assert isinstance(call, SolCall)
    tx = call.build()
    assert isinstance(tx, VersionedTransaction)
    message = tx.message
    assert message.header.num_required_signatures == 2
    assert message.account_keys[0] == keypair.pubkey()                                   # fee payer first
    unique = Pubkey.from_string(call.quote.unique_message_address)
    assert message.account_keys[1] == unique
    assert tx.signatures[0] == Signature.default()                                        # fee payer unsigned
    assert tx.signatures[1] != Signature.default()
    assert tx.signatures[1].verify(unique, to_bytes_versioned(message))
    programs = [str(message.account_keys[ix.program_id_index]) for ix in message.instructions]
    assert programs == ["ComputeBudget111111111111111111111111111111", WARP_PROGRAM_ADDRESS]
    assert fake.sent == [] and call.quote.total_lamports == 676_207_914_240


def test_send_adds_the_fee_payer_signature_confirms_and_extracts_the_message_id():
    mod, fake, keypair = module()
    result = mod.transfer_remote(RECIPIENT, amount_atomic=AMOUNT).send()
    assert isinstance(result, DispatchReceipt)
    assert result.transaction_id == SIGNATURE and result.route_id == sl.SOLANA_ROUTE_ID
    assert result.message_id == EXPECTED_MESSAGE_ID and result.amount_atomic == AMOUNT
    receipt = result.receipt
    assert receipt.status is Status.DELIVERY_PENDING and receipt.id == EXPECTED_MESSAGE_ID
    assert receipt.source_tx_id == SIGNATURE and receipt.protocol == "hyperlane"
    assert receipt.protocol_state["messageId"] == EXPECTED_MESSAGE_ID
    assert "messageIdUnavailable" not in receipt.protocol_state
    assert len(fake.sent) == 1
    tx = VersionedTransaction.from_bytes(fake.sent[0])
    assert all(signature != Signature.default() for signature in tx.signatures)
    assert tx.signatures[0].verify(keypair.pubkey(), to_bytes_versioned(tx.message))
    tx.verify_and_hash_message()                                                          # raises if any signature is invalid
    opts = fake.sent_opts[0]
    assert isinstance(opts, sol.SendOptions)
    assert opts.skip_preflight is False and opts.preflight_commitment == "confirmed" and opts.skip_confirmation is True
    assert fake.status_calls == [True]                                                    # searchTransactionHistory
    assert fake.transaction_calls == [("confirmed", 0)]                                   # confirmed commitment, v0


def test_send_with_a_non_keypair_signer():
    inner = Keypair()

    class RemoteSigner:
        def pubkey(self):
            return inner.pubkey()

        def sign_message(self, message: bytes):
            return inner.sign_message(message)

    fake = FakeSolanaClient()
    mod = SolModule(stub_bridge(), Solana(client=fake, signer=RemoteSigner()))
    result = mod.transfer_remote(RECIPIENT, amount_atomic=1).send()
    tx = VersionedTransaction.from_bytes(fake.sent[0])
    assert tx.signatures[0].verify(inner.pubkey(), to_bytes_versioned(tx.message))
    assert result.receipt.status is Status.DELIVERY_PENDING


def test_send_refuses_a_plan_prepared_for_another_sender_before_any_read():
    mod, fake, _ = module()
    plan = mod.quote_transfer_remote(RECIPIENT, amount_atomic=1, sender=TRANSFER["senderAddress"]).plan
    fake.calls.clear()
    with pytest.raises(ConfigurationError, match=f"Prepared sender {TRANSFER['senderAddress']} does not match connected account"):
        mod.transfer_remote(RECIPIENT, amount_atomic=1, plan=plan).send()
    assert fake.calls == [] and fake.sent == []


def test_send_refuses_a_stale_or_foreign_plan():
    mod, fake, _ = module()
    plan = mod.quote_transfer_remote(RECIPIENT, amount_atomic=1).plan
    stale = dataclasses.replace(plan, registry_version="2020-01-01.stale.0")
    with pytest.raises(RegistryVersionMismatchError):
        mod.transfer_remote(RECIPIENT, plan=stale).send()
    foreign = dataclasses.replace(plan, route_id="hyperlane:ethereum/eth->aleo/eth")
    with pytest.raises(UnsupportedRouteError):
        mod.transfer_remote(RECIPIENT, plan=foreign).send()
    assert fake.sent == []


def test_send_uses_the_plan_when_it_matches_the_wallet():
    mod, fake, keypair = module()
    plan = mod.quote_transfer_remote(RECIPIENT, amount_atomic=3).plan
    assert plan.sender == str(keypair.pubkey())
    result = mod.transfer_remote(RECIPIENT, amount_atomic=3, plan=plan).send()
    assert result.receipt.protocol_state["quotedLamports"] == str(3 + 2_900_000 + 10_000 + 5_004_240)


def test_send_insufficient_balance_names_the_amount_gas_and_rent_split():
    mod, fake, _ = module(FakeSolanaClient(balance=0))
    with pytest.raises(InsufficientBalanceError) as excinfo:
        mod.transfer_remote(RECIPIENT, amount_atomic=AMOUNT).send()
    message = str(excinfo.value)
    for fragment in ("balance 0 lamports", "required 676207914240 lamports", "amount 676200000000", "gas 2910000", "rent 5004240"):
        assert fragment in message
    assert fake.sent == []


def test_send_checkpoints_source_confirming_before_the_first_status_read():
    mod, fake, _ = module()
    seen = []

    def on_checkpoint(checkpoint):
        fake.calls.append("checkpoint")
        seen.append(checkpoint)

    mod.transfer_remote(RECIPIENT, amount_atomic=AMOUNT).send(on_checkpoint=on_checkpoint)
    assert len(seen) == 1
    checkpoint = seen[0]
    assert checkpoint.id == SIGNATURE
    assert checkpoint.route == {"id": sl.SOLANA_ROUTE_ID, "registryVersion": mod.registry.version}
    assert set(checkpoint.source) == CHECKPOINT_SOURCE_KEYS
    assert checkpoint.source["transactionId"] == SIGNATURE
    assert checkpoint.source["blockhash"] == str(BLOCKHASH) and checkpoint.source["lastValidBlockHeight"] == "100"
    assert fake.calls.index("send_raw_transaction") < fake.calls.index("checkpoint") < fake.calls.index("get_signature_statuses")


def test_bound_store_saves_the_source_checkpoint_after_the_caller_callback():
    saved = []

    class RecordingStore:
        def save(self, checkpoint):
            saved.append(checkpoint)

        def load(self, checkpoint_id):
            return None

        def list(self):
            return []

        def delete(self, checkpoint_id):
            return None

    mod, fake, _ = module(checkpoints=RecordingStore())
    seen = []
    mod.transfer_remote(RECIPIENT, amount_atomic=1).send(on_checkpoint=seen.append)
    assert [cp.id for cp in saved] == [SIGNATURE] == [cp.id for cp in seen]


def test_store_failure_after_broadcast_reports_the_signature_and_never_hides_it():
    """The transaction is already on the wire: the caller's callback must have run first, the
    error must name the signature and the checkpoint, and no status poll may follow the failure."""
    store = ExplodingStore()
    mod, fake, _ = module(checkpoints=store)
    seen = []
    with pytest.raises(BridgeError) as excinfo:
        mod.transfer_remote(RECIPIENT, amount_atomic=1).send(on_checkpoint=seen.append)
    message = str(excinfo.value)
    assert SIGNATURE in message and "broadcast" in message and "checkpoint" in message.lower()
    assert [cp.id for cp in seen] == [SIGNATURE]                  # callback ran before the store
    assert [cp.id for cp in store.attempts] == [SIGNATURE]
    assert len(fake.sent) == 1                                    # broadcast happened exactly once
    assert "get_signature_statuses" not in fake.calls             # nothing polled after the failure


def test_send_failed_status_raises_naming_the_signature():
    mod, _, _ = module(FakeSolanaClient(statuses=[FakeSignatureStatus(err={"InstructionError": [2, "Custom"]})]))
    with pytest.raises(BridgeError, match=SIGNATURE):
        mod.transfer_remote(RECIPIENT, amount_atomic=1).send()


def test_send_timeout_returns_a_pending_source_confirming_receipt():
    mod, fake, _ = module(FakeSolanaClient(statuses=[None]))
    result = mod.transfer_remote(RECIPIENT, amount_atomic=1).send(timeout_seconds=0)
    receipt = result.receipt
    assert receipt.status is Status.SOURCE_CONFIRMING and receipt.id == SIGNATURE
    assert result.message_id is None and "messageId" not in receipt.protocol_state
    assert "get_transaction" not in fake.calls and len(fake.sent) == 1


def test_send_expired_blockhash_returns_expired_without_resubmitting():
    mod, fake, _ = module(FakeSolanaClient(statuses=[None], blockhash_valid=False))
    result = mod.transfer_remote(RECIPIENT, amount_atomic=1).send()
    receipt = result.receipt
    assert receipt.status is Status.EXPIRED
    assert receipt.protocol_state["blockhashExpired"] is True
    assert SIGNATURE in receipt.protocol_state["sourceError"]
    assert len(fake.sent) == 1


def test_send_swallows_transient_status_read_errors():
    fake = FakeSolanaClient(statuses=[RuntimeError("rate limited"), RuntimeError("again"), FakeSignatureStatus()], blockhash_valid=RuntimeError("advisory"))
    mod, _, _ = module(fake)
    result = mod.transfer_remote(RECIPIENT, amount_atomic=1).send(timeout_seconds=5, poll_seconds=0)
    assert result.receipt.status is Status.DELIVERY_PENDING and len(fake.status_calls) >= 3


def test_send_persistent_status_errors_time_out_to_pending():
    mod, _, _ = module(FakeSolanaClient(statuses=[RuntimeError("down")]))
    result = mod.transfer_remote(RECIPIENT, amount_atomic=1).send(timeout_seconds=0)
    assert result.receipt.status is Status.SOURCE_CONFIRMING


def test_send_without_wait_skips_polling():
    mod, fake, _ = module()
    result = mod.transfer_remote(RECIPIENT, amount_atomic=1).send(wait=False)
    assert result.receipt.status is Status.SOURCE_CONFIRMING and "get_signature_statuses" not in fake.calls


def test_finalized_counts_as_confirmed_and_missing_log_marks_message_id_unavailable():
    mod, _, _ = module(FakeSolanaClient(statuses=[FakeSignatureStatus(confirmation_status="finalized")], logs=[]))
    result = mod.transfer_remote(RECIPIENT, amount_atomic=1).send()
    receipt = result.receipt
    assert receipt.status is Status.DELIVERY_PENDING and receipt.id == SIGNATURE
    assert receipt.protocol_state["messageIdUnavailable"] is True and result.message_id is None
    processed_then_confirmed = FakeSolanaClient(statuses=[FakeSignatureStatus(confirmation_status="processed"), FakeSignatureStatus()])
    mod2, _, _ = module(processed_then_confirmed)
    assert mod2.transfer_remote(RECIPIENT, amount_atomic=1).send(poll_seconds=0).receipt.status is Status.DELIVERY_PENDING


def test_send_requires_a_signer():
    fake = FakeSolanaClient()
    read_only = SolModule(stub_bridge(), Solana(client=fake))
    with pytest.raises(ConfigurationError, match="read-only"):
        read_only.transfer_remote(RECIPIENT, amount_atomic=1).build()
    assert fake.sent == []
