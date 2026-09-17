import dataclasses

import pytest

pytest.importorskip("solders")

from aleo_bridge.errors import BridgeError, CheckpointInvalidError
from aleo_bridge.sol import SolModule, Solana
from aleo_bridge.types import Receipt, Status
from tests.fakes.fake_solana import BLOCKHASH, STUB_SIGNATURE, FakeSignatureStatus, FakeSolanaClient, stub_bridge
from tests.fakes.sealevel_fixtures import EXPECTED_MESSAGE_ID, TRANSFER

SIGNATURE = str(STUB_SIGNATURE)


def setup(fake=None):
    fake = fake or FakeSolanaClient()
    mod = SolModule(stub_bridge(), Solana(client=fake))
    plan = mod.quote_transfer_remote(TRANSFER["recipientAleoAddress"], amount_atomic=1, sender=TRANSFER["senderAddress"]).plan
    fake.calls.clear()
    return mod, fake, plan


def receipt(plan, **state) -> Receipt:
    return Receipt(id=SIGNATURE, protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id=SIGNATURE,
                   protocol_state={"routeId": plan.route_id, **state})


def test_confirmed_signature_advances_to_delivery_pending_with_message_id():
    mod, fake, plan = setup()
    out = mod.source_status(plan, receipt(plan))
    assert out.status is Status.DELIVERY_PENDING and out.id == EXPECTED_MESSAGE_ID
    assert out.protocol_state["messageId"] == EXPECTED_MESSAGE_ID and out.source_tx_id == SIGNATURE
    assert fake.status_calls == [True] and fake.calls == ["get_signature_statuses", "get_transaction"]


def test_finalized_without_log_marks_message_id_unavailable():
    mod, _, plan = setup(FakeSolanaClient(statuses=[FakeSignatureStatus(confirmation_status="finalized")], no_logs=True))
    out = mod.source_status(plan, receipt(plan))
    assert out.status is Status.DELIVERY_PENDING and out.id == SIGNATURE and out.protocol_state["messageIdUnavailable"] is True


def test_processed_and_unknown_without_lifetime_are_unchanged():
    mod, _, plan = setup(FakeSolanaClient(statuses=[FakeSignatureStatus(confirmation_status="processed")]))
    original = receipt(plan)
    assert mod.source_status(plan, original) == original
    mod2, fake2, plan2 = setup(FakeSolanaClient(statuses=[None]))
    assert mod2.source_status(plan2, receipt(plan2)) == receipt(plan2)
    assert "is_blockhash_valid" not in fake2.calls


def test_unknown_signature_with_lifetime_checks_the_blockhash():
    lifetime = {"blockhash": str(BLOCKHASH), "lastValidBlockHeight": "100"}
    mod, fake, plan = setup(FakeSolanaClient(statuses=[None], blockhash_valid=True))
    assert mod.source_status(plan, receipt(plan, **lifetime)) == receipt(plan, **lifetime)
    assert "is_blockhash_valid" in fake.calls
    expired_mod, _, plan = setup(FakeSolanaClient(statuses=[None], blockhash_valid=False))
    out = expired_mod.source_status(plan, receipt(plan, **lifetime))
    assert out.status is Status.EXPIRED and out.protocol_state["blockhashExpired"] is True
    assert SIGNATURE in out.protocol_state["sourceError"]
    flaky_mod, _, plan = setup(FakeSolanaClient(statuses=[None], blockhash_valid=RuntimeError("rpc")))
    assert flaky_mod.source_status(plan, receipt(plan, **lifetime)) == receipt(plan, **lifetime)


def test_malformed_lifetime_is_a_checkpoint_error():
    mod, _, plan = setup(FakeSolanaClient(statuses=[None]))
    with pytest.raises(CheckpointInvalidError, match="blockhash lifetime"):
        mod.source_status(plan, receipt(plan, blockhash=str(BLOCKHASH)))
    with pytest.raises(CheckpointInvalidError, match="blockhash lifetime"):
        mod.source_status(plan, receipt(plan, blockhash=str(BLOCKHASH), lastValidBlockHeight="soon"))
    with pytest.raises(CheckpointInvalidError, match="blockhash lifetime"):
        mod.source_status(plan, receipt(plan, blockhash="not-base58-0OIl", lastValidBlockHeight="100"))


def test_failed_signature_raises_and_status_read_errors_propagate():
    mod, _, plan = setup(FakeSolanaClient(statuses=[FakeSignatureStatus(err={"InstructionError": [1, "Custom"]})]))
    with pytest.raises(BridgeError, match=SIGNATURE):
        mod.source_status(plan, receipt(plan))
    down, _, plan = setup(FakeSolanaClient(statuses=[RuntimeError("rpc down")]))
    with pytest.raises(RuntimeError):
        down.source_status(plan, receipt(plan))


def test_guards():
    mod, _, plan = setup()
    with pytest.raises(BridgeError, match="source-confirming"):
        mod.source_status(plan, dataclasses.replace(receipt(plan), status=Status.DELIVERY_PENDING))
    with pytest.raises(BridgeError, match="source-confirming"):
        mod.source_status(plan, dataclasses.replace(receipt(plan), source_tx_id=None))
    with pytest.raises(BridgeError, match="source-confirming"):
        mod.source_status(plan, dataclasses.replace(receipt(plan), protocol="xreserve"))
    with pytest.raises(BridgeError, match="does not match"):
        mod.source_status(plan, Receipt(id=SIGNATURE, protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                        source_tx_id=SIGNATURE, protocol_state={"routeId": "hyperlane:ethereum/eth->aleo/eth"}))
