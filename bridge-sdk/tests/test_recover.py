import json

import pytest

from aleo_bridge.checkpoint import Checkpoint, FileCheckpointStore, create_checkpoint
from aleo_bridge.errors import (CheckpointInvalidError, RegistryVersionMismatchError, UnsupportedRouteError)
from aleo_bridge.lifecycle import prepare, recover
from aleo_bridge.types import Receipt, Status
from tests.fakes.fake_bridge import ALEO_RECIPIENT, EVM_ADDRESS, SOL_ADDRESS, FakeBridge

EVM1 = EVM_ADDRESS


def _aleo_eth_checkpoint(b, **source):
    plan = prepare(b.registry, source_chain="aleo", source_asset="eth", destination_chain="ethereum", destination_asset="eth", amount="0.000000000000000001",
                   recipient=EVM1)
    return plan, {"version": 1,
                  "intent": {"source": {"chain": "aleo", "asset": "eth"}, "destination": {"chain": "ethereum", "asset": "eth"},
                             "bridgeProtocol": "hyperlane", "amount": plan.amount, "recipient": plan.recipient},
                  "route": {"id": plan.route_id, "registryVersion": plan.registry_version},
                  "source": source}


def test_version_route_and_registry_checks():
    b = FakeBridge(ethereum=False)
    plan, cp = _aleo_eth_checkpoint(b, transactionId="at1x")
    with pytest.raises(CheckpointInvalidError, match="version"):
        recover(b, {**cp, "version": 2})
    with pytest.raises(CheckpointInvalidError, match="does not match"):
        recover(b, {**cp, "route": {**cp["route"], "id": "hyperlane:aleo/wbtc->ethereum/wbtc"}})
    with pytest.raises(RegistryVersionMismatchError):
        recover(b, {**cp, "route": {**cp["route"], "registryVersion": "2020-01-01.old"}})
    with pytest.raises(CheckpointInvalidError, match="intent"):
        recover(b, {**cp, "intent": {"source": {"chain": "aleo"}}})


def test_prepared_but_unbroadcast_aleo_transaction_resumes_without_network():
    b = FakeBridge(ethereum=False)
    serialized = json.dumps({"type": "execute", "id": "at1prepared", "fee": {}})
    plan, cp = _aleo_eth_checkpoint(b, preparedTransaction={"transactionId": "at1prepared",
                                                            "serializedTransaction": serialized})
    cp["deliveryVerification"] = {"balanceBeforeAtomic": "100", "expectedIncreaseAtomic": "1"}
    progress = recover(b, json.dumps(cp))                       # JSON string accepted
    assert progress.next == "resume" and progress.receipt.status is Status.SOURCE_SUBMISSION_PENDING
    assert progress.receipt.id == "at1prepared"
    assert progress.receipt.protocol_state == {"routeId": plan.route_id, "preparedTransaction": serialized,
                                               "destinationBalanceBeforeAtomic": "100",
                                               "expectedDestinationIncreaseAtomic": "1"}
    assert b.calls == [] and b.events == []
    with pytest.raises(CheckpointInvalidError, match="id does not match"):
        recover(b, {**cp, "source": {"preparedTransaction": {"transactionId": "at1other", "serializedTransaction": serialized}}})
    with pytest.raises(CheckpointInvalidError, match="invalid prepared"):
        recover(b, {**cp, "source": {"preparedTransaction": {"transactionId": "at1prepared", "serializedTransaction": "{not json"}}})
    with pytest.raises(CheckpointInvalidError, match="invalid for a prepared Aleo"):
        recover(b, {**cp, "destination": {"transactionId": "at1d"}})


def test_submitted_aleo_source_is_observed_once_never_rebroadcast():
    b = FakeBridge(ethereum=False)
    plan, cp = _aleo_eth_checkpoint(b, transactionId="at1burn")
    progress = recover(b, Checkpoint.from_dict(cp))
    assert progress.next == "wait" and progress.receipt.status is Status.SOURCE_CONFIRMING
    b.aleo.confirmed_transactions["at1burn"] = {"status": "accepted"}
    progress = recover(b, cp)
    assert progress.next == "wait" and progress.receipt.status is Status.DELIVERY_PENDING
    assert progress.receipt.source_tx_id == "at1burn" and b.submitted == []
    # Checkpoint.from_dict (which we do not modify) derives receiptId from source/destination when
    # absent, and raises its own "no submitted or prepared transaction" error before recover() ever
    # sees an empty source — so a receiptId must be supplied for recover()'s OWN check to be reached.
    with pytest.raises(CheckpointInvalidError, match="no submitted source transaction"):
        recover(b, {**cp, "receiptId": "at1burn", "source": {}})
    with pytest.raises(CheckpointInvalidError, match="invalid for an Aleo"):
        recover(b, {**cp, "source": {"transactionId": "at1burn", "approvalTransactionIds": ["0x1"]}})


def test_solana_checkpoint_validates_blockhash_pair_and_reads_status():
    b = FakeBridge(solana=True)
    plan = prepare(b.registry, source_chain="solana", source_asset="sol", destination_chain="aleo", destination_asset="sol", amount="0.000000001",
                   recipient=ALEO_RECIPIENT, sender=SOL_ADDRESS)
    cp = create_checkpoint(plan, Receipt(id="sig", protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id="sig",
                                         protocol_state={"routeId": plan.route_id, "blockhash": "recent",
                                                         "lastValidBlockHeight": "123456789"}), b.registry)
    b.sol.source_status_result = Receipt(id="sig", protocol="hyperlane", status=Status.EXPIRED, source_tx_id="sig",
                                         protocol_state={"routeId": plan.route_id, "blockhashExpired": True,
                                                         "sourceError": "Solana transaction expired before confirmation: sig"})
    progress = recover(b, cp)
    assert progress.next == "failed" and progress.receipt.status is Status.EXPIRED
    assert progress.error == "Solana transaction expired before confirmation: sig"
    assert b.calls == [("sol.source_status", Status.SOURCE_CONFIRMING)]
    d = cp.to_dict()
    with pytest.raises(CheckpointInvalidError, match="blockhash"):
        recover(b, {**d, "source": {"transactionId": "sig", "blockhash": "recent"}})
    with pytest.raises(CheckpointInvalidError, match="no submitted source transaction"):
        recover(b, {**d, "source": {"blockhash": "recent", "lastValidBlockHeight": "1"}})


def test_evm_hyperlane_delegates_to_eth_recover_source():
    b = FakeBridge()
    plan = prepare(b.registry, source_chain="ethereum", source_asset="wbtc", destination_chain="aleo", destination_asset="wbtc", amount="0.001", recipient=ALEO_RECIPIENT,
                   sender=EVM1)
    cp = create_checkpoint(plan, Receipt(id="0x" + "11" * 32, protocol="hyperlane", status=Status.SOURCE_APPROVAL_PENDING,
                                         protocol_state={"routeId": plan.route_id, "approvalTxIds": ["0x" + "11" * 32]}),
                           b.registry)
    b.eth.recover_result = Receipt(id="0x" + "11" * 32, protocol="hyperlane", status=Status.SOURCE_SUBMISSION_PENDING,
                                   protocol_state={"routeId": plan.route_id, "approvalTxIds": ["0x" + "11" * 32],
                                                   "sourceSender": EVM1})
    progress = recover(b, cp)
    assert progress.next == "resume" and b.calls[0][0] == "eth.recover_source" and b.calls[0][2] is False
    with pytest.raises(CheckpointInvalidError, match="destination transaction"):
        recover(b, {**cp.to_dict(), "destination": {"transactionId": "at1x"}})


def test_evm_xreserve_recovery_paths():
    b = FakeBridge(environment="testnet")
    plan = prepare(b.registry, source_chain="sepolia", source_asset="usdc", destination_chain="aleo-testnet", destination_asset="usdcx", amount="2",
                   recipient=ALEO_RECIPIENT, mint_mode="private", sender=EVM1)
    base = create_checkpoint(plan, Receipt(id="0x" + "22" * 32, protocol="xreserve", status=Status.SOURCE_CONFIRMING,
                                           source_tx_id="0x" + "22" * 32, protocol_state={"routeId": plan.route_id}),
                             b.registry).to_dict()
    attested = Receipt(id="0x" + "cc" * 32, protocol="xreserve", status=Status.ATTESTATION_PENDING, source_tx_id="0x" + "22" * 32,
                       protocol_state={"routeId": plan.route_id, "mintMode": "private", "intendedRecipient": ALEO_RECIPIENT,
                                       "messageHash": "0x" + "cc" * 32, "nonce": "0x" + "dd" * 32,
                                       "bridgeProgram": "test_usdcx_bridge_v2.aleo"})
    b.eth.recover_result = attested
    # ATTESTATION_PENDING → one get_status (Circle not ready) → wait
    assert recover(b, base).next == "wait"
    assert [c[0] for c in b.calls] == ["eth.recover_source", "xreserve.is_delivered", "xreserve.get_attestation"]
    # submitted destination tx → DESTINATION_CONFIRMING, observed once
    b.calls.clear()
    b.aleo.confirmed_transactions["at1private"] = {"status": "accepted"}
    done = recover(b, {**base, "destination": {"transactionId": "at1private"}})
    assert done.next == "done" and done.receipt.destination_tx_id == "at1private"
    # both prepared and submitted destination → invalid
    serialized = json.dumps({"type": "execute", "id": "at1mint", "fee": {}})
    with pytest.raises(CheckpointInvalidError, match="both prepared and submitted"):
        recover(b, {**base, "destination": {"transactionId": "at1private",
                                            "preparedTransaction": {"transactionId": "at1mint", "serializedTransaction": serialized}}})
    # prepared destination survives only while DESTINATION_ACTION_REQUIRED
    b.xreserve.delivered_nonces.clear()
    with pytest.raises(CheckpointInvalidError, match="no longer valid"):
        recover(b, {**base, "destination": {"preparedTransaction": {"transactionId": "at1mint", "serializedTransaction": serialized}}})
    from aleo_bridge.types import Attestation
    b.xreserve.attestations["0x" + "cc" * 32] = Attestation(b"\x00" * 305, bytes.fromhex("cc" * 32), b"\x11" * 65, "complete")
    ready = recover(b, {**base, "destination": {"preparedTransaction": {"transactionId": "at1mint", "serializedTransaction": serialized}}})
    assert ready.next == "complete" and ready.receipt.id == "at1mint"
    assert ready.receipt.protocol_state["preparedDestinationTransaction"] == serialized


def test_unsupported_route_and_terminal_cleanup(tmp_path):
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(ethereum=False, checkpoints=store)
    plan, cp = _aleo_eth_checkpoint(b, transactionId="at1burn")
    store.save(Checkpoint.from_dict(cp))
    b.aleo.confirmed_transactions["at1burn"] = {"status": "rejected"}
    progress = recover(b, cp)
    assert progress.next == "failed" and store.list() == []
    burn = prepare(b.registry, source_chain="aleo", source_asset="usdcx", destination_chain="ethereum", destination_asset="usdc", amount="2.1", recipient=EVM1)
    ok = recover(b, create_checkpoint(burn, Receipt(id="at1b", protocol="xreserve", status=Status.SOURCE_CONFIRMING,
                                                    source_tx_id="at1b", protocol_state={"routeId": burn.route_id}), b.registry))
    assert ok.next == "wait"


def test_malformed_delivery_verification_raises_checkpoint_invalid():
    # Item 8 (carried from Task 7 review): a hand-edited/foreign checkpoint whose
    # deliveryVerification block is missing a key or holds a non-digit value must raise
    # CheckpointInvalidError, never KeyError, before any network read.
    b = FakeBridge(ethereum=False)
    plan, cp = _aleo_eth_checkpoint(b, transactionId="at1burn")
    with pytest.raises(CheckpointInvalidError, match="destination balance verification"):
        recover(b, {**cp, "deliveryVerification": {"balanceBeforeAtomic": "100"}})  # missing key
    with pytest.raises(CheckpointInvalidError, match="destination balance verification"):
        recover(b, {**cp, "deliveryVerification": {"balanceBeforeAtomic": "100", "expectedIncreaseAtomic": "abc"}})
    with pytest.raises(CheckpointInvalidError, match="destination balance verification"):
        recover(b, {**cp, "deliveryVerification": {"balanceBeforeAtomic": None, "expectedIncreaseAtomic": "1"}})
    assert b.calls == [] and b.events == []


def test_terminal_cleanup_deletes_by_checkpoint_id_not_receipt_id(tmp_path):
    # Item 7 (carried from Task 7 review): a Solana checkpoint whose stored id (the source
    # signature) differs from the id the refreshed receipt ends up carrying (EXPIRED status
    # can flip the receipt id to a message id). _finish must delete the record keyed on
    # cp.id ("sig"), never one keyed on receipt.id ("msg-divergent").
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(solana=True, checkpoints=store)
    plan = prepare(b.registry, source_chain="solana", source_asset="sol", destination_chain="aleo", destination_asset="sol", amount="0.000000001",
                   recipient=ALEO_RECIPIENT, sender=SOL_ADDRESS)
    cp = create_checkpoint(plan, Receipt(id="sig", protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                         source_tx_id="sig", protocol_state={"routeId": plan.route_id,
                                                                             "blockhash": "recent",
                                                                             "lastValidBlockHeight": "123456789"}),
                           b.registry)
    assert cp.id == "sig"
    store.save(cp)
    b.sol.source_status_result = Receipt(id="msg-divergent", protocol="hyperlane", status=Status.EXPIRED,
                                         source_tx_id="sig",
                                         protocol_state={"routeId": plan.route_id,
                                                         "sourceError": "Solana transaction expired before confirmation: sig"})
    progress = recover(b, cp)
    assert progress.next == "failed" and progress.receipt.id == "msg-divergent"
    assert store.load("sig") is None
    assert store.load("msg-divergent") is None      # nothing was ever stored under this key to begin with


def test_recovered_plan_round_trips_through_checkpoint():
    # Controller ruling (task-7-controller-notes.md #1): _plan_from_intent rebuilds the plan via
    # prepare(), which is proven field-identical to build_plan for every active route
    # (tests/test_prepare.py::test_prepare_equals_build_plan_for_every_active_route). Confirm the
    # round trip (create_checkpoint -> to_dict -> recover's internal from_dict/_plan_from_intent)
    # holds for an EVM, a Solana and an Aleo-origin route by checking recover()'s Progress.plan.
    b = FakeBridge(solana=True)

    evm_plan = prepare(b.registry, source_chain="ethereum", source_asset="wbtc", destination_chain="aleo", destination_asset="wbtc", amount="0.001",
                       recipient=ALEO_RECIPIENT, sender=EVM1)
    evm_cp = create_checkpoint(evm_plan, Receipt(id="0x" + "11" * 32, protocol="hyperlane",
                                                 status=Status.SOURCE_APPROVAL_PENDING,
                                                 protocol_state={"routeId": evm_plan.route_id,
                                                                 "approvalTxIds": ["0x" + "11" * 32]}), b.registry)
    b.eth.recover_result = Receipt(id="0x" + "11" * 32, protocol="hyperlane", status=Status.SOURCE_SUBMISSION_PENDING,
                                   protocol_state={"routeId": evm_plan.route_id, "approvalTxIds": ["0x" + "11" * 32]})
    assert recover(b, evm_cp).plan == evm_plan

    sol_plan = prepare(b.registry, source_chain="solana", source_asset="sol", destination_chain="aleo", destination_asset="sol", amount="0.000000001",
                       recipient=ALEO_RECIPIENT, sender=SOL_ADDRESS)
    sol_cp = create_checkpoint(sol_plan, Receipt(id="sig", protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                                 source_tx_id="sig", protocol_state={"routeId": sol_plan.route_id}),
                               b.registry)
    assert recover(b, sol_cp).plan == sol_plan

    aleo_plan, aleo_cp = _aleo_eth_checkpoint(b, transactionId="at1burn")
    assert recover(b, aleo_cp).plan == aleo_plan
