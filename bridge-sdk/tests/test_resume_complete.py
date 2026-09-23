"""``lifecycle.resume`` and ``lifecycle.complete`` — the two caller-boundary verbs.

``resume`` continues an interrupted SOURCE leg: an Aleo leg rebroadcasts the checkpointed bytes
byte-for-byte (a duplicate answer means the first broadcast won and counts as success), an EVM leg
re-scans source history first and only authorizes the one step history proves is still missing.
``complete`` submits the single user-signed Aleo destination transaction a private USDCx mint
needs, and is idempotent through the same rebroadcast rule.

The funds-critical invariants under test: neither verb ever repeats an irreversible step, veil's
two resume guards (hook-data commitment, surviving allowance) refuse rather than guess, a private
resume/complete without its ``secret_nonce`` is refused before any RPC, and the secret nonce never
reaches a checkpoint, a protocol_state, or a Progress.
"""
import json

import pytest
from aleo import AleoNetworkError

from aleo_bridge._calls import is_duplicate_submission
from aleo_bridge.checkpoint import FileCheckpointStore
from aleo_bridge.errors import (AttestationError, CheckpointInvalidError, ConfigurationError,
                                NotResumableError)
from aleo_bridge.lifecycle import (complete, is_duplicate_broadcast_error, prepare, resume,
                                   submit_serialized)
from aleo_bridge.types import Receipt, Status, to_progress
from tests.fakes.fake_bridge import ALEO_RECIPIENT, EVM_ADDRESS, SOL_ADDRESS, FakeBridge
from tests.test_get_status import SIG, _inbound_private

APPROVAL = "0x" + "11" * 32
DROPPED_DISPATCH = "0x" + "22" * 32


def _prepared_progress(b):
    """An Aleo-origin transfer proved but never broadcast — exactly what ``execute`` checkpoints
    between ``delegate_prepared`` and ``submit_prepared``."""
    plan = prepare(b.registry, source_chain="aleo", source_asset="eth", destination_chain="ethereum", destination_asset="eth",
                   amount="0.000000000000000001", recipient=EVM_ADDRESS)
    serialized = json.dumps({"type": "execute", "id": "at1prepared", "fee": {}})
    receipt = Receipt(id="at1prepared", protocol="hyperlane", status=Status.SOURCE_SUBMISSION_PENDING,
                      protocol_state={"routeId": plan.route_id, "preparedTransaction": serialized,
                                      "destinationBalanceBeforeAtomic": "100",
                                      "expectedDestinationIncreaseAtomic": "1"})
    return plan, serialized, to_progress(plan, receipt)


def _xreserve_progress(b, **plan_kw):
    plan = prepare(b.registry, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx", amount="2",
                   recipient=ALEO_RECIPIENT, sender=EVM_ADDRESS, **plan_kw)
    receipt = Receipt(id=APPROVAL, protocol="xreserve", status=Status.SOURCE_SUBMISSION_PENDING,
                      protocol_state={"routeId": plan.route_id, "approvalTxIds": [APPROVAL],
                                      "sourceSender": EVM_ADDRESS,
                                      "hookData": "0x" + b.eth.hook_data.hex()})
    return plan, receipt


# ── duplicate-broadcast classification (plan 1's rule, not a second one) ──────

def test_duplicate_detection_is_plan_1s_rule_and_refuses_double_spend_shapes():
    assert is_duplicate_broadcast_error is is_duplicate_submission
    assert is_duplicate_broadcast_error(AleoNetworkError("Transaction 'at1x' already exists in the ledger", status=400))
    assert is_duplicate_broadcast_error(AleoNetworkError("transaction at1x already exists in the memory pool"))
    # a DIFFERENT transaction colliding with this one's records is a real failure, never success
    assert not is_duplicate_broadcast_error(AleoNetworkError("Duplicate serial number found in transaction"))
    assert not is_duplicate_broadcast_error(AleoNetworkError("duplicate output id in transaction"))
    assert not is_duplicate_broadcast_error(AleoNetworkError("Duplicate transaction at1x"))
    assert not is_duplicate_broadcast_error(AleoNetworkError("Invalid transaction: fee verification failed", status=400))


def test_submit_serialized_refuses_an_id_the_node_did_not_echo():
    b = FakeBridge(ethereum=False)
    serialized = json.dumps({"type": "execute", "id": "at1prepared", "fee": {}})
    assert submit_serialized(b, serialized, "at1prepared") == "at1prepared"
    b.aleo.network.submit_transaction = lambda tx: "at1other"
    with pytest.raises(CheckpointInvalidError, match="expected at1prepared"):
        submit_serialized(b, serialized, "at1prepared")


# ── resume: Aleo source ───────────────────────────────────────────────────────

def test_resume_rebroadcasts_identical_bytes_and_treats_a_duplicate_as_success():
    b = FakeBridge(ethereum=False)
    plan, serialized, progress = _prepared_progress(b)
    cps = []
    out = resume(b, progress, on_checkpoint=cps.append)
    assert b.aleo.submitted == [serialized]                       # byte-for-byte, never re-proved
    assert out.next == "wait" and out.receipt.status is Status.SOURCE_CONFIRMING
    assert out.receipt.source_tx_id == "at1prepared"
    assert out.receipt.protocol_state == {"routeId": plan.route_id,
                                          "destinationBalanceBeforeAtomic": "100",
                                          "expectedDestinationIncreaseAtomic": "1"}   # bytes discarded
    assert len(cps) == 1 and cps[0].source == {"transactionId": "at1prepared"}
    assert cps[0].delivery_verification == {"balanceBeforeAtomic": "100", "expectedIncreaseAtomic": "1"}

    # the node says it already knows this transaction: the earlier broadcast won the race
    b2 = FakeBridge(ethereum=False)
    b2.aleo.duplicate_on_submit = True
    assert resume(b2, progress).receipt.status is Status.SOURCE_CONFIRMING

    # any other node rejection is a real failure and propagates
    b3 = FakeBridge(ethereum=False)

    def invalid(tx):
        raise AleoNetworkError("Invalid transaction: fee verification failed", status=400)

    b3.aleo.network.submit_transaction = invalid
    with pytest.raises(AleoNetworkError):
        resume(b3, progress)

    # a node that answers with a different id never gets recorded as this transfer's transaction
    b4 = FakeBridge(ethereum=False)
    b4.aleo.network.submit_transaction = lambda tx: "at1other"
    with pytest.raises(CheckpointInvalidError, match="expected at1prepared"):
        resume(b4, progress)


def test_resume_replaces_the_prepared_checkpoint_in_the_bound_store(tmp_path):
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(ethereum=False, checkpoints=store)
    plan, serialized, progress = _prepared_progress(b)
    from aleo_bridge.checkpoint import create_checkpoint
    store.save(create_checkpoint(plan, progress.receipt, b.registry))
    assert store.list()[0].source["preparedTransaction"]["serializedTransaction"] == serialized
    resume(b, progress)
    saved = store.list()
    assert [c.id for c in saved] == ["at1prepared"]
    assert saved[0].source == {"transactionId": "at1prepared"}      # the unbroadcast bytes are gone


def test_resume_refuses_a_wrong_state_a_foreign_route_and_mismatched_bytes():
    b = FakeBridge(ethereum=False)
    plan, serialized, progress = _prepared_progress(b)
    with pytest.raises(NotResumableError, match="SOURCE_SUBMISSION_PENDING"):
        resume(b, to_progress(plan, progress.receipt.replace(status=Status.SOURCE_CONFIRMING)))
    with pytest.raises(NotResumableError, match="serialized transaction"):
        resume(b, to_progress(plan, progress.receipt.replace(protocol_state={"routeId": plan.route_id})))
    with pytest.raises(CheckpointInvalidError, match="id does not match"):
        resume(b, to_progress(plan, progress.receipt.replace(id="at1else")))
    with pytest.raises(CheckpointInvalidError, match="does not match the prepared route"):
        resume(b, to_progress(plan, progress.receipt.replace(
            protocol_state={**progress.receipt.protocol_state, "routeId": "other"})))
    assert b.aleo.submitted == []


def test_resume_refuses_a_solana_source_leg_and_points_at_recover():
    """``SolCall`` has no approval step and no resumable pre-broadcast state: there is nothing to
    continue, so resume never guesses — it sends the caller to recover()/wait()."""
    b = FakeBridge(solana=True)
    plan = prepare(b.registry, source_chain="solana", source_asset="sol", destination_chain="aleo", destination_asset="sol", amount="0.000000001",
                   recipient=ALEO_RECIPIENT, sender=SOL_ADDRESS)
    sig = "5igNature" * 8
    receipt = Receipt(id=sig, protocol="hyperlane", status=Status.SOURCE_SUBMISSION_PENDING,
                      protocol_state={"routeId": plan.route_id})
    with pytest.raises(NotResumableError, match="recover"):
        resume(b, to_progress(plan, receipt))
    assert b.calls == []


# ── resume: EVM source ────────────────────────────────────────────────────────

def test_resume_evm_xreserve_rescans_then_deposits_once():
    b = FakeBridge()
    plan, receipt = _xreserve_progress(b, mint_mode="private")
    b.eth.recover_result = receipt                 # history has no deposit → still submission pending
    cps = []
    out = resume(b, to_progress(plan, receipt), secret_nonce="7scalar", on_checkpoint=cps.append)
    assert [c[0] for c in b.calls] == ["eth.recover_source", "eth.quote_deposit_usdc", "eth.deposit_usdc"]
    assert b.calls[0][2] is True                   # required=True: refuse to guess without an approval block
    assert b.calls[1][1] == {"plan": plan, "secret_nonce": "7scalar"}
    assert b.calls[2][1] == {"plan": plan, "secret_nonce": "7scalar"}
    assert out.receipt.status is Status.ATTESTATION_PENDING
    assert out.receipt.protocol_state["approvalTxIds"] == [APPROVAL]     # prior approval carried forward
    # the secret nonce reaches the module and nothing else
    assert "7scalar" not in json.dumps(out.receipt.protocol_state)
    assert all("7scalar" not in json.dumps(c.to_dict()) for c in cps)


def test_resume_evm_never_repeats_a_deposit_history_already_contains():
    b = FakeBridge()
    plan, receipt = _xreserve_progress(b, mint_mode="private")
    b.eth.recover_result = receipt.replace(status=Status.ATTESTATION_PENDING, id="0x" + "cc" * 32)
    out = resume(b, to_progress(plan, receipt), secret_nonce="7scalar")
    assert [c[0] for c in b.calls] == ["eth.recover_source"]
    assert out.receipt.status is Status.ATTESTATION_PENDING and out.next == "wait"


def test_resume_refuses_when_the_re_quoted_hook_data_does_not_match_the_checkpoint():
    """veil guard 1: the hook commits ``(recipient, secret_nonce)``. A deposit built with a
    different nonce than the approval was quoted against mints to a commitment nobody can open."""
    b = FakeBridge()
    plan, receipt = _xreserve_progress(b, mint_mode="private")
    b.eth.recover_result = receipt
    b.eth.hook_data = bytes([2]) + b"\x99" * 64
    with pytest.raises(NotResumableError, match="secret nonce"):
        resume(b, to_progress(plan, receipt), secret_nonce="7scalar")
    assert "eth.deposit_usdc" not in [c[0] for c in b.calls]


@pytest.mark.parametrize("hook", [None, "not-hex", "0x", "0x" + "11" * 64, "0x" + "11" * 66, 65])
def test_resume_refuses_an_xreserve_checkpoint_without_usable_hook_data(hook):
    """Fix round 1 (R1): with no checkpointed hook there is nothing to compare the re-quote
    against, so the equality guard below would pass vacuously and the deposit could be re-hooked to
    a different commitment. Missing, malformed or wrong-width hook data is refused before any RPC —
    including the history scan, so nothing is read on a transfer resume() will not finish."""
    b = FakeBridge()
    plan, receipt = _xreserve_progress(b, mint_mode="private")
    state = {k: v for k, v in receipt.protocol_state.items() if k != "hookData"}
    if hook is not None:
        state["hookData"] = hook
    b.eth.recover_result = receipt
    with pytest.raises(NotResumableError, match="hook data"):
        resume(b, to_progress(plan, receipt.replace(protocol_state=state)), secret_nonce="7scalar")
    assert b.calls == []


def test_resume_refuses_when_the_recovered_allowance_is_gone():
    """veil guard 2: the approval this checkpoint recorded no longer covers the deposit — something
    else spent it. Re-approving here would be a second irreversible step resume never owns."""
    b = FakeBridge()
    plan, receipt = _xreserve_progress(b, mint_mode="private")
    b.eth.recover_result = receipt
    b.eth.approval_required = True
    with pytest.raises(NotResumableError, match="allowance"):
        resume(b, to_progress(plan, receipt), secret_nonce="7scalar")
    assert "eth.deposit_usdc" not in [c[0] for c in b.calls]


def test_resume_of_a_private_mint_without_its_secret_nonce_is_refused_before_any_rpc():
    b = FakeBridge()
    plan, receipt = _xreserve_progress(b, mint_mode="private")
    b.eth.recover_result = receipt
    with pytest.raises(ConfigurationError, match="secret_nonce"):
        resume(b, to_progress(plan, receipt))
    assert b.calls == []
    # a public mint has nothing to commit to: the default is fine
    b2 = FakeBridge()
    plan2, receipt2 = _xreserve_progress(b2)
    b2.eth.recover_result = receipt2
    assert resume(b2, to_progress(plan2, receipt2)).receipt.status is Status.ATTESTATION_PENDING
    assert b2.calls[1][1]["secret_nonce"] == "0scalar"


def test_resume_evm_hyperlane_redispatches_without_re_approving():
    b = FakeBridge()
    plan = prepare(b.registry, source_chain="ethereum", source_asset="wbtc", destination_chain="aleo", destination_asset="wbtc", amount="0.001",
                   recipient=ALEO_RECIPIENT)
    receipt = Receipt(id=APPROVAL, protocol="hyperlane", status=Status.SOURCE_SUBMISSION_PENDING,
                      protocol_state={"routeId": plan.route_id, "approvalTxIds": [APPROVAL],
                                      "sourceSender": EVM_ADDRESS})
    b.eth.recover_result = receipt
    out = resume(b, to_progress(plan, receipt))
    assert [c[0] for c in b.calls] == ["eth.recover_source", "eth.quote_transfer_remote", "eth.transfer_remote"]
    assert out.receipt.status is Status.SOURCE_CONFIRMING
    assert out.receipt.protocol_state["approvalTxIds"] == [APPROVAL]
    b2 = FakeBridge()
    b2.eth.recover_result = receipt
    b2.eth.approval_required = True
    with pytest.raises(NotResumableError, match="allowance"):
        resume(b2, to_progress(plan, receipt))


def test_resume_after_a_dropped_dispatch_leaves_only_the_new_record(tmp_path):
    """A dropped transfer's checkpoint is deliberately KEPT (nothing moved, recover() re-scans from
    it), so the re-dispatch has to supersede it — otherwise the store lists the failed ghost of a
    transfer that has just been sent again."""
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(checkpoints=store)
    plan = prepare(b.registry, source_chain="ethereum", source_asset="wbtc", destination_chain="aleo", destination_asset="wbtc", amount="0.001",
                   recipient=ALEO_RECIPIENT)
    dropped = Receipt(id=DROPPED_DISPATCH, protocol="hyperlane", status=Status.SOURCE_SUBMISSION_PENDING,
                      source_tx_id=DROPPED_DISPATCH,
                      protocol_state={"routeId": plan.route_id, "approvalTxIds": [APPROVAL],
                                      "sourceSender": EVM_ADDRESS, "sourceNonce": "83", "dropped": True,
                                      "sourceError": "transaction was dropped or replaced before it mined"})
    from aleo_bridge.checkpoint import create_checkpoint
    store.save(create_checkpoint(plan, dropped, b.registry))
    assert [c.id for c in store.list()] == [DROPPED_DISPATCH]
    b.eth.recover_result = dropped
    out = resume(b, to_progress(plan, dropped))
    assert out.receipt.status is Status.SOURCE_CONFIRMING and out.receipt.source_tx_id != DROPPED_DISPATCH
    assert [c.id for c in store.list()] == [out.receipt.source_tx_id]     # the stale record is gone


def test_resume_refuses_a_plan_prepared_for_another_account():
    b = FakeBridge()
    plan, receipt = _xreserve_progress(b)
    stale = prepare(b.registry, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx", amount="2",
                    recipient=ALEO_RECIPIENT, sender="0x0000000000000000000000000000000000000009")
    with pytest.raises(ConfigurationError, match="sender"):
        resume(b, to_progress(stale, receipt.replace(protocol_state={**receipt.protocol_state,
                                                                     "routeId": stale.route_id})))
    assert b.calls == []


# ── complete ──────────────────────────────────────────────────────────────────

def _ready(b):
    plan, payload, message_hash, receipt = _inbound_private(b)
    ready = receipt.replace(status=Status.DESTINATION_ACTION_REQUIRED,
                            next_action={"kind": "xreserve-private-mint", "chainId": "aleo-testnet"},
                            protocol_state={**receipt.protocol_state, "attestation": SIG})
    return plan, payload, message_hash, ready


def test_complete_proves_checkpoints_then_submits_one_private_mint():
    b = FakeBridge(environment="testnet")
    plan, payload, message_hash, ready = _ready(b)
    b.xreserve.expected_secret_nonce = "7scalar"
    cps = []
    out = complete(b, to_progress(plan, ready), secret_nonce="7scalar", on_checkpoint=cps.append)
    assert b.calls[-1][0] == "xreserve.private_mint" and b.calls[-1][1]["secret_nonce"] == "7scalar"
    assert [e[0] for e in b.events] == ["delegate_prepared", "checkpoint:DESTINATION_ACTION_REQUIRED",
                                        "submit", "checkpoint:DESTINATION_CONFIRMING"]
    serialized = json.dumps({"type": "execute", "id": "at1fake1", "fee": {}})
    assert cps[0].destination == {"preparedTransaction": {"transactionId": "at1fake1",
                                                          "serializedTransaction": serialized}}
    assert cps[0].source == {"transactionId": "0x" + "22" * 32}
    assert cps[1].destination == {"transactionId": "at1fake1"}
    assert out.next == "wait" and out.receipt.status is Status.DESTINATION_CONFIRMING
    assert out.receipt.destination_tx_id == "at1fake1" and out.receipt.next_action is None
    assert out.receipt.protocol_state["payload"] == "0x" + payload.hex()
    assert out.receipt.protocol_state["destinationFunction"] == "private_mint"
    assert "preparedDestinationTransaction" not in out.receipt.protocol_state
    # the nonce, the attestation and the hook never reach the recovery record
    for cp in cps:
        text = json.dumps(cp.to_dict())
        assert "7scalar" not in text and SIG not in text and "attestation" not in text
    assert "7scalar" not in json.dumps(out.receipt.protocol_state)
    assert "secretNonce" not in json.dumps(out.receipt.protocol_state)


def test_complete_never_writes_a_secret_to_the_bound_checkpoint_store(tmp_path):
    """Fix round 1 (R2): the same claim as above, but proved against what actually reaches disk —
    every byte the store wrote, not just the Checkpoint objects handed to the callback. The secret
    nonce, Circle's attestation and the hook the deposit committed to must appear in none of it."""
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(environment="testnet", checkpoints=store)
    plan, payload, message_hash, ready = _ready(b)
    b.xreserve.expected_secret_nonce = "7scalar"
    written = []
    real_save = store.save
    store.save = lambda cp: (written.append(cp.to_json()), real_save(cp))[1]

    complete(b, to_progress(plan, ready), secret_nonce="7scalar")

    hook_hex = payload[-65:].hex()                        # the commitment the deposit was hooked to
    secrets = ["7scalar", "secretNonce", SIG, SIG[2:], hook_hex, payload.hex(), "attestation"]
    on_disk = [p.read_text(encoding="utf-8") for p in tmp_path.glob("*.json")]
    assert written and on_disk                            # the store really was exercised
    for text in written + on_disk:
        for secret in secrets:
            assert secret not in text, f"{secret!r} leaked into a checkpoint"
    # ...and what IS kept is enough to recover the mint
    assert [c.id for c in store.list()] == [message_hash]
    assert store.list()[0].destination == {"transactionId": "at1fake1"}


def test_complete_never_reaches_proving_with_a_nonce_that_opens_no_commitment():
    b = FakeBridge(environment="testnet")
    plan, payload, message_hash, ready = _ready(b)
    b.xreserve.expected_secret_nonce = "7scalar"
    with pytest.raises(AttestationError):
        complete(b, to_progress(plan, ready), secret_nonce="8scalar")
    assert b.events == [] and b.submitted == []


def test_complete_of_a_private_mint_without_its_secret_nonce_is_refused_before_any_rpc():
    b = FakeBridge(environment="testnet")
    plan, payload, message_hash, ready = _ready(b)
    with pytest.raises(ConfigurationError, match="secret_nonce"):
        complete(b, to_progress(plan, ready))
    assert b.calls == [] and b.events == []


def test_complete_rebroadcasts_a_prepared_destination_without_reproving():
    b = FakeBridge(environment="testnet")
    plan, payload, message_hash, ready = _ready(b)
    serialized = json.dumps({"type": "execute", "id": "at1private", "fee": {}})
    ready = ready.replace(id="at1private",
                          protocol_state={**ready.protocol_state, "preparedDestinationTransaction": serialized})
    cps = []
    out = complete(b, to_progress(plan, ready), on_checkpoint=cps.append)      # no secret nonce needed
    assert b.aleo.submitted == [serialized]
    assert not any(e[0] == "delegate_prepared" for e in b.events)
    assert out.receipt.status is Status.DESTINATION_CONFIRMING and out.receipt.destination_tx_id == "at1private"
    assert "preparedDestinationTransaction" not in out.receipt.protocol_state
    assert len(cps) == 1 and cps[0].destination == {"transactionId": "at1private"}
    # the node already knows it: the earlier broadcast won
    b.aleo.duplicate_on_submit = True
    assert complete(b, to_progress(plan, ready)).receipt.status is Status.DESTINATION_CONFIRMING
    # ...but a mismatched id never becomes this transfer's destination transaction
    b2 = FakeBridge(environment="testnet")
    b2.aleo.network.submit_transaction = lambda tx: "at1other"
    with pytest.raises(CheckpointInvalidError, match="expected at1private"):
        complete(b2, to_progress(plan, ready))


def test_complete_guards():
    b = FakeBridge(environment="testnet")
    plan, payload, message_hash, ready = _ready(b)
    with pytest.raises(NotResumableError, match="complete"):
        complete(b, to_progress(plan, ready.replace(status=Status.ATTESTATION_PENDING, next_action=None)))
    with pytest.raises(NotResumableError, match="destination action"):
        complete(b, to_progress(plan, ready.replace(next_action={"kind": "other", "chainId": "aleo-testnet"})))
    with pytest.raises(AttestationError, match="attestation"):
        complete(b, to_progress(plan, ready.replace(protocol_state={**ready.protocol_state, "attestation": "zz"})))
    with pytest.raises(AttestationError):
        complete(b, to_progress(plan, ready.replace(
            protocol_state={k: v for k, v in ready.protocol_state.items() if k != "payload"})))
    with pytest.raises(CheckpointInvalidError, match="does not match the prepared route"):
        complete(b, to_progress(plan, ready.replace(
            protocol_state={**ready.protocol_state, "routeId": "other"})))
    assert b.aleo.submitted == [] and b.submitted == [] and b.events == []
