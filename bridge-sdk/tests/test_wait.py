"""``lifecycle.wait`` — poll ``get_status`` to a caller boundary.

Always stops at ``types.CALLER_BOUNDARIES`` plus any statuses in ``until``, tolerates transient
read errors (network/RPC hiccups) up to ``max_consecutive_errors`` before giving up, and treats a
timeout as "still in flight", never as failure.
"""
import pytest

from aleo_bridge.checkpoint import FileCheckpointStore
from aleo_bridge.errors import BridgeError, CheckpointInvalidError, ConfigurationError, PollingTimeoutError
from aleo_bridge.lifecycle import prepare, wait
from aleo_bridge.types import Attestation, Progress, Receipt, Status, to_progress
from tests.fakes.fake_bridge import ALEO_RECIPIENT, SOL_ADDRESS, FakeBridge
from tests.test_get_status import SIG, _inbound_private


def _sol_progress(b):
    plan = prepare(b.registry, source="aleo/sol", destination="solana/sol", amount="0.000000001", recipient=SOL_ADDRESS)
    receipt = Receipt(id="at1source", protocol="hyperlane", status=Status.DELIVERY_PENDING, source_tx_id="at1source",
                      protocol_state={"routeId": plan.route_id, "destinationBalanceBeforeAtomic": "100",
                                      "expectedDestinationIncreaseAtomic": "1"})
    return plan, to_progress(plan, receipt)


def test_returns_immediately_when_next_is_not_wait_or_status_is_a_boundary():
    b = FakeBridge(solana=True)
    plan, progress = _sol_progress(b)
    resume = to_progress(plan, progress.receipt.replace(status=Status.SOURCE_SUBMISSION_PENDING))
    assert wait(b, resume) is not None and wait(b, resume).next == "resume" and b.calls == []
    done = to_progress(plan, progress.receipt.replace(status=Status.COMPLETED))
    assert wait(b, done).next == "done" and b.calls == []


def test_polls_until_completion_and_reports_only_changes(monkeypatch):
    b = FakeBridge(solana=True)
    plan, progress = _sol_progress(b)
    reads = iter([100, 100, 101])
    b.sol.balance = lambda: next(reads)
    slept = []
    monkeypatch.setattr("aleo_bridge.lifecycle.time.sleep", slept.append)
    updates = []
    out = wait(b, progress, poll_seconds=0, timeout_seconds=10, on_update=updates.append)
    assert out.next == "done" and out.receipt.status is Status.COMPLETED
    assert updates == [out]                       # two unchanged reads produced no update
    assert slept == [0.0, 0.0]


def test_until_adds_stops_and_empty_until_is_an_error(monkeypatch):
    b = FakeBridge(environment="testnet")
    plan, payload, message_hash, receipt = _inbound_private(b)
    monkeypatch.setattr("aleo_bridge.lifecycle.time.sleep", lambda s: None)
    with pytest.raises(ConfigurationError, match="until"):
        wait(b, to_progress(plan, receipt), until=[])
    calls = {"n": 0}
    real_get = b.xreserve.get_attestation

    def flaky(message_hash, *, route=None):
        calls["n"] += 1
        return None if calls["n"] == 1 else Attestation(payload, bytes.fromhex(message_hash[2:]), bytes.fromhex(SIG[2:]), "complete")
    b.xreserve.get_attestation = flaky
    updates = []
    out = wait(b, to_progress(plan, receipt), until=[Status.DESTINATION_ACTION_REQUIRED], poll_seconds=0,
               timeout_seconds=10, on_update=updates.append)
    assert calls["n"] == 2 and out.next == "complete" and updates == [out]
    # string statuses are accepted in until
    calls["n"] = 0
    assert wait(b, to_progress(plan, receipt), until=["DESTINATION_ACTION_REQUIRED"], poll_seconds=0,
                timeout_seconds=10).next == "complete"


def test_until_rejects_an_unknown_status_name():
    b = FakeBridge(solana=True)
    plan, progress = _sol_progress(b)
    with pytest.raises(ConfigurationError):
        wait(b, progress, until=["NOT_A_REAL_STATUS"])


def test_timeout_carries_status_and_progress(monkeypatch):
    b = FakeBridge(solana=True)
    plan, progress = _sol_progress(b)
    b.sol.balance_lamports = 100
    clock = iter([0.0, 0.0, 5.0, 11.0])
    monkeypatch.setattr("aleo_bridge.lifecycle.time.monotonic", lambda: next(clock))
    monkeypatch.setattr("aleo_bridge.lifecycle.time.sleep", lambda s: None)
    with pytest.raises(PollingTimeoutError, match="DELIVERY_PENDING") as exc:
        wait(b, progress, poll_seconds=0.05, timeout_seconds=10)
    assert exc.value.status is Status.DELIVERY_PENDING
    assert isinstance(exc.value.progress, Progress) and exc.value.progress.next == "wait"


def test_poll_floor_and_negative_controls(monkeypatch):
    b = FakeBridge(solana=True)
    plan, progress = _sol_progress(b)
    reads = iter([100, 101])
    b.sol.balance = lambda: next(reads)
    slept = []
    monkeypatch.setattr("aleo_bridge.lifecycle.time.sleep", slept.append)
    wait(b, progress, poll_seconds=0.01, timeout_seconds=10)
    assert slept == [0.1]                          # floor 0.1 unless exactly 0
    with pytest.raises(ConfigurationError):
        wait(b, progress, poll_seconds=-1)
    with pytest.raises(ConfigurationError):
        wait(b, progress, timeout_seconds=-1)


def test_bound_store_tracks_changes_and_deletes_terminal(tmp_path, monkeypatch):
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(solana=True, checkpoints=store)
    plan, progress = _sol_progress(b)
    from aleo_bridge.checkpoint import create_checkpoint
    store.save(create_checkpoint(plan, progress.receipt, b.registry))
    b.sol.balance_lamports = 101
    monkeypatch.setattr("aleo_bridge.lifecycle.time.sleep", lambda s: None)
    assert wait(b, progress, poll_seconds=0, timeout_seconds=10).next == "done"
    assert store.list() == []


def test_an_aleo_to_evm_xreserve_burn_terminates_and_frees_its_checkpoint(tmp_path, monkeypatch):
    """I2 end to end: before the branch-8 balance check this poll ran until the timeout and left the
    checkpoint behind forever. With the baseline execute records, ``wait`` reaches ``done`` and the
    bound store is cleaned up like every other terminal transfer."""
    from aleo_bridge.checkpoint import create_checkpoint
    from tests.fakes.fake_bridge import EVM_ADDRESS
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(checkpoints=store)
    plan = prepare(b.registry, source="aleo/usdcx", destination="ethereum/usdc", amount="2",
                   recipient=EVM_ADDRESS)
    receipt = Receipt(id="at1burn", protocol="xreserve", status=Status.DELIVERY_PENDING, source_tx_id="at1burn",
                      protocol_state={"routeId": plan.route_id, "destinationBalanceBeforeAtomic": "100",
                                      "expectedDestinationIncreaseAtomic": "2000000"})
    progress = to_progress(plan, receipt)
    store.save(create_checkpoint(plan, receipt, b.registry))
    reads = iter([100, 100, 2_000_100])
    b.eth.balance = lambda asset: next(reads)
    monkeypatch.setattr("aleo_bridge.lifecycle.time.sleep", lambda s: None)

    out = wait(b, progress, poll_seconds=0, timeout_seconds=10)
    assert out.next == "done" and out.receipt.status is Status.COMPLETED
    assert store.list() == []


# ── transient-error tolerance (controller notes item 1) ───────────────────────

def test_transient_errors_are_retried_with_the_normal_poll_interval(monkeypatch):
    b = FakeBridge(solana=True)
    plan, progress = _sol_progress(b)
    slept = []
    monkeypatch.setattr("aleo_bridge.lifecycle.time.sleep", slept.append)
    calls = {"n": 0}

    def flaky(bridge, plan_, receipt):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise BridgeError("Solana RPC request failed with HTTP status 429")
        return receipt.replace(status=Status.COMPLETED)

    monkeypatch.setattr("aleo_bridge.lifecycle.get_status", flaky)
    errors = []
    out = wait(b, progress, poll_seconds=0, timeout_seconds=10, on_error=errors.append)
    assert calls["n"] == 4
    assert out.next == "done" and out.receipt.status is Status.COMPLETED
    assert len(errors) == 3 and all(isinstance(e, BridgeError) for e in errors)
    assert slept == [0.0, 0.0, 0.0]


def test_six_consecutive_transient_errors_reraise_the_sixth(monkeypatch):
    b = FakeBridge(solana=True)
    plan, progress = _sol_progress(b)
    monkeypatch.setattr("aleo_bridge.lifecycle.time.sleep", lambda s: None)
    calls = {"n": 0}

    def always_flaky(bridge, plan_, receipt):
        calls["n"] += 1
        raise BridgeError("request failed: connection reset")

    monkeypatch.setattr("aleo_bridge.lifecycle.get_status", always_flaky)
    with pytest.raises(BridgeError, match="connection reset"):
        wait(b, progress, poll_seconds=0, timeout_seconds=10)
    assert calls["n"] == 6                          # five tolerated, the sixth re-raises


def test_a_flaky_destination_balance_reader_is_retried_not_swallowed(monkeypatch):
    """Task 6 review item 8, end to end: the branch-6 balance read raises a 429-shaped transport
    error through the REAL ``get_status``, and ``wait`` retries it until the delivery is visible."""
    b = FakeBridge(solana=True)
    plan, progress = _sol_progress(b)
    monkeypatch.setattr("aleo_bridge.lifecycle.time.sleep", lambda s: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise BridgeError("Solana RPC request failed with HTTP status 429")
        return 101

    b.sol.balance = flaky
    errors = []
    out = wait(b, progress, poll_seconds=0, timeout_seconds=10, on_error=errors.append)
    assert calls["n"] == 3 and len(errors) == 2
    assert out.next == "done" and out.receipt.status is Status.COMPLETED


def test_a_non_transient_error_propagates_on_the_first_attempt(monkeypatch):
    b = FakeBridge(solana=True)
    plan, progress = _sol_progress(b)
    calls = {"n": 0}

    def raises_checkpoint_invalid(bridge, plan_, receipt):
        calls["n"] += 1
        raise CheckpointInvalidError("stale checkpoint")

    monkeypatch.setattr("aleo_bridge.lifecycle.get_status", raises_checkpoint_invalid)
    with pytest.raises(CheckpointInvalidError, match="stale checkpoint"):
        wait(b, progress, poll_seconds=0, timeout_seconds=10)
    assert calls["n"] == 1
