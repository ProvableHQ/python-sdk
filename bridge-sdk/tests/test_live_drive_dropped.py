"""Hermetic: the live harness follows a dropped-transaction verdict once (recover → resume)."""
from pathlib import Path

import pytest

from aleo_bridge.registry import DEFAULT_REGISTRY
from aleo_bridge.types import Progress, Receipt, Status
from tests.live import cases as live_cases
from tests.live.helpers import LiveBenchmark, LiveCaseError, LiveState

ROUTE = "hyperlane:ethereum/eth->aleo/eth"
HASH = "0x" + "ab" * 32


def _plan():
    from aleo_bridge.lifecycle import prepare
    return prepare(DEFAULT_REGISTRY, source="ethereum/eth", destination="aleo/eth", amount_atomic=1,
                   recipient="aleo1k5a999p502ty77ur8yryj7qmj2rtxyl8jylpnkcnfuvgyt0vxvyssyn69g",
                   sender="0x734C0a5AB55885974cEDb9D6ff71d8E8448c7375")


def _dropped_progress(plan):
    receipt = Receipt(id=HASH, protocol="hyperlane", status=Status.EXPIRED, source_tx_id=HASH,
                      protocol_state={"routeId": plan.route_id, "sourceNonce": "83", "dropped": True,
                                      "sourceError": f"transaction {HASH} (nonce 83) was dropped"})
    return Progress(next="failed", plan=plan, receipt=receipt, error=receipt.protocol_state["sourceError"])


class _Bridge:
    """Just enough of a Bridge for `_drive`: recover → resumable, resume → done."""
    registry = DEFAULT_REGISTRY

    def __init__(self, plan, *, recover_to="resume"):
        self.plan, self.calls, self.recover_to = plan, [], recover_to

    def wait(self, progress, **_):
        self.calls.append("wait")
        return progress

    def recover(self, checkpoint):
        self.calls.append("recover")
        assert checkpoint.source["transactionId"] == HASH and checkpoint.source["sourceNonce"] == "83"
        if self.recover_to == "resume":
            receipt = Receipt(id=HASH, protocol="hyperlane", status=Status.SOURCE_SUBMISSION_PENDING,
                              protocol_state={"routeId": self.plan.route_id, "sourceNonce": "83"})
            return Progress(next="resume", plan=self.plan, receipt=receipt)
        return _dropped_progress(self.plan)

    def resume(self, progress, **_):
        self.calls.append("resume")
        receipt = Receipt(id="0x" + "cd" * 32, protocol="hyperlane", status=Status.COMPLETED,
                          source_tx_id="0x" + "cd" * 32, protocol_state={"routeId": self.plan.route_id})
        return Progress(next="done", plan=self.plan, receipt=receipt)


def _drive(bridge, progress, tmp_path):
    spec = live_cases.CASES["evm-hyperlane"] if hasattr(live_cases, "CASES") else live_cases.case_for_route(
        DEFAULT_REGISTRY.route(ROUTE), DEFAULT_REGISTRY)
    return live_cases._drive(bridge, progress, LiveState(route_id=ROUTE), tmp_path / "s.json", spec=spec,
                             secret_nonce=None, benchmark=LiveBenchmark("t"), save=lambda _cp: None,
                             wait_timeout_seconds=1, wait_poll_seconds=0, log=lambda _m: None)


def test_a_dropped_verdict_is_followed_once_through_recover_and_resume(tmp_path: Path):
    plan = _plan()
    bridge = _Bridge(plan)
    out = _drive(bridge, _dropped_progress(plan), tmp_path)
    assert out.next == "done"
    assert bridge.calls == ["recover", "resume"]          # never `execute`, never a blind resend


def test_a_dropped_verdict_that_recovers_to_dropped_again_is_final(tmp_path: Path):
    plan = _plan()
    bridge = _Bridge(plan, recover_to="failed")
    with pytest.raises(LiveCaseError, match="dropped"):
        _drive(bridge, _dropped_progress(plan), tmp_path)
    assert bridge.calls == ["recover"]                    # followed exactly once


def test_a_plain_failure_is_not_followed(tmp_path: Path):
    plan = _plan()
    bridge = _Bridge(plan)
    receipt = Receipt(id=HASH, protocol="hyperlane", status=Status.FAILED, source_tx_id=HASH,
                      protocol_state={"routeId": plan.route_id, "sourceError": "reverted"})
    with pytest.raises(LiveCaseError, match="reverted"):
        _drive(bridge, Progress(next="failed", plan=plan, receipt=receipt, error="reverted"), tmp_path)
    assert bridge.calls == []
