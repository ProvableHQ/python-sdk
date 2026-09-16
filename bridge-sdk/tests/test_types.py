import dataclasses
import json

import pytest

from aleo_bridge.types import (CALLER_BOUNDARIES, TERMINAL, AleoHyperlaneQuote, Attestation, BridgeStatus,
                               ChainStatus, DispatchReceipt, Fee, GasQuote, Plan, PreparedTx, PrivacyReceipt,
                               Progress, Receipt, Status, Step, to_progress)


def _plan() -> Plan:
    return Plan(route_id="hyperlane:aleo/wbtc->ethereum/wbtc", registry_version="2026-08-31.solana-deposits.1",
                protocol="hyperlane", environment="mainnet", source_asset_id="aleo/wbtc",
                destination_asset_id="ethereum/wbtc", amount="0.001", amount_atomic=100_000,
                recipient="0x0000000000000000000000000000000000000001", sender=None, mint_mode="public",
                steps=(Step("source-dispatch", "dispatch", "aleo-wallet", True),
                       Step("message-delivery", "wait-delivery", "protocol", False)))


def test_status_enum_and_sets():
    assert [s.value for s in Status] == [
        "PREPARED", "SOURCE_APPROVAL_PENDING", "SOURCE_SUBMISSION_PENDING", "SOURCE_CONFIRMING", "ATTESTATION_PENDING",
        "DESTINATION_ACTION_REQUIRED", "DELIVERY_PENDING", "DESTINATION_CONFIRMING", "COMPLETED", "FAILED", "EXPIRED"]
    assert Status.COMPLETED == "COMPLETED" and Status("FAILED") is Status.FAILED
    assert TERMINAL == {Status.COMPLETED, Status.FAILED, Status.EXPIRED}
    assert CALLER_BOUNDARIES == {Status.SOURCE_SUBMISSION_PENDING, Status.DESTINATION_ACTION_REQUIRED,
                                 Status.COMPLETED, Status.FAILED, Status.EXPIRED}
    assert json.dumps({"s": Status.COMPLETED}) == '{"s": "COMPLETED"}'


@pytest.mark.parametrize("status,expected", [
    (Status.SOURCE_SUBMISSION_PENDING, "resume"), (Status.DESTINATION_ACTION_REQUIRED, "complete"),
    (Status.COMPLETED, "done"), (Status.FAILED, "failed"), (Status.EXPIRED, "failed"),
    (Status.PREPARED, "wait"), (Status.SOURCE_APPROVAL_PENDING, "wait"), (Status.SOURCE_CONFIRMING, "wait"),
    (Status.ATTESTATION_PENDING, "wait"), (Status.DELIVERY_PENDING, "wait"), (Status.DESTINATION_CONFIRMING, "wait"),
])
def test_to_progress_table(status, expected):
    receipt = Receipt(id="at1x", protocol="hyperlane", status=status, protocol_state={"routeId": "r"})
    plan = _plan()
    progress = to_progress(plan, receipt)
    assert progress.next == expected
    assert progress.plan is plan and progress.receipt is receipt and progress.error is None


def test_to_progress_accepts_status_strings_and_requires_route_id():
    receipt = Receipt(id="at1x", protocol="hyperlane", status="COMPLETED", protocol_state={"routeId": "r"})
    assert to_progress(_plan(), receipt).next == "done"
    with pytest.raises(ValueError, match="routeId"):
        to_progress(_plan(), Receipt(id="at1x", protocol="hyperlane", status=Status.COMPLETED))


def test_plan_round_trip():
    plan = _plan()
    d = plan.to_dict()
    assert d["steps"][0] == {"id": "source-dispatch", "kind": "dispatch", "executor": "aleo-wallet", "irreversible": True}
    assert Plan.from_dict(json.loads(json.dumps(d))) == plan
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.amount = "2"  # type: ignore[misc]


def test_receipt_replace_and_defaults():
    r = Receipt(id="0xabc", protocol="xreserve", status=Status.ATTESTATION_PENDING, protocol_state={"routeId": "x"})
    assert r.source_tx_id is None and r.destination_tx_id is None and r.next_action is None
    r2 = r.replace(status=Status.DESTINATION_ACTION_REQUIRED, next_action={"kind": "xreserve-private-mint", "chainId": "aleo"})
    assert r2.status is Status.DESTINATION_ACTION_REQUIRED and r.status is Status.ATTESTATION_PENDING
    assert r2.protocol_state == {"routeId": "x"} and r2.next_action["kind"] == "xreserve-private-mint"


def test_result_dataclasses():
    receipt = Receipt(id="at1x", protocol="hyperlane", status=Status.SOURCE_CONFIRMING, protocol_state={"routeId": "r"})
    dr = DispatchReceipt(transaction_id="at1x", route_id="r", message_id=None, amount_atomic=1, receipt=receipt)
    assert dr.receipt.protocol_state["routeId"] == "r"
    assert GasQuote("r", 44000, 159337, 1000000000, 402, 8174147).payment_microcredits == 8174147
    att = Attestation(payload=bytes(305), message_hash=bytes(32), attestation=bytes(65), status="complete")
    assert len(att.payload) == 305
    assert PreparedTx("at1x", "{}").serialized == "{}"
    assert PrivacyReceipt("at1s", "aleo/eth", "0.000000000000000001", 1, "shield").direction == "shield"
    q = AleoHyperlaneQuote(kind="aleo-hyperlane", plan=_plan(), fees=(Fee("hook", "aleo", "aleo/aleo", "8.174147", False),),
                           amount_out="0.001", gas_limit=44000, gas_overhead=159337, gas_price=1000000000,
                           exchange_rate=402, payment_microcredits=8174147)
    assert q.fees[0].estimated is False and q.kind == "aleo-hyperlane"
    status = BridgeStatus(environment="mainnet", registry_version="v", chains=[ChainStatus("aleo", None, False, {})], pending=[])
    assert status.chains[0].can_sign is False
    assert isinstance(Progress("wait", _plan(), receipt), Progress)
