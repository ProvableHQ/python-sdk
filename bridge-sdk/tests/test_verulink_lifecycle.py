"""Funds-safety tests for public Verulink execution, claims, and recovery."""
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from eth_abi import encode
from eth_utils import keccak

from aleo_bridge import BridgeError, Receipt, RoutePausedError, Status, create_checkpoint, lifecycle, to_progress, PreparedTx
from aleo_bridge._verulink_wire import address_literal, incoming, packet_hash
from tests.fakes.fake_verulink import Harness, FIXTURE, M


def receipt(h):
    tx_id = FIXTURE["aleoSource"]["transaction"]["id"] if h.outbound else FIXTURE["bscSource"]["tx"]["hash"]
    return Receipt(tx_id, "verulink", Status.SOURCE_CONFIRMING, source_tx_id=tx_id, protocol_state={"routeId": h.plan.route_id})


def ready(h):
    return lifecycle.get_status(h.bridge, h.plan, receipt(h))


def input_(value):
    return {"type": "public", "value": value}


def prepared_claim(h, p, signatures=None):
    sigs = signatures or FIXTURE["toAleo"]["status"]["signatures"]
    signers, signed = [s["signer"] for s in sigs], [s["sign"] for s in sigs]
    while len(signers) < 5:
        signers.append("aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc")
        signed.append(signed[0])
    return {"id": "at1prepared", "execution": {"transitions": [
        {"program": "credits.aleo", "function": "transfer_public", "inputs": [input_(p["recipient"]), input_(p["amount"] + "u64")]},
        {"program": M["aleoTokenService"], "function": "token_receive_public", "outputs": [{"type": "public", "value": "true"}],
         "inputs": list(map(input_, [address_literal(p["sender"]), p["recipient"], p["amount"] + "u64",
            p["sequence"] + "u64", p["height"] + "u64", "[" + ", ".join(signers) + "]", "[" + ", ".join(signed) + "]",
            p["sourceChain"] + "u128", address_literal(p["sourceAddress"])]))}]}}


@pytest.mark.parametrize("outbound", [True, False])
def test_attestation_then_claim_and_no_duplicate_submission(outbound, monkeypatch):
    h = Harness(outbound)
    r = ready(h)
    assert r.status == Status.DESTINATION_ACTION_REQUIRED
    if not outbound:
        tx = prepared_claim(h, r.protocol_state["packet"])
        monkeypatch.setattr(lifecycle, "_prepare_aleo", lambda *args: PreparedTx(tx["id"], json.dumps(tx)))
    progress = h.bridge.complete(to_progress(h.plan, r), on_checkpoint=h.saved.append)
    assert progress.receipt.status == Status.DESTINATION_CONFIRMING
    assert h.saved[-1].destination["transactionId"] == progress.receipt.destination_tx_id
    assert len(h.sent if outbound else h.aleo.submitted) == 1
    if not outbound:
        assert h.saved[0].destination["preparedTransaction"]["transactionId"] == "at1prepared"
    with pytest.raises(BridgeError, match="already submitted"):
        h.bridge.complete(progress)


@pytest.mark.parametrize("outbound", [True, False])
@pytest.mark.parametrize("paused_side", ["source", "destination"])
def test_claim_only_requires_destination_to_be_unpaused(outbound, paused_side, monkeypatch):
    h = Harness(outbound)
    r = ready(h)
    if not outbound:
        tx = prepared_claim(h, r.protocol_state["packet"])
        monkeypatch.setattr(lifecycle, "_prepare_aleo", lambda *args: PreparedTx(tx["id"], json.dumps(tx)))
    if outbound == (paused_side == "source"):
        h.aleo.mappings[M["aleoBridge"]]["bridge_settings"]["3u8"] = "0u8"
    else:
        h.rpc.values["paused"] = True
    if paused_side == "destination":
        with pytest.raises(RoutePausedError):
            h.bridge.complete(to_progress(h.plan, r))
        assert not h.sent and not h.aleo.submitted
    else:
        progress = h.bridge.complete(to_progress(h.plan, r))
        assert progress.receipt.status == Status.DESTINATION_CONFIRMING
        assert len(h.sent if outbound else h.aleo.submitted) == 1


@pytest.mark.parametrize("outbound", [True, False])
def test_consumption_without_delivery_evidence_stays_pending(outbound):
    h = Harness(outbound)
    r = ready(h)
    p = r.protocol_state["packet"]
    if outbound:
        h.rpc.values["consumedPackets"] = packet_hash(p)
    else:
        h.aleo.mappings[M["aleoBridge"]]["in_packet_consumed"] = {
            f"{{ chain_id: {p['sourceChain']}u128, sequence: {p['sequence']}u64 }}": "true"}
    result = lifecycle.get_status(h.bridge, h.plan, r)
    assert result.status == Status.DELIVERY_PENDING
    assert result.next_action is None
    with pytest.raises(BridgeError):
        h.bridge.complete(to_progress(h.plan, r))
    assert not h.sent and not h.aleo.submitted


@pytest.mark.parametrize("consumed", [False, True])
def test_prepared_claim_survives_recovery_and_is_followed(consumed):
    h = Harness(False)
    r = ready(h)
    p = r.protocol_state["packet"]
    tx = prepared_claim(h, p)
    r = r.replace(id=tx["id"], protocol_state={**r.protocol_state, "preparedDestinationTransaction": json.dumps(tx)})
    h.rows = []
    if consumed:
        h.aleo.mappings[M["aleoBridge"]]["in_packet_consumed"] = {
            f"{{ chain_id: {p['sourceChain']}u128, sequence: {p['sequence']}u64 }}": "true"}
    cp = create_checkpoint(h.plan, r, h.bridge.registry)
    progress = h.bridge.recover(cp)
    saved = create_checkpoint(progress.plan, progress.receipt, h.bridge.registry)
    assert saved.destination == cp.destination
    progress = h.bridge.recover(saved)
    if consumed:
        h.aleo.confirmed_transactions[tx["id"]] = {"status": "accepted", "transaction": tx}
        progress = h.bridge.wait(progress, timeout_seconds=0)
        assert progress.next == "done"
        assert progress.receipt.destination_tx_id == tx["id"]
        assert not h.aleo.submitted
    else:
        h.rows = [FIXTURE["toAleo"]]
        progress = h.bridge.wait(progress, timeout_seconds=0)
        progress = h.bridge.complete(progress)
        assert h.aleo.submitted == [json.dumps(tx)]
        assert "preparedDestinationTransaction" not in progress.receipt.protocol_state


@pytest.mark.parametrize("fresh", [False, True])
def test_forged_prepared_claim_signatures_cannot_broadcast(fresh, monkeypatch):
    h = Harness(False)
    r = ready(h)
    signatures = copy.deepcopy(FIXTURE["toAleo"]["status"]["signatures"])
    signatures[1]["sign"] = signatures[0]["sign"]
    tx = prepared_claim(h, r.protocol_state["packet"], signatures)
    if fresh:
        monkeypatch.setattr(lifecycle, "_prepare_aleo", lambda *args: PreparedTx(tx["id"], json.dumps(tx)))
    else:
        r = r.replace(id=tx["id"], protocol_state={**r.protocol_state, "preparedDestinationTransaction": json.dumps(tx)})
    with pytest.raises(BridgeError, match="signature"):
        h.bridge.complete(to_progress(h.plan, r))
    assert not h.aleo.submitted


def test_aleo_delivery_and_holding_are_distinct():
    h = Harness(False)
    r = ready(h)
    p = r.protocol_state["packet"]
    tx = prepared_claim(h, p)
    h.aleo.confirmed_transactions[tx["id"]] = {"status": "accepted", "transaction": tx}
    h.aleo.mappings[M["aleoBridge"]]["in_packet_consumed"] = {
        f"{{ chain_id: {p['sourceChain']}u128, sequence: {p['sequence']}u64 }}": "true"}
    r = r.replace(destination_tx_id=tx["id"])
    assert lifecycle.get_status(h.bridge, h.plan, r).status == Status.COMPLETED
    tx["execution"]["transitions"][-1]["outputs"][0]["value"] = "false"
    result = lifecycle.get_status(h.bridge, h.plan, r)
    assert result.status == Status.FAILED
    assert result.protocol_state["held"] is True


def test_holding_keeps_the_durable_checkpoint(tmp_path):
    from aleo_bridge import FileCheckpointStore
    h = Harness(False)
    h.bridge.checkpoints = FileCheckpointStore(tmp_path)
    r = ready(h)
    p = r.protocol_state["packet"]
    tx = prepared_claim(h, p)
    tx["execution"]["transitions"][-1]["outputs"][0]["value"] = "false"
    h.aleo.confirmed_transactions[tx["id"]] = {"status": "accepted", "transaction": tx}
    h.aleo.mappings[M["aleoBridge"]]["in_packet_consumed"] = {
        f"{'{'} chain_id: {p['sourceChain']}u128, sequence: {p['sequence']}u64 {'}'}": "true"}
    r = r.replace(destination_tx_id=tx["id"], status=Status.DESTINATION_CONFIRMING, next_action=None)
    cp = create_checkpoint(h.plan, r, h.bridge.registry)
    h.bridge.checkpoints.save(cp)
    progress = h.bridge.wait(to_progress(h.plan, r), timeout_seconds=0)
    assert progress.next == "failed" and progress.receipt.protocol_state["held"]
    stored = h.bridge.checkpoints.load(cp.id)
    assert stored is not None
    assert stored.source["transactionId"] == r.source_tx_id
    assert stored.destination["transactionId"] == tx["id"]


def test_bsc_finality_and_exact_recipient_mint():
    h = Harness()
    r = ready(h)
    p = r.protocol_state["packet"]
    tx_id = "0x" + "ab" * 32
    h.rpc.values["consumedPackets"] = packet_hash(p)
    consume = {"address": M["evmBridge"], "topics": [keccak(text="Consumed(uint256,uint256,bytes32,uint8)")],
               "data": "0x" + encode(["uint256", "uint256", "bytes32", "uint8"],
                                     [int(p["sourceChain"]), int(p["sequence"]), packet_hash(p), 3]).hex()}
    mint = {"address": M["evmToken"], "topics": [keccak(text="Transfer(address,address,uint256)"),
            bytes(32), bytes.fromhex(p["recipient"][2:]).rjust(32, b"\0")],
            "data": "0x" + encode(["uint256"], [int(p["amount"])]).hex()}
    for i, log in enumerate((consume, mint)):
        log.update(logIndex=hex(i), transactionIndex="0x0", transactionHash=tx_id, blockHash="0x" + "bb" * 32, blockNumber="0x64")
    h.rpc.receipts[tx_id] = {"status": "0x1", "transactionHash": tx_id, "blockNumber": "0x64", "logs": [consume, mint]}
    h.rpc.transactions[tx_id] = {"hash": tx_id, "to": M["evmTokenService"], "from": p["recipient"], "blockNumber": "0x64",
                                  "input": h.rpc.codec.encode_abi("tokenReceive", args=[incoming(p), b""])}
    r = r.replace(destination_tx_id=tx_id)
    h.rpc.height = 100
    assert lifecycle.get_status(h.bridge, h.plan, r).status == Status.DESTINATION_CONFIRMING
    h.rpc.height = 114
    assert lifecycle.get_status(h.bridge, h.plan, r).status == Status.COMPLETED
    h.rpc.receipts[tx_id]["logs"] = [consume]
    with pytest.raises(BridgeError, match="delivery"):
        lifecycle.get_status(h.bridge, h.plan, r)
    h.rpc.receipts[tx_id]["logs"] = [consume, mint]
    mint["topics"][2] = bytes.fromhex(h.rpc.values["holding"][2:]).rjust(32, b"\0")
    result = lifecycle.get_status(h.bridge, h.plan, r)
    assert result.status == Status.FAILED and result.protocol_state["held"]


def test_evm_source_approval_timeout_and_ambiguous_resume():
    h = Harness(False)
    q = h.quote()
    h.rpc.values["allowance"] = 0
    progress = h.bridge.execute(q.plan, on_checkpoint=h.saved.append, timeout_seconds=0)
    assert progress.receipt.status == Status.SOURCE_APPROVAL_PENDING
    assert len(h.sent) == 1
    assert h.rpc.codec.decode_function_input(h.sent[0]["data"])[0].fn_name == "approve"
    r = progress.receipt.replace(status=Status.SOURCE_SUBMISSION_PENDING)
    with pytest.raises(BridgeError, match="cannot safely resubmit"):
        h.bridge.resume(to_progress(progress.plan, r))
    assert len(h.sent) == 1


def test_evm_source_burn_and_changed_fee_rejected():
    h = Harness(False)
    q = h.quote()
    h.rpc.values["platformFees"] = 200
    with pytest.raises(BridgeError, match="fee increased"):
        h.bridge.execute(q.plan)
    assert not h.sent
    h.rpc.values["platformFees"] = 100
    progress = h.bridge.execute(q.plan, on_checkpoint=h.saved.append)
    assert progress.receipt.status == Status.SOURCE_CONFIRMING
    fn, args = h.rpc.codec.decode_function_input(h.sent[0]["data"])
    assert fn.fn_name == "publicTokenSend" and args["amount"] == 11_000_001
    assert h.saved[-1].intent["amountOut"] == "10.989001"
    assert h.saved[-1].source["transactionId"] == progress.receipt.source_tx_id


def test_prepared_source_checkpoint_precedes_broadcast_and_resume_is_identical(monkeypatch):
    h = Harness()
    q = h.quote()
    tx = copy.deepcopy(FIXTURE["aleoSource"]["transaction"])
    tx["id"] = "at1prepared"
    transitions = tx["execution"]["transitions"]
    root = transitions[-1]
    root["inputs"][1]["value"] = "11000001u64"
    root["inputs"][5]["value"] = "11000u64"
    publish = next(t for t in transitions if t["program"] == M["aleoBridge"] and t["function"] == "publish")
    publish["inputs"][6]["value"] = "10989001u128"
    monkeypatch.setattr(lifecycle, "_prepare_aleo", lambda *args: PreparedTx(tx["id"], json.dumps(tx)))
    def checkpoint(cp):
        h.saved.append(cp)
        if cp.source.get("preparedTransaction"):
            assert not h.aleo.submitted
            raise RuntimeError("interrupted before broadcast")
    with pytest.raises(RuntimeError, match="interrupted"):
        h.bridge.execute(q.plan, on_checkpoint=checkpoint)
    assert not h.aleo.submitted
    progress = h.bridge.recover(h.saved[-1])
    assert progress.next == "resume"
    assert progress.plan.amount_out == q.amount_out
    progress = h.bridge.resume(progress)
    assert h.aleo.submitted == [json.dumps(tx)]
    assert progress.receipt.source_tx_id == "at1prepared"


def test_agent_bsc_quote_and_public_claim_need_no_nonce():
    from aleo_bridge.agent import bridge_tools, dispatch_tool
    h = Harness(False)
    result = dispatch_tool(h.bridge, "bridge_quote", {"source_chain": "bsc", "source_asset": "aleo",
        "destination_chain": "aleo", "recipient": h.plan.recipient, "amount": "20", "bridge_protocol": "verulink"})
    assert "error" not in result
    assert result["kind"] == "evm-verulink"
    r = ready(h)
    cp = create_checkpoint(h.plan, r, h.bridge.registry)
    preview = dispatch_tool(h.bridge, "bridge_complete", {"checkpoint": cp.to_dict()})
    assert "error" not in preview
    assert not h.aleo.submitted
    schemas = {t["name"]: t["input_schema"] for t in bridge_tools()}
    assert "verulink" in schemas["bridge_list_routes"]["properties"]["bridge_protocol"]["enum"]


def test_agent_rejects_read_only_bsc_before_execution():
    from aleo_bridge import dispatch_tool
    h = Harness(False)
    h.evm.w3.eth.default_account = None
    result = dispatch_tool(h.bridge, "bridge_execute", {
        "source_chain": "bsc", "source_asset": "aleo", "destination_chain": "aleo",
        "recipient": h.plan.recipient, "amount": "20", "bridge_protocol": "verulink", "confirm": True})
    assert result["error_type"] == "ConfigurationError" and "read-only" in result["error"]
    assert "BSC_RPC_URL" in result["how_to_fix"]
    assert not h.sent and not h.aleo.submitted
