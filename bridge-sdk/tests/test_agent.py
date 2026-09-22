"""Agent tools: the Claude-shape surface, the confirm gate, and what must never leave the process.

The brief's own cases (surface/schemas, _serialize, reads, the four write gates) are kept as
written; the controller's rulings add: no "0scalar" fallback for a private mint, no secret in any
tool result, the synthetic xReserve max-fee entry, structured errors for an unconfigured chain,
and `next: "recover"` guidance when a send's outcome is ambiguous.
"""
import json
from pathlib import Path

import pytest

from aleo_bridge.agent import _serialize, _summarize_input, bridge_tools, dispatch_tool
from aleo_bridge.checkpoint import Checkpoint, FileCheckpointStore, create_checkpoint
from aleo_bridge.errors import BridgeError
from aleo_bridge.lifecycle import prepare
from aleo_bridge.types import Fee, Receipt, Status
from tests.fakes.fake_bridge import ALEO_RECIPIENT, EVM_ADDRESS, RECORD_PLAINTEXT, FakeBridge

READS = {"bridge_status", "bridge_list_assets", "bridge_list_routes", "bridge_quote", "bridge_get_progress",
         "bridge_pending"}
WRITES = {"bridge_execute", "bridge_resume", "bridge_complete", "bridge_shield", "bridge_unshield"}
QUOTE_ARGS = {"source": "ethereum/usdc", "destination": "aleo/usdcx", "amount": "2", "recipient": ALEO_RECIPIENT}
NONCE = "7scalar"


def _aleo_out_checkpoint(b):
    """A checkpoint for an Aleo-origin Hyperlane transfer that was proved but never broadcast."""
    plan = prepare(b.registry, source="aleo/eth", destination="ethereum/eth", amount="0.000000000000000001",
                   recipient=EVM_ADDRESS)
    serialized = json.dumps({"type": "execute", "id": "at1prepared", "fee": {}})
    cp = {"version": 1,
          "intent": {"source": {"chain": "aleo", "asset": "eth"},
                     "destination": {"chain": "ethereum", "asset": "eth"},
                     "bridgeProtocol": "hyperlane", "amount": plan.amount, "recipient": EVM_ADDRESS},
          "route": {"id": plan.route_id, "registryVersion": plan.registry_version},
          "source": {"preparedTransaction": {"transactionId": "at1prepared",
                                             "serializedTransaction": serialized}}}
    return cp, serialized


def _inbound_private_checkpoint(b):
    """A testnet xReserve deposit attested and waiting for its private mint."""
    from aleo_bridge.types import Attestation
    from tests.test_get_status import SIG, _inbound_private

    plan, payload, message_hash, receipt = _inbound_private(b)
    b.xreserve.attestations[message_hash] = Attestation(payload, bytes.fromhex(message_hash[2:]),
                                                        bytes.fromhex(SIG[2:]), "complete")
    b.eth.recover_result = receipt
    return {"version": 1,
            "intent": {"source": {"chain": "sepolia", "asset": "usdc"},
                       "destination": {"chain": "aleo-testnet", "asset": "usdcx"},
                       "bridgeProtocol": "xreserve", "amount": "2", "recipient": ALEO_RECIPIENT,
                       "mintMode": "private"},
            "route": {"id": plan.route_id, "registryVersion": plan.registry_version},
            "source": {"transactionId": "0x" + "22" * 32}}


# ── surface ───────────────────────────────────────────────────────────────────

def test_tool_surface_and_schemas():
    tools = bridge_tools()
    assert {t["name"] for t in tools} == READS | WRITES
    assert {t["name"] for t in bridge_tools(include_writes=False)} == READS
    for t in tools:
        assert t["description"] and t["input_schema"]["type"] == "object"
        json.dumps(t)
        if t["name"] in WRITES:
            assert t["input_schema"]["properties"]["confirm"]["type"] == "boolean"
            assert "confirm" in t["description"].lower()
    execute = next(t for t in tools if t["name"] == "bridge_execute")
    assert set(execute["input_schema"]["required"]) == {"source", "destination", "amount", "recipient"}
    assert "plan" not in execute["input_schema"]["properties"]                # agents never carry plans
    quote = next(t for t in tools if t["name"] == "bridge_quote")
    assert quote["input_schema"]["properties"]["mint_mode"]["enum"] == ["public", "record", "private"]
    assert "private key" not in json.dumps(tools).lower()


def test_private_mint_descriptions_state_the_nonce_requirement():
    tools = {t["name"]: t for t in bridge_tools()}
    for name in ("bridge_quote", "bridge_execute", "bridge_complete"):
        assert "secret_nonce" in tools[name]["input_schema"]["properties"]
    for name in ("bridge_execute", "bridge_complete"):
        description = tools[name]["input_schema"]["properties"]["secret_nonce"]["description"].lower()
        assert "required" in description and "0scalar" not in description


# ── _serialize ────────────────────────────────────────────────────────────────

def test_serialize():
    fee = Fee(kind="protocol", chain_id="aleo", asset_id="aleo/aleo", amount="8.174147", estimated=True)
    out = _serialize({"fee": fee, "raw": b"\x01\xff", "status": Status.COMPLETED, "n": 10**20, "t": (1, 2), "none": None})
    assert out == {"fee": {"kind": "protocol", "chain_id": "aleo", "asset_id": "aleo/aleo", "amount": "8.174147",
                           "estimated": True}, "raw": "0x01ff", "status": "COMPLETED", "n": 10**20, "t": [1, 2], "none": None}
    json.dumps(out)


def test_serialize_adds_the_xreserve_max_fee_entry():
    b = FakeBridge()
    quote = dispatch_tool(b, "bridge_quote", QUOTE_ARGS)
    assert quote["fees"] == [{"kind": "protocol", "chain_id": "ethereum", "asset_id": "ethereum/usdc",
                              "amount": "0.1", "estimated": True, "label": "xReserve max fee"}]
    # and the same entry appears when _serialize is handed the quote object on its own
    from aleo_bridge import lifecycle
    raw = lifecycle.quote(b, source="ethereum/usdc", destination="aleo/usdcx", amount="2",
                          recipient=ALEO_RECIPIENT)
    assert _serialize(raw)["fees"][-1]["label"] == "xReserve max fee"


def test_serialize_drops_secret_and_bulky_receipt_state():
    receipt = Receipt(id="0x01", protocol="xreserve", status=Status.ATTESTATION_PENDING,
                      protocol_state={"routeId": "r", "payload": "0x" + "ee" * 305, "attestation": "0x11",
                                      "secretNonce": NONCE, "preparedTransaction": "{...}", "mintMode": "private"})
    out = _serialize(receipt)
    assert out["protocol_state"] == {"routeId": "r", "mintMode": "private"}
    assert NONCE not in json.dumps(out)


# ── reads ─────────────────────────────────────────────────────────────────────

def test_read_tools():
    b = FakeBridge()
    b.public_balances["aleo/usdcx"] = 5_000_000
    status = dispatch_tool(b, "bridge_status", {})
    assert status["environment"] == "mainnet" and status["chains"][0]["balances"] == {"aleo/usdcx": 5_000_000}
    assets = dispatch_tool(b, "bridge_list_assets", {"chain": "aleo"})
    assert {a["symbol"] for a in assets} >= {"USDCx", "ETH", "WBTC", "USDT", "SOL"} and all(a["chain_id"] == "aleo" for a in assets)
    routes = dispatch_tool(b, "bridge_list_routes", {"protocol": "xreserve"})
    assert routes and all(r["protocol"] == "xreserve" and r["availability"] == "active" for r in routes)
    all_routes = dispatch_tool(b, "bridge_list_routes", {"include_unavailable": True})
    assert any(r["availability"] == "metadata-required" for r in all_routes)
    q = dispatch_tool(b, "bridge_quote", {**QUOTE_ARGS, "mint_mode": "private", "secret_nonce": NONCE})
    assert q["kind"] == "evm-xreserve" and q["plan"]["route_id"] == "xreserve:ethereum/usdc->aleo/usdcx"
    assert q["plan"]["amount"] == "2" and q["hook_data"].startswith("0x")
    json.dumps(q)
    assert NONCE not in json.dumps(q)
    assert b.events == []                                                     # nothing signed


# ── writes ────────────────────────────────────────────────────────────────────

def test_execute_requires_confirm_and_requotes_internally():
    b = FakeBridge()
    out = dispatch_tool(b, "bridge_execute", QUOTE_ARGS)
    assert out["confirmation_required"] is True and out["how_to_confirm"] == "re-call with confirm=true"
    assert out["quote"]["kind"] == "evm-xreserve" and out["quote"]["plan"]["amount"] == "2"
    assert [c[0] for c in b.calls] == ["eth.quote_deposit_usdc"] and b.events == []
    b.calls.clear()
    out = dispatch_tool(b, "bridge_execute", {**QUOTE_ARGS, "confirm": True})
    assert [c[0] for c in b.calls] == ["eth.quote_deposit_usdc", "eth.deposit_usdc"]
    assert out["progress"]["next"] == "wait" and out["progress"]["receipt"]["status"] == "ATTESTATION_PENDING"
    assert out["checkpoint"]["version"] == 1 and out["checkpoint"]["route"]["id"] == "xreserve:ethereum/usdc->aleo/usdcx"
    assert "secretNonce" not in json.dumps(out) and "payload" not in json.dumps(out["checkpoint"])
    json.dumps(out)


def test_get_progress_and_pending(tmp_path):
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(ethereum=False, checkpoints=store)
    out = dispatch_tool(b, "bridge_execute", {"source": "aleo/eth", "destination": "ethereum/eth",
                                              "amount": "0.000000000000000001", "recipient": EVM_ADDRESS,
                                              "gas_payment_microcredits": 1, "confirm": True})
    assert out["progress"]["receipt"]["status"] == "SOURCE_CONFIRMING"
    recovered = dispatch_tool(b, "bridge_get_progress", {"checkpoint": out["checkpoint"]})
    assert recovered["progress"]["next"] == "wait"
    assert recovered["progress"]["receipt"]["source_tx_id"] == "at1fake1"
    # a read hands back a checkpoint too: the model never has to keep the one execute returned
    assert recovered["checkpoint"]["receiptId"] == "at1fake1"
    assert recovered["checkpoint"]["route"]["id"] == out["checkpoint"]["route"]["id"]
    assert b.submitted.count(b.submitted[0]) == 1                            # recover never rebroadcasts
    pending = dispatch_tool(b, "bridge_pending", {})
    assert [p["progress"]["receipt"]["id"] for p in pending] == ["at1fake1"]
    assert [p["checkpoint"] for p in pending] == [cp.to_dict() for cp in store.list()]
    json.dumps(pending)


def test_pending_hands_back_the_stored_checkpoint_that_restarts_the_flow(tmp_path):
    """The restart path: a fresh process lists pending transfers and resumes one from the
    checkpoint the listing echoed — never a checkpoint re-derived from a recovered receipt."""
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(ethereum=False, checkpoints=store)
    cp, serialized = _aleo_out_checkpoint(b)
    store.save(Checkpoint.from_dict(cp))

    entries = dispatch_tool(b, "bridge_pending", {})
    assert len(entries) == 1 and entries[0]["progress"]["next"] == "resume"
    assert entries[0]["checkpoint"] == Checkpoint.from_dict(cp).to_dict()     # stored form, verbatim
    out = dispatch_tool(b, "bridge_resume", {"checkpoint": entries[0]["checkpoint"], "confirm": True})
    assert out["progress"]["receipt"]["status"] == "SOURCE_CONFIRMING" and b.aleo.submitted == [serialized]


def test_pending_checkpoint_completes_a_mint_whose_offline_progress_still_says_wait(tmp_path):
    """``bridge_pending`` is offline, so its progress can lag the chain (here: 'wait' while Circle
    has in fact attested). The echoed checkpoint is still exactly what ``bridge_complete`` takes."""
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(environment="testnet", checkpoints=store)
    cp = _inbound_private_checkpoint(b)
    store.save(Checkpoint.from_dict(cp))

    entries = dispatch_tool(b, "bridge_pending", {})
    assert len(entries) == 1 and entries[0]["progress"]["next"] == "wait"     # offline view
    out = dispatch_tool(b, "bridge_complete", {"checkpoint": entries[0]["checkpoint"],
                                               "secret_nonce": NONCE, "confirm": True})
    assert out["progress"]["receipt"]["status"] == "DESTINATION_CONFIRMING"
    assert NONCE not in json.dumps(out)


def test_pending_reports_a_malformed_record_instead_of_collapsing_the_list(tmp_path):
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(ethereum=False, checkpoints=store)
    cp, _ = _aleo_out_checkpoint(b)
    store.save(Checkpoint.from_dict(cp))
    (tmp_path / "at1broken.json").write_text(json.dumps(
        {"version": 1, "receiptId": "at1broken", "intent": {}, "route": {"id": "x", "registryVersion": "y"}}),
        encoding="utf-8")

    entries = dispatch_tool(b, "bridge_pending", {})
    assert len(entries) == 2
    healthy = [e for e in entries if "progress" in e]
    broken = [e for e in entries if "error" in e]
    assert len(healthy) == 1 and healthy[0]["checkpoint"] == Checkpoint.from_dict(cp).to_dict()
    assert len(broken) == 1 and broken[0]["checkpoint_id"] == "at1broken"
    assert broken[0]["error_type"] == "CheckpointInvalidError" and broken[0]["error"]
    assert "progress" not in broken[0]
    json.dumps(entries)


def test_pending_reports_files_the_store_could_not_read_instead_of_hiding_them(tmp_path):
    # I1: a file the STORE itself cannot read back (a future version, a half-written file) is not a
    # checkpoint at all, so it never reaches progress_from_checkpoint — it must still be reported,
    # named by path, next to the healthy transfers rather than silently skipped.
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(ethereum=False, checkpoints=store)
    cp, _ = _aleo_out_checkpoint(b)
    store.save(Checkpoint.from_dict(cp))
    (tmp_path / "future.json").write_text(json.dumps({"version": 2, "intent": {}, "route": {}}), encoding="utf-8")
    (tmp_path / "garbage.json").write_text("{not json at all", encoding="utf-8")

    entries = dispatch_tool(b, "bridge_pending", {})
    healthy = [e for e in entries if "progress" in e]
    unreadable = [e for e in entries if "path" in e]
    assert len(healthy) == 1 and healthy[0]["checkpoint"] == Checkpoint.from_dict(cp).to_dict()
    assert sorted(Path(e["path"]).name for e in unreadable) == ["future.json", "garbage.json"]
    assert all(e["error"] and e["error_type"] and "progress" not in e for e in unreadable)
    json.dumps(entries)


def test_checkpoint_property_names_exactly_the_tools_that_return_one():
    tools = {t["name"]: t for t in bridge_tools()}
    description = tools["bridge_get_progress"]["input_schema"]["properties"]["checkpoint"]["description"]
    returns_one = {"bridge_execute", "bridge_resume", "bridge_complete", "bridge_get_progress", "bridge_pending"}
    assert {name for name in tools if name in description} == returns_one


def test_resume_and_complete_gates():
    b = FakeBridge(ethereum=False)
    cp, serialized = _aleo_out_checkpoint(b)
    out = dispatch_tool(b, "bridge_resume", {"checkpoint": cp})
    assert out["confirmation_required"] is True and out["progress"]["next"] == "resume" and b.aleo.submitted == []
    out = dispatch_tool(b, "bridge_resume", {"checkpoint": cp, "confirm": True})
    # a resumed Aleo leg rebroadcasts the checkpointed bytes through the network, never re-proving
    assert out["progress"]["receipt"]["status"] == "SOURCE_CONFIRMING" and b.aleo.submitted == [serialized]

    b2 = FakeBridge(environment="testnet")
    cp2 = _inbound_private_checkpoint(b2)
    out = dispatch_tool(b2, "bridge_complete", {"checkpoint": cp2, "secret_nonce": NONCE})
    assert out["confirmation_required"] is True and out["progress"]["next"] == "complete" and b2.events == []
    out = dispatch_tool(b2, "bridge_complete", {"checkpoint": cp2, "secret_nonce": NONCE, "confirm": True})
    assert out["progress"]["receipt"]["status"] == "DESTINATION_CONFIRMING"
    assert NONCE not in json.dumps(out)


def test_shield_unshield_gates():
    b = FakeBridge()
    out = dispatch_tool(b, "bridge_shield", {"asset": "aleo/eth", "amount": "0.000000000000000001"})
    assert out["confirmation_required"] is True and out["call"]["function"] == "shield" and b.events == []
    out = dispatch_tool(b, "bridge_shield", {"asset": "aleo/eth", "amount": "0.000000000000000001", "confirm": True})
    assert out["direction"] == "shield" and b.events[-1][0] == "delegate"
    out = dispatch_tool(b, "bridge_unshield", {"asset": "aleo/eth", "amount_atomic": 1, "confirm": True})
    assert out["direction"] == "unshield" and out["amount_atomic"] == 1


def test_privacy_previews_never_echo_a_record(tmp_path):
    """An unshield call's inputs carry the selected record's PLAINTEXT — the private balance
    itself. The preview an agent shows a user must summarize it, never echo it."""
    b = FakeBridge()
    out = dispatch_tool(b, "bridge_unshield", {"asset": "aleo/eth", "amount_atomic": 1})
    blob = json.dumps(out)
    assert out["confirmation_required"] is True
    assert out["call"] == {"program": "arc20_eth.aleo", "function": "unshield",
                           "inputs": ["<record>", "1u128"]}                  # amount literal kept
    assert RECORD_PLAINTEXT not in blob and "_nonce" not in blob and "owner:" not in blob
    # shielding has no record to leak: its literals pass through untouched
    shield = dispatch_tool(b, "bridge_shield", {"asset": "aleo/eth", "amount_atomic": 1})
    assert shield["call"]["inputs"] == ["1u128"]


@pytest.mark.parametrize("value", [
    RECORD_PLAINTEXT,
    "{ owner: aleo1abc.private, microcredits: 1500000u64.private }",
    "record1qyqsqpe2szk2wwwq56akkwx586hkndl3r8vzdwve32lm7elvphh37rsyqyxx66trwfhkxun9v35hguerqqpqzq"
    "8tc0y3cc45vqs0vzcqmqxqwqsy3y6qw6w4vdkvq3qsqrqwqsyq",
    "at1abcdefghijklmnop.record",
])
def test_summarizer_redacts_record_shaped_inputs(value):
    assert _summarize_input(value) == "<record>"


@pytest.mark.parametrize("value", ["1u128", ALEO_RECIPIENT, "arc20_eth.aleo", "transfer_private_to_public",
                                   "{ siblings: [ 0field ], leaf_index: 0u32 }", "true", "0u64"])
def test_summarizer_keeps_public_literals(value):
    assert _summarize_input(value) == value


def test_bridge_tools_returns_deep_copies():
    first = bridge_tools()
    first[0]["description"] = "mutated"
    first[0]["input_schema"]["properties"]["injected"] = {"type": "string"}
    quote = next(t for t in first if t["name"] == "bridge_quote")
    quote["input_schema"]["required"].append("injected")

    second = bridge_tools()
    assert second[0]["description"] != "mutated"
    assert "injected" not in second[0]["input_schema"]["properties"]
    fresh_quote = next(t for t in second if t["name"] == "bridge_quote")
    assert fresh_quote["input_schema"]["required"] == ["source", "destination", "amount", "recipient"]
    assert fresh_quote["input_schema"]["properties"] is not quote["input_schema"]["properties"]


def test_unknown_tool():
    with pytest.raises(ValueError, match="Unknown bridge tool"):
        dispatch_tool(FakeBridge(), "nope", {})


# ── controller rulings ────────────────────────────────────────────────────────

def test_private_mint_never_defaults_the_secret_nonce():
    b = FakeBridge()
    private = {**QUOTE_ARGS, "mint_mode": "private"}
    for tool in ("bridge_quote", "bridge_execute"):
        out = dispatch_tool(b, tool, dict(private))
        assert "secret_nonce" in out["error"] and out["how_to_fix"]
        assert "confirmation_required" not in out and "quote" not in out
    out = dispatch_tool(b, "bridge_execute", {**private, "confirm": True})
    assert "secret_nonce" in out["error"]
    assert b.calls == [] and b.events == []                                   # nothing quoted, nothing moved
    # a public mint still defaults quietly
    assert dispatch_tool(b, "bridge_quote", QUOTE_ARGS)["kind"] == "evm-xreserve"

    b2 = FakeBridge(environment="testnet")
    cp2 = _inbound_private_checkpoint(b2)
    out = dispatch_tool(b2, "bridge_complete", {"checkpoint": cp2, "confirm": True})
    assert "secret_nonce" in out["error"] and out["how_to_fix"] and b2.events == []


def test_no_write_tool_ever_echoes_the_secret_nonce():
    """Every write tool, with and without confirm: the nonce is not in the JSON result."""
    results = []

    b = FakeBridge()
    private = {**QUOTE_ARGS, "mint_mode": "private", "secret_nonce": NONCE}
    results.append(dispatch_tool(b, "bridge_execute", dict(private)))
    results.append(dispatch_tool(b, "bridge_execute", {**private, "confirm": True}))
    results.append(dispatch_tool(b, "bridge_shield", {"asset": "aleo/eth", "amount_atomic": 1,
                                                      "secret_nonce": NONCE}))
    results.append(dispatch_tool(b, "bridge_shield", {"asset": "aleo/eth", "amount_atomic": 1,
                                                      "secret_nonce": NONCE, "confirm": True}))
    results.append(dispatch_tool(b, "bridge_unshield", {"asset": "aleo/eth", "amount_atomic": 1,
                                                        "secret_nonce": NONCE}))
    results.append(dispatch_tool(b, "bridge_unshield", {"asset": "aleo/eth", "amount_atomic": 1,
                                                        "secret_nonce": NONCE, "confirm": True}))

    b2 = FakeBridge(ethereum=False)
    cp, _ = _aleo_out_checkpoint(b2)
    results.append(dispatch_tool(b2, "bridge_resume", {"checkpoint": cp, "secret_nonce": NONCE}))
    results.append(dispatch_tool(b2, "bridge_resume", {"checkpoint": cp, "secret_nonce": NONCE, "confirm": True}))

    b3 = FakeBridge(environment="testnet")
    cp2 = _inbound_private_checkpoint(b3)
    results.append(dispatch_tool(b3, "bridge_complete", {"checkpoint": cp2, "secret_nonce": NONCE}))
    results.append(dispatch_tool(b3, "bridge_complete", {"checkpoint": cp2, "secret_nonce": NONCE, "confirm": True}))

    assert len(results) == 10
    for result in results:
        blob = json.dumps(result)
        assert NONCE not in blob and "secretNonce" not in blob and "secret_nonce" not in blob


def test_unconfigured_chain_returns_a_structured_error():
    b = FakeBridge(ethereum=False)                                            # no EVM connection
    for tool, args in (("bridge_quote", QUOTE_ARGS), ("bridge_execute", {**QUOTE_ARGS, "confirm": True})):
        out = dispatch_tool(b, tool, dict(args))
        assert out["error"] and out["error_type"] in {"ConfigurationError", "BridgeError"}
        assert "EVM_PRIVATE_KEY" in out["how_to_fix"] and "ETHEREUM_RPC_URL" in out["how_to_fix"]
        assert "next" not in out
    assert b.calls == [] and b.events == []
    out = dispatch_tool(FakeBridge(solana=False), "bridge_quote",
                        {"source": "solana/sol", "destination": "aleo/sol", "amount": "0.1",
                         "recipient": ALEO_RECIPIENT})
    assert "SOLANA_PRIVATE_KEY" in out["how_to_fix"]


def test_bridge_error_from_a_read_is_structured_too():
    out = dispatch_tool(FakeBridge(), "bridge_quote", {**QUOTE_ARGS, "amount": "0"})
    assert out["error"] and out["error_type"] == "InvalidAmountError"


def test_ambiguous_send_surfaces_recover_guidance_and_the_checkpoint():
    b = FakeBridge()
    route_id = "xreserve:ethereum/usdc->aleo/usdcx"
    approval = Receipt(id="0x" + "11" * 32, protocol="xreserve", status=Status.SOURCE_APPROVAL_PENDING,
                       protocol_state={"routeId": route_id, "approvalTxIds": ["0x" + "11" * 32],
                                       "sourceSender": EVM_ADDRESS})
    b.eth.intermediates = [approval]
    b.eth.send_error = BridgeError("Ethereum deposit 0x" + "bb" * 32 + " may already be broadcast: "
                                   "the RPC response was lost")
    out = dispatch_tool(b, "bridge_execute", {**QUOTE_ARGS, "confirm": True})
    assert out["error_type"] == "BridgeError" and "may already be broadcast" in out["error"]
    assert out["next"] == "recover" and "bridge_get_progress" in out["how_to_fix"]
    assert out["checkpoint"]["receiptId"] == "0x" + "11" * 32
    assert out["checkpoint"]["route"]["id"] == route_id
    json.dumps(out)


def test_resume_write_error_surfaces_recover_guidance_and_the_checkpoint():
    b = FakeBridge()
    plan = prepare(b.registry, source="ethereum/usdc", destination="aleo/usdcx", amount="2",
                   recipient=ALEO_RECIPIENT, sender=EVM_ADDRESS, mint_mode="private")
    approval = "0x" + "11" * 32
    receipt = Receipt(id=approval, protocol="xreserve", status=Status.SOURCE_SUBMISSION_PENDING,
                      protocol_state={"routeId": plan.route_id, "approvalTxIds": [approval],
                                      "sourceSender": EVM_ADDRESS,
                                      "hookData": "0x" + b.eth.hook_data.hex()})
    b.eth.recover_result = receipt                      # history still has no deposit
    b.eth.intermediates = [receipt]                     # re-checkpointed before the deposit goes out
    b.eth.send_error = BridgeError("Ethereum deposit may already be broadcast: the RPC response was lost")
    cp = create_checkpoint(plan, receipt, b.registry).to_dict()

    out = dispatch_tool(b, "bridge_resume", {"checkpoint": cp, "secret_nonce": NONCE, "confirm": True})
    assert out["error_type"] == "BridgeError" and "may already be broadcast" in out["error"]
    assert out["next"] == "recover" and "bridge_get_progress" in out["how_to_fix"]
    assert out["checkpoint"]["receiptId"] == approval and out["checkpoint"]["route"]["id"] == plan.route_id
    assert NONCE not in json.dumps(out)


def test_complete_write_error_surfaces_recover_guidance_and_the_prepared_checkpoint():
    b = FakeBridge(environment="testnet")
    cp = _inbound_private_checkpoint(b)
    b.submit_error = BridgeError("Aleo mint at1fake1 may already be broadcast: the node answer was lost")

    out = dispatch_tool(b, "bridge_complete", {"checkpoint": cp, "secret_nonce": NONCE, "confirm": True})
    assert out["error_type"] == "BridgeError" and out["next"] == "recover"
    assert "bridge_get_progress" in out["how_to_fix"]
    # the pre-broadcast checkpoint made it out: the proved mint can be rebroadcast, never re-proved
    assert out["checkpoint"]["destination"]["preparedTransaction"]["transactionId"] == "at1fake1"
    assert NONCE not in json.dumps(out)
