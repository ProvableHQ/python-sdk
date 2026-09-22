"""Tests for the version-1 checkpoint allowlist and FileCheckpointStore (brief §2.10).

``lifecycle.prepare`` does not exist yet (plan 4 task order runs checkpoint.py before
lifecycle.py, since EvmCall.send/SolCall.send depend on create_checkpoint) — plans are
built by hand from the real DEFAULT_REGISTRY instead of going through prepare().
"""
import json
import os
import stat
import time
from decimal import Decimal
from pathlib import Path

import pytest

from aleo_bridge.checkpoint import Checkpoint, FileCheckpointStore, create_checkpoint
from aleo_bridge.errors import CheckpointInvalidError
from aleo_bridge.registry import DEFAULT_REGISTRY, Registry
from aleo_bridge.types import Plan, Receipt, Status

RECIPIENT = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
APPROVAL = "0x" + "11" * 32
SOURCE = "0x" + "22" * 32
EVM1 = "0x0000000000000000000000000000000000000001"


def _make_plan(registry: Registry, *, source, destination, amount, recipient,
               sender=None, protocol=None, mint_mode="public") -> Plan:
    route = registry.find_route(source, destination, protocol=protocol)
    src = registry.asset(source)
    dst = registry.asset(destination)
    amount_atomic = int(Decimal(amount) * (10 ** src.decimals))
    return Plan(route_id=route.id, registry_version=registry.version, protocol=route.protocol,
                environment=route.environment, source_asset_id=src.id, destination_asset_id=dst.id,
                amount=amount, amount_atomic=amount_atomic, recipient=recipient, sender=sender,
                mint_mode=mint_mode, steps=())


def _plan(**kw):
    base = dict(source="sepolia/usdc", destination="aleo-testnet/usdcx", amount="2",
                recipient=RECIPIENT, mint_mode="private")
    base.update(kw)
    return _make_plan(DEFAULT_REGISTRY, **base)


def test_allowlist_persists_intent_and_ids_only():
    plan = _plan()
    receipt = Receipt(id="at1destination", protocol="xreserve", status=Status.DESTINATION_CONFIRMING,
                      source_tx_id=SOURCE, destination_tx_id="at1destination",
                      protocol_state={"routeId": plan.route_id, "approvalTxIds": [APPROVAL],
                                      "payload": "0xdeadbeef", "messageHash": "0x" + "33" * 32,
                                      "nonce": "0x" + "44" * 32, "attestation": "0x" + "55" * 65,
                                      "amountAtomic": "2000000", "maxFeeAtomic": "100000",
                                      "remoteRecipientBytes32": "0x" + "66" * 32,
                                      "secretNonce": "7scalar"})
    cp = create_checkpoint(plan, receipt, DEFAULT_REGISTRY)
    assert cp.to_dict() == {
        "version": 1,
        "receiptId": "at1destination",
        "intent": {"source": {"chain": "sepolia", "asset": "usdc"},
                   "destination": {"chain": "aleo-testnet", "asset": "usdcx"},
                   "bridgeProtocol": "xreserve", "amount": "2", "recipient": RECIPIENT,
                   "mintMode": "private"},
        "route": {"id": plan.route_id, "registryVersion": plan.registry_version},
        "source": {"approvalTransactionIds": [APPROVAL], "transactionId": SOURCE},
        "destination": {"transactionId": "at1destination"},
    }
    text = cp.to_json()
    for forbidden in ("payload", "messageHash", "nonce", "attestation", "secretNonce",
                      "amountAtomic", "maxFeeAtomic", "remoteRecipientBytes32", "0xdeadbeef"):
        assert forbidden not in text
    assert cp.id == "at1destination"
    assert Checkpoint.from_json(text) == cp


def test_mint_mode_only_when_destination_is_an_aleo_program():
    outbound = _make_plan(DEFAULT_REGISTRY, source="aleo/wbtc", destination="ethereum/wbtc",
                          amount="0.1", recipient=EVM1)
    cp = create_checkpoint(outbound, Receipt(id="at1x", protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                             source_tx_id="at1x", protocol_state={"routeId": outbound.route_id}),
                           DEFAULT_REGISTRY)
    assert "mintMode" not in cp.intent and cp.intent["bridgeProtocol"] == "hyperlane"
    assert cp.source == {"transactionId": "at1x"} and cp.destination is None
    assert "destination" not in cp.to_dict()


def test_sender_from_plan_or_source_sender():
    plan = _plan(sender=EVM1)
    receipt = Receipt(id=APPROVAL, protocol="xreserve", status=Status.SOURCE_APPROVAL_PENDING,
                      protocol_state={"routeId": plan.route_id, "approvalTxIds": [APPROVAL],
                                      "sourceSender": "0x0000000000000000000000000000000000000002"})
    assert create_checkpoint(plan, receipt, DEFAULT_REGISTRY).intent["sender"] == EVM1
    plan2 = _plan()
    assert create_checkpoint(plan2, receipt.replace(protocol_state={**receipt.protocol_state, "routeId": plan2.route_id}),
                             DEFAULT_REGISTRY).intent["sender"] == "0x0000000000000000000000000000000000000002"
    assert create_checkpoint(plan2, Receipt(id=SOURCE, protocol="xreserve", status=Status.SOURCE_CONFIRMING,
                                            source_tx_id=SOURCE, protocol_state={"routeId": plan2.route_id}),
                             DEFAULT_REGISTRY).intent.get("sender") is None


def test_solana_blockhash_pair_both_or_neither():
    plan = _make_plan(DEFAULT_REGISTRY, source="solana/sol", destination="aleo/sol", amount="0.000000001",
                      recipient=RECIPIENT, sender="11111111111111111111111111111111")
    ok = Receipt(id="sig", protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id="sig",
                 protocol_state={"routeId": plan.route_id, "blockhash": "recent", "lastValidBlockHeight": "123456789"})
    assert create_checkpoint(plan, ok, DEFAULT_REGISTRY).source == {
        "transactionId": "sig", "blockhash": "recent", "lastValidBlockHeight": "123456789"}
    with pytest.raises(CheckpointInvalidError, match="blockhash"):
        create_checkpoint(plan, ok.replace(protocol_state={"routeId": plan.route_id, "blockhash": "recent"}),
                          DEFAULT_REGISTRY)
    with pytest.raises(CheckpointInvalidError, match="blockhash"):
        create_checkpoint(plan, ok.replace(protocol_state={"routeId": plan.route_id, "blockhash": "recent",
                                                            "lastValidBlockHeight": "abc"}), DEFAULT_REGISTRY)


def test_delivery_verification_pair_both_or_neither():
    plan = _make_plan(DEFAULT_REGISTRY, source="aleo/sol", destination="solana/sol", amount="0.000000001",
                      recipient="11111111111111111111111111111111")
    receipt = Receipt(id="at1s", protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id="at1s",
                      protocol_state={"routeId": plan.route_id, "destinationBalanceBeforeAtomic": "100",
                                      "expectedDestinationIncreaseAtomic": "1"})
    cp = create_checkpoint(plan, receipt, DEFAULT_REGISTRY)
    assert cp.delivery_verification == {"balanceBeforeAtomic": "100", "expectedIncreaseAtomic": "1"}
    with pytest.raises(CheckpointInvalidError, match="verification"):
        create_checkpoint(plan, receipt.replace(protocol_state={"routeId": plan.route_id,
                                                                 "destinationBalanceBeforeAtomic": "100"}),
                          DEFAULT_REGISTRY)


def test_prepared_transactions_and_hook_data():
    plan = _make_plan(DEFAULT_REGISTRY, source="aleo/eth", destination="ethereum/eth",
                      amount="0.000000000000000001", recipient=EVM1)
    serialized = json.dumps({"type": "execute", "id": "at1prepared", "fee": {}})
    cp = create_checkpoint(plan, Receipt(id="at1prepared", protocol="hyperlane",
                                         status=Status.SOURCE_SUBMISSION_PENDING,
                                         protocol_state={"routeId": plan.route_id,
                                                         "preparedTransaction": serialized}),
                           DEFAULT_REGISTRY)
    assert cp.source == {"preparedTransaction": {"transactionId": "at1prepared",
                                                 "serializedTransaction": serialized}}
    with pytest.raises(CheckpointInvalidError, match="prepared transaction"):
        create_checkpoint(plan, Receipt(id="x", protocol="hyperlane", status=Status.SOURCE_SUBMISSION_PENDING,
                                        protocol_state={"routeId": plan.route_id, "preparedTransaction": ""}),
                          DEFAULT_REGISTRY)
    inbound = _plan()
    cp2 = create_checkpoint(inbound, Receipt(id="at1mint", protocol="xreserve",
                                             status=Status.DESTINATION_ACTION_REQUIRED, source_tx_id=SOURCE,
                                             protocol_state={"routeId": inbound.route_id, "hookData": "0x02" + "00" * 64,
                                                             "preparedDestinationTransaction": serialized}),
                            DEFAULT_REGISTRY)
    assert cp2.source == {"transactionId": SOURCE, "hookData": "0x02" + "00" * 64}
    assert cp2.destination == {"preparedTransaction": {"transactionId": "at1mint",
                                                       "serializedTransaction": serialized}}


def test_rejects_receipts_from_other_routes_and_bad_approvals():
    plan = _plan()
    with pytest.raises(CheckpointInvalidError, match="does not match"):
        create_checkpoint(plan, Receipt(id=SOURCE, protocol="xreserve", status=Status.SOURCE_CONFIRMING,
                                        protocol_state={"routeId": "xreserve:wrong/route"}), DEFAULT_REGISTRY)
    with pytest.raises(CheckpointInvalidError, match="does not match"):
        create_checkpoint(plan, Receipt(id=SOURCE, protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                        protocol_state={"routeId": plan.route_id}), DEFAULT_REGISTRY)
    with pytest.raises(CheckpointInvalidError, match="approval"):
        create_checkpoint(plan, Receipt(id=SOURCE, protocol="xreserve", status=Status.SOURCE_CONFIRMING,
                                        protocol_state={"routeId": plan.route_id, "approvalTxIds": [1, 2]}),
                          DEFAULT_REGISTRY)


def test_from_dict_accepts_veil_shaped_checkpoints_without_receipt_id():
    veil = {"version": 1,
            "intent": {"source": {"chain": "aleo", "asset": "eth"}, "destination": {"chain": "ethereum", "asset": "eth"},
                       "bridgeProtocol": "hyperlane", "amount": "0.000000000000000001", "recipient": EVM1},
            "route": {"id": "hyperlane:aleo/eth->ethereum/eth", "registryVersion": DEFAULT_REGISTRY.version},
            "source": {"preparedTransaction": {"transactionId": "at1prepared", "serializedTransaction": "{}"}}}
    cp = Checkpoint.from_dict(veil)
    assert cp.id == "at1prepared" and cp.version == 1 and cp.delivery_verification is None
    assert Checkpoint.from_dict({**veil, "source": {"transactionId": "at1src"}}).id == "at1src"
    assert Checkpoint.from_dict({**veil, "source": {"approvalTransactionIds": [APPROVAL]}}).id == APPROVAL
    with pytest.raises(CheckpointInvalidError):
        Checkpoint.from_dict({"version": 2, "intent": {}, "route": {}})
    with pytest.raises(CheckpointInvalidError):
        Checkpoint.from_dict({**veil, "source": {}})


def test_file_store_roundtrip_mode_and_atomic_rename(tmp_path):
    store = FileCheckpointStore(tmp_path / "cps")
    plan = _plan()
    cp = create_checkpoint(plan, Receipt(id=SOURCE, protocol="xreserve", status=Status.SOURCE_CONFIRMING,
                                         source_tx_id=SOURCE, protocol_state={"routeId": plan.route_id}),
                           DEFAULT_REGISTRY)
    store.save(cp)
    files = list((tmp_path / "cps").iterdir())
    assert [f.name for f in files] == [f"{SOURCE}.json"]
    assert stat.S_IMODE(os.stat(files[0]).st_mode) == 0o600
    assert not list((tmp_path / "cps").glob("*.tmp"))
    assert store.load(SOURCE) == cp
    assert store.load("missing") is None
    assert store.list() == [cp]
    store.delete(SOURCE)
    store.delete(SOURCE)                                # idempotent
    assert store.list() == [] and store.load(SOURCE) is None


def test_file_store_skips_unreadable_files_and_reports_them(tmp_path):
    # I1: one file the store cannot read back must never hide the transfers next to it — list()
    # returns the healthy records, and the rejects come back through list_with_problems().
    store = FileCheckpointStore(tmp_path)
    plan = _plan()
    cp = create_checkpoint(plan, Receipt(id=SOURCE, protocol="xreserve", status=Status.SOURCE_CONFIRMING,
                                         source_tx_id=SOURCE, protocol_state={"routeId": plan.route_id}),
                           DEFAULT_REGISTRY)
    store.save(cp)
    (tmp_path / "future.json").write_text(json.dumps({"version": 2, "intent": {}, "route": {}}), encoding="utf-8")
    (tmp_path / "garbage.json").write_text("{not json at all", encoding="utf-8")

    assert store.list() == [cp]
    checkpoints, problems = store.list_with_problems()
    assert checkpoints == [cp]
    assert {Path(p.path).name for p in problems} == {"future.json", "garbage.json"}
    assert {p.error_type for p in problems} == {"CheckpointInvalidError"}
    assert all(p.error for p in problems)
    assert [p.to_dict() for p in store.list_problems()] == [p.to_dict() for p in problems]
    assert set(problems[0].to_dict()) == {"error", "error_type", "path"}


def test_file_store_sanitizes_ids_and_orders_by_mtime(tmp_path):
    store = FileCheckpointStore(str(tmp_path))
    plan = _make_plan(DEFAULT_REGISTRY, source="aleo/eth", destination="ethereum/eth",
                      amount="0.000000000000000001", recipient=EVM1)
    a = create_checkpoint(plan, Receipt(id="at1a", protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                        source_tx_id="at1a", protocol_state={"routeId": plan.route_id}), DEFAULT_REGISTRY)
    weird = create_checkpoint(plan, Receipt(id="../evil id", protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                            source_tx_id="../evil id", protocol_state={"routeId": plan.route_id}),
                              DEFAULT_REGISTRY)
    store.save(a)
    time.sleep(0.01)                                    # keep mtimes ordered on coarse filesystems
    store.save(weird)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["___evil_id.json", "at1a.json"]
    assert [c.id for c in store.list()] == ["at1a", "../evil id"]     # oldest first
    assert store.load("../evil id") == weird
