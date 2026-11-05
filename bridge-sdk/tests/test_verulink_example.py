"""Round-trip retries must use the saved transfer, never a second source send."""
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from aleo_bridge import BridgeError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))


def example():
    return importlib.import_module("verulink_roundtrip")


def options(path, **extra):
    return dict(aleo_address="aleo-address", bsc_address="0xaddress", amount="20",
                state_path=path, **extra)


def test_preview_quotes_both_legs_without_a_journal(tmp_path):
    module, bridge = example(), Mock()
    bridge.quote.side_effect = [SimpleNamespace(amount_out="19.98"), SimpleNamespace(amount_out="19.96002")]
    module.run_roundtrip(bridge, **options(tmp_path / "journey.json"))
    assert bridge.quote.call_args_list[1].kwargs["amount"] == "19.98"
    bridge.execute.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_execution_requires_explicit_ack_before_wallet_loading(monkeypatch, tmp_path):
    module = example()
    monkeypatch.delenv("BRIDGE_LIVE_MAINNET_EXECUTE", raising=False)
    build = Mock()
    monkeypatch.setattr(module, "build_bridge", build)
    with pytest.raises(BridgeError, match="BRIDGE_LIVE_MAINNET_EXECUTE"):
        module.main(["--aleo-address", "aleo-address", "--bsc-address", "0xaddress",
                     "--state", str(tmp_path / "journey.json"), "--execute"])
    build.assert_not_called()


def test_uncertain_submission_cannot_be_executed_again(monkeypatch, tmp_path):
    module, bridge = example(), Mock()
    monkeypatch.setenv("BRIDGE_LIVE_MAINNET_EXECUTE", module.ACK)
    bridge.aleo_address.return_value = "aleo-address"
    bridge.evm.return_value.conn.require_address.return_value = "0xaddress"
    bridge.evm.return_value.conn.w3.eth.get_balance.return_value = 10**18
    bridge.quote.return_value = SimpleNamespace(amount_out="19.98", plan=object())
    bridge.execute.side_effect = RuntimeError("uncertain")
    path = tmp_path / "journey.json"
    with pytest.raises(RuntimeError, match="uncertain"):
        module.run_roundtrip(bridge, **options(path, execute=True))
    assert json.loads(path.read_text())["legs"][0]["source_attempted"]
    with pytest.raises(BridgeError, match="uncertain"):
        module.run_roundtrip(bridge, **options(path, execute=True))
    assert bridge.execute.call_count == 1
    assert not path.with_suffix(".json.lock").exists()


def test_existing_lock_prevents_execution_and_remains_owned(monkeypatch, tmp_path):
    module, bridge = example(), Mock()
    monkeypatch.setenv("BRIDGE_LIVE_MAINNET_EXECUTE", module.ACK)
    path = tmp_path / "journey.json"
    lock = path.with_suffix(".json.lock")
    lock.write_text("other-process")
    with pytest.raises(BridgeError, match="lock"):
        module.run_roundtrip(bridge, **options(path, execute=True))
    bridge.execute.assert_not_called()
    assert lock.read_text() == "other-process"


def test_ambiguous_claim_needs_transaction_hash(monkeypatch, tmp_path):
    module, bridge = example(), Mock()
    monkeypatch.setenv("BRIDGE_LIVE_MAINNET_EXECUTE", module.ACK)
    bridge.aleo_address.return_value = "aleo-address"
    bridge.evm.return_value.conn.require_address.return_value = "0xaddress"
    path = tmp_path / "journey.json"
    state = {"version": 1, "aleo": "aleo-address", "bsc": "0xaddress", "amount": "20",
             "legs": [{"amount": "20", "source_attempted": True, "claim_attempted": True,
                       "checkpoint": {"source": {"transactionId": "source"}}}]}
    path.write_text(json.dumps(state))
    bridge.recover.return_value = SimpleNamespace(next="complete", receipt=SimpleNamespace(
        source_tx_id="source", destination_tx_id=None), plan=SimpleNamespace(
            route_id=module.ROUTES[0], amount="20", amount_atomic=20_000_000, sender="aleo-address", recipient="0xaddress"))
    with pytest.raises(BridgeError, match="Claim submission is uncertain"):
        module.run_roundtrip(bridge, **options(path, execute=True))
    bridge.execute.assert_not_called()
    bridge.complete.assert_not_called()


def test_copied_checkpoint_cannot_authorize_a_claim(monkeypatch, tmp_path):
    module, bridge = example(), Mock()
    monkeypatch.setenv("BRIDGE_LIVE_MAINNET_EXECUTE", module.ACK)
    bridge.aleo_address.return_value = "aleo-address"
    bridge.evm.return_value.conn.require_address.return_value = "0xaddress"
    path = tmp_path / "journey.json"
    path.write_text(json.dumps({"version": 1, "aleo": "aleo-address", "bsc": "0xaddress", "amount": "20",
        "legs": [{"amount": "20", "source_attempted": True, "checkpoint": {"source": {"transactionId": "other"}}}]}))
    bridge.recover.return_value = SimpleNamespace(next="complete", plan=SimpleNamespace(
        route_id=module.ROUTES[0], amount="20", amount_atomic=20_000_000, sender="aleo-address", recipient="different-recipient"),
        receipt=SimpleNamespace(source_tx_id="other", destination_tx_id=None))
    with pytest.raises(BridgeError, match="checkpoint differs"):
        module.run_roundtrip(bridge, **options(path, execute=True))
    bridge.complete.assert_not_called()
    bridge.execute.assert_not_called()


def test_roundtrip_uses_received_amount_and_reverifies_completed_legs(monkeypatch, tmp_path):
    from aleo_bridge import DEFAULT_REGISTRY, Receipt, Status, create_checkpoint, prepare, to_progress
    from tests.test_verulink import ALEO, EVM
    module, bridge = example(), Mock()
    monkeypatch.setenv("BRIDGE_LIVE_MAINNET_EXECUTE", module.ACK)
    bridge.registry = DEFAULT_REGISTRY
    bridge.aleo_address.return_value = ALEO
    bridge.evm.return_value.conn.require_address.return_value = EVM
    bridge.evm.return_value.conn.w3.eth.get_balance.return_value = 10**18
    completed = {}

    def quote(**kwargs):
        plan = prepare(DEFAULT_REGISTRY, **kwargs)
        return SimpleNamespace(plan=plan, amount_out="19.98" if plan.route_id == module.ROUTES[0] else "19.96002")

    def execute(plan, *, on_checkpoint, **kwargs):
        outbound = plan.route_id == module.ROUTES[0]
        r = Receipt("transfer-" + str(outbound), "verulink", Status.COMPLETED,
                    source_tx_id="source-" + str(outbound), destination_tx_id="claim-" + str(outbound),
                    protocol_state={"routeId": plan.route_id, "packet": {"amount": "19980000" if outbound else "19960020"}})
        cp = create_checkpoint(plan, r, DEFAULT_REGISTRY)
        completed[cp.id] = to_progress(plan, r)
        on_checkpoint(cp)
        return completed[cp.id]

    bridge.quote.side_effect = quote
    bridge.execute.side_effect = execute
    bridge.recover.side_effect = lambda cp: completed[cp["receiptId"]]
    args = dict(aleo_address=ALEO, bsc_address=EVM, amount="20.0", state_path=tmp_path / "journey.json", execute=True)
    module.run_roundtrip(bridge, **args)
    assert bridge.execute.call_count == 2
    assert bridge.execute.call_args_list[1].args[0].amount == "19.98"
    module.run_roundtrip(bridge, **args)
    assert bridge.execute.call_count == 2 and bridge.recover.call_count == 4
