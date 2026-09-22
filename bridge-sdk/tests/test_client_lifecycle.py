"""The Bridge methods are one-liners; test that each forwards every argument to
lifecycle.* by calling the unbound methods with the FakeBridge as ``self``.

Deviation from the task-9 brief (recorded in task-9-report.md): the brief's Step 3 also has
``__init__.py`` import ``bridge_tools``/``dispatch_tool`` from a new ``.agent`` module and add
``agent_guide()`` (reading a packaged ``AGENTS.md``), and has ``__main__.py`` print that guide.
The task-9 controller notes (ruling 5) explicitly forbid creating ``agent.py``/``AGENTS.md`` in
that task — those are Task 10/12's files.  Task 10 landed ``agent.py`` and those three exports,
so the export pin below now covers them; ``AGENTS.md`` is still Task 12's, and ``__main__.py``
is left untouched.
"""
import inspect
import json
from pathlib import Path

import aleo_bridge
from aleo_bridge import agent, lifecycle
from aleo_bridge.checkpoint import Checkpoint, FileCheckpointStore, create_checkpoint
from aleo_bridge.client import Bridge
from aleo_bridge.types import Receipt, Status
from tests.fakes.fake_bridge import ALEO_RECIPIENT, EVM_ADDRESS, SOL_ADDRESS, FakeBridge

# The full pre-existing __all__ (plans 1-3, before this task's additive edit) — pinned so this
# task's edit can only ever ADD names, never drop one (controller ruling 1).
PRE_EXISTING_EXPORTS = [
    "__version__", "AmbiguousRouteError", "AttestationError", "BridgeError", "ChainMismatchError",
    "CheckpointInvalidError", "ConfigurationError", "DeliveryUnknownError", "InsufficientBalanceError",
    "InvalidAmountError", "InvalidRecipientError", "MissingExtraError", "NotResumableError",
    "PollingTimeoutError", "RegistryVersionMismatchError", "RouteNotFoundError", "RouteUnavailableError",
    "UnsupportedRouteError",
    "Asset", "Chain", "DEFAULT_REGISTRY", "Locator", "Privacy", "Registry", "Route", "validate_registry",
    "CALLER_BOUNDARIES", "TERMINAL", "AleoHyperlaneQuote", "AleoXReserveQuote", "Attestation", "BridgeStatus",
    "BurnReceipt", "ChainStatus", "DepositReceipt", "DispatchReceipt", "EvmHyperlaneQuote", "EvmXReserveQuote",
    "Fee", "GasQuote", "MintReceipt", "Plan", "PreparedTx", "PrivacyReceipt", "Progress", "Quote", "Receipt",
    "SolanaHyperlaneQuote", "Status", "Step", "to_progress",
    "AleoCall", "Bridge", "CircleClient", "DEFAULT_ENDPOINT", "EMPTY_MERKLE_PROOF_PAIR", "FreezeList",
    "HyperlaneModule", "PrivacyModule", "Profile", "XReserveModule",
    "Checkpoint", "CheckpointStore", "FileCheckpointStore", "create_checkpoint",
    "EthModule", "Ethereum", "EvmCall",
    "DEFAULT_SOLANA_RPC_URL", "Solana", "SolCall", "SolModule",
]

# This task's own additions (lifecycle module + the pure prepare() convenience import), then the
# agent surface Task 9 deferred to Task 10 (bridge_tools/dispatch_tool + the packaged guide).
NEW_EXPORTS = ["lifecycle", "prepare", "agent_guide", "bridge_tools", "dispatch_tool"]


def _spy(monkeypatch, name):
    seen = {}

    def fake(bridge, *args, **kwargs):
        seen["args"], seen["kwargs"], seen["bridge"] = args, kwargs, bridge
        return "result"
    monkeypatch.setattr(lifecycle, name, fake)
    return seen


def test_quote_forwards(monkeypatch):
    seen = _spy(monkeypatch, "quote")
    b = FakeBridge()
    assert Bridge.quote(b, "ethereum/usdc", "aleo/usdcx", amount="2", recipient=ALEO_RECIPIENT,
                        mint_mode="private", secret_nonce="7scalar", sender=EVM_ADDRESS, protocol="xreserve") == "result"
    assert seen["bridge"] is b and seen["args"] == ()
    assert seen["kwargs"] == dict(source="ethereum/usdc", destination="aleo/usdcx", amount="2", amount_atomic=None,
                                  recipient=ALEO_RECIPIENT, sender=EVM_ADDRESS, protocol="xreserve",
                                  mint_mode="private", secret_nonce="7scalar")


def test_execute_wait_get_status_recover_resume_complete_forward(monkeypatch):
    b = FakeBridge()
    seen = _spy(monkeypatch, "execute")
    cb = lambda cp: None
    Bridge.execute(b, "PLAN", on_checkpoint=cb, proving="local", mode="signer", record="r", merkle_proof="m",
                   gas_payment_microcredits=5, secret_nonce="1scalar", poll_seconds=2.0, timeout_seconds=3.0)
    assert seen["args"] == ("PLAN",) and seen["kwargs"] == dict(
        on_checkpoint=cb, proving="local", mode="signer", record="r", merkle_proof="m", gas_payment_microcredits=5,
        secret_nonce="1scalar", poll_seconds=2.0, timeout_seconds=3.0)
    seen = _spy(monkeypatch, "wait")
    on_err = lambda exc: None
    Bridge.wait(b, "PROGRESS", until=[Status.DELIVERY_PENDING], poll_seconds=1, timeout_seconds=2, on_update=cb,
                on_error=on_err, max_consecutive_errors=9)
    assert seen["args"] == ("PROGRESS",) and seen["kwargs"] == dict(until=[Status.DELIVERY_PENDING], poll_seconds=1,
                                                                    timeout_seconds=2, on_update=cb, on_error=on_err,
                                                                    max_consecutive_errors=9)
    seen = _spy(monkeypatch, "get_status")
    Bridge.get_status(b, "PLAN", "RECEIPT")
    assert seen["args"] == ("PLAN", "RECEIPT")
    seen = _spy(monkeypatch, "recover")
    Bridge.recover(b, {"version": 1})
    assert seen["args"] == ({"version": 1},)
    seen = _spy(monkeypatch, "resume")
    Bridge.resume(b, "PROGRESS", on_checkpoint=cb, secret_nonce="1scalar", poll_seconds=1.0, timeout_seconds=9.0)
    assert seen["kwargs"] == dict(on_checkpoint=cb, secret_nonce="1scalar", poll_seconds=1.0, timeout_seconds=9.0,
                                  proving="delegate")
    seen = _spy(monkeypatch, "complete")
    Bridge.complete(b, "PROGRESS", secret_nonce="7scalar", on_checkpoint=cb)
    assert seen["kwargs"] == dict(secret_nonce="7scalar", on_checkpoint=cb, proving="delegate")


class _Boom:
    """Any attribute access returns a callable that raises — stands in for ``eth``/``sol`` so a
    test can assert ``pending()`` never touches the network, not merely that it happened to answer
    correctly this time."""

    def __getattr__(self, name):
        def raiser(*args, **kwargs):
            raise AssertionError(f"Bridge.pending() must not touch the network (called .{name})")
        return raiser


def test_pending_recovers_every_stored_checkpoint_offline(tmp_path):
    # Fix round 1 (F1): pending() must not touch the network — one unreachable chain must never
    # hide every other in-flight transfer. An EVM and a Solana checkpoint in the same store, with
    # both side-chain modules wired to raise on ANY call, still both come back.
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(solana=True, checkpoints=store)
    b._eth, b._sol = _Boom(), _Boom()

    evm_plan = lifecycle.prepare(b.registry, source="ethereum/wbtc", destination="aleo/wbtc", amount="0.001",
                                 recipient=ALEO_RECIPIENT, sender=EVM_ADDRESS)
    store.save(create_checkpoint(evm_plan, Receipt(id="0x" + "11" * 32, protocol="hyperlane",
                                                   status=Status.SOURCE_CONFIRMING, source_tx_id="0x" + "11" * 32,
                                                   protocol_state={"routeId": evm_plan.route_id}), b.registry))
    sol_plan = lifecycle.prepare(b.registry, source="solana/sol", destination="aleo/sol", amount="0.000000001",
                                 recipient=ALEO_RECIPIENT, sender=SOL_ADDRESS)
    store.save(create_checkpoint(sol_plan, Receipt(id="sig", protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                                   source_tx_id="sig", protocol_state={"routeId": sol_plan.route_id}),
                                 b.registry))

    out = Bridge.pending(b)
    assert len(out) == 2 and all(p.next == "wait" for p in out) and b.calls == []
    assert {p.receipt.source_tx_id for p in out} == {"0x" + "11" * 32, "sig"}
    assert Bridge.pending(FakeBridge(ethereum=False)) == []


def test_pending_folds_a_malformed_checkpoint_into_a_failed_entry(tmp_path):
    # F1: a checkpoint that cannot be interpreted (well-formed JSON/route, but nothing to build a
    # receipt from — e.g. a truncated file that lost its "source" block) must not hide the
    # healthy checkpoints alongside it in the same store.
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(ethereum=False, checkpoints=store)
    plan = lifecycle.prepare(b.registry, source="aleo/eth", destination="ethereum/eth",
                             amount="0.000000000000000001", recipient=EVM_ADDRESS)
    store.save(create_checkpoint(plan, Receipt(id="at1good", protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                               source_tx_id="at1good", protocol_state={"routeId": plan.route_id}),
                                 b.registry))
    store.save(create_checkpoint(plan, Receipt(id="at1ghost", protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                               protocol_state={"routeId": plan.route_id}), b.registry))  # no source tx at all

    out = Bridge.pending(b)
    assert len(out) == 2
    good, bad = out
    assert good.next == "wait" and good.receipt.source_tx_id == "at1good"
    assert bad.next == "failed" and bad.error is not None and "no submitted source transaction" in bad.error


def test_pending_reports_unreadable_files_and_version_mismatches_instead_of_dropping_them(tmp_path):
    # I1/m4: neither a file the STORE cannot parse (a v2 record, a garbage file) nor a checkpoint
    # THIS client cannot interpret (registry version mismatch) may abort or silently shrink the
    # listing — the healthy transfer still comes back, and each problem is its own failed entry.
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(ethereum=False, checkpoints=store)
    plan = lifecycle.prepare(b.registry, source="aleo/eth", destination="ethereum/eth",
                             amount="0.000000000000000001", recipient=EVM_ADDRESS)
    healthy = create_checkpoint(plan, Receipt(id="at1good", protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                              source_tx_id="at1good", protocol_state={"routeId": plan.route_id}),
                                b.registry)
    store.save(healthy)
    store.save(Checkpoint(version=1, receipt_id="at1old", intent=healthy.intent,
                          route={"id": plan.route_id, "registryVersion": "0.0.0-ancient"},
                          source={"transactionId": "at1old"}))
    (tmp_path / "future.json").write_text(json.dumps({"version": 2, "intent": {}, "route": {}}), encoding="utf-8")
    (tmp_path / "garbage.json").write_text("{not json", encoding="utf-8")

    out = Bridge.pending(b)
    progresses = [p for p in out if not isinstance(p, dict)]
    failures = [p for p in out if isinstance(p, dict)]
    assert len(progresses) == 1 and progresses[0].receipt.source_tx_id == "at1good"
    assert all(f["next"] == "failed" and f["error"] and f["error_type"] for f in failures)
    mismatch = [f for f in failures if f.get("checkpoint_id") == "at1old"]
    assert len(mismatch) == 1 and mismatch[0]["error_type"] == "RegistryVersionMismatchError"
    unreadable = sorted(Path(f["path"]).name for f in failures if "path" in f)
    assert unreadable == ["future.json", "garbage.json"]
    assert b.calls == []                                   # still offline


def test_bridge_verb_signatures_are_a_superset_of_the_lifecycle_verb_they_forward_to():
    # F2 guard: a future edit that drops a kwarg from a Bridge verb (without dropping it from the
    # matching lifecycle verb too) fails loudly here instead of silently losing a caller option.
    for name in ("quote", "execute", "get_status", "wait", "recover", "resume", "complete"):
        bridge_params = set(inspect.signature(getattr(Bridge, name)).parameters)
        lifecycle_params = set(inspect.signature(getattr(lifecycle, name)).parameters) - {"bridge"}
        assert lifecycle_params <= bridge_params, name


def test_public_exports_pin_pre_existing_set_and_add_lifecycle_names():
    for name in PRE_EXISTING_EXPORTS + NEW_EXPORTS:
        assert hasattr(aleo_bridge, name), name
    assert set(PRE_EXISTING_EXPORTS) <= set(aleo_bridge.__all__)
    assert set(NEW_EXPORTS) <= set(aleo_bridge.__all__)
    assert aleo_bridge.__version__ == "0.1.0"
    assert aleo_bridge.lifecycle is lifecycle
    assert aleo_bridge.prepare is lifecycle.prepare
    assert aleo_bridge.bridge_tools is agent.bridge_tools
    assert aleo_bridge.dispatch_tool is agent.dispatch_tool
    # AGENTS.md is Task 12's generated file: until it ships, the guide is a pointer, never an error
    guide = aleo_bridge.agent_guide()
    assert isinstance(guide, str) and guide
    if not (Path(aleo_bridge.__file__).with_name("AGENTS.md")).exists():
        assert "codegen/gen_context.py" in guide
