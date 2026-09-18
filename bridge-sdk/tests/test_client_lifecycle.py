"""The Bridge methods are one-liners; test that each forwards every argument to
lifecycle.* by calling the unbound methods with the FakeBridge as ``self``.

Deviation from the task-9 brief (recorded in task-9-report.md): the brief's Step 3 also has
``__init__.py`` import ``bridge_tools``/``dispatch_tool`` from a new ``.agent`` module and add
``agent_guide()`` (reading a packaged ``AGENTS.md``), and has ``__main__.py`` print that guide.
The task-9 controller notes (ruling 5) explicitly forbid creating ``agent.py``/``AGENTS.md`` in
this task — those are Task 10/12's files — so this test file does not exercise them, and
``__main__.py`` is left untouched.
"""
import aleo_bridge
from aleo_bridge import lifecycle
from aleo_bridge.checkpoint import FileCheckpointStore, create_checkpoint
from aleo_bridge.client import Bridge
from aleo_bridge.types import Receipt, Status
from tests.fakes.fake_bridge import ALEO_RECIPIENT, EVM_ADDRESS, FakeBridge

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

# This task's own additions (lifecycle module + the pure prepare() convenience import).
NEW_EXPORTS = ["lifecycle", "prepare"]


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
    Bridge.wait(b, "PROGRESS", until=[Status.DELIVERY_PENDING], poll_seconds=1, timeout_seconds=2, on_update=cb)
    assert seen["args"] == ("PROGRESS",) and seen["kwargs"] == dict(until=[Status.DELIVERY_PENDING], poll_seconds=1,
                                                                    timeout_seconds=2, on_update=cb)
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


def test_pending_recovers_every_stored_checkpoint(tmp_path):
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(ethereum=False, checkpoints=store)
    plan = lifecycle.prepare(b.registry, source="aleo/eth", destination="ethereum/eth",
                             amount="0.000000000000000001", recipient=EVM_ADDRESS)
    for tx in ("at1one", "at1two"):
        store.save(create_checkpoint(plan, Receipt(id=tx, protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                                                   source_tx_id=tx, protocol_state={"routeId": plan.route_id}), b.registry))
    out = Bridge.pending(b)
    assert [p.receipt.source_tx_id for p in out] == ["at1one", "at1two"] and all(p.next == "wait" for p in out)
    assert Bridge.pending(FakeBridge(ethereum=False)) == []


def test_public_exports_pin_pre_existing_set_and_add_lifecycle_names():
    for name in PRE_EXISTING_EXPORTS + NEW_EXPORTS:
        assert hasattr(aleo_bridge, name), name
    assert set(PRE_EXISTING_EXPORTS) <= set(aleo_bridge.__all__)
    assert set(NEW_EXPORTS) <= set(aleo_bridge.__all__)
    assert aleo_bridge.__version__ == "0.1.0"
    assert aleo_bridge.lifecycle is lifecycle
    assert aleo_bridge.prepare is lifecycle.prepare
