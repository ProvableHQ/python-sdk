"""Private-mint recovery tools preserve nonce guidance before any submission."""
import json

import pytest

from aleo_bridge.checkpoint import create_checkpoint
from aleo_bridge.agent import dispatch_tool
from aleo_bridge.errors import BridgeError
from tests.test_private_mint_lifecycle import QUOTE, _bridge, _ready, _reserve, _resumable


def _recovery_case(phase, nonce, *, store=True):
    bridge = _bridge()
    if not store:
        bridge.private_mint_identities = None
    plan, receipt = (_resumable if phase == "resume" else _ready)(bridge, nonce)
    bridge.xreserve.expected_secret_nonce = nonce
    bridge.eth.recover_result = receipt
    return bridge, create_checkpoint(plan, receipt, bridge.registry).to_dict()


@pytest.mark.parametrize("phase", ["resume", "complete"])
def test_missing_supplied_nonce_keeps_configuration_guidance(phase):
    confirm = True
    bridge, checkpoint = _recovery_case(phase, "123scalar")
    out = dispatch_tool(bridge, f"bridge_{phase}", {"checkpoint": checkpoint, "confirm": confirm})
    assert out["error_type"] == "ConfigurationError"
    assert "secret_nonce" in out["how_to_fix"]
    assert "next" not in out and "checkpoint" not in out
    assert "123scalar" not in json.dumps(out)
    assert bridge.events == []
    assert not any(name in {"eth.deposit_usdc", "xreserve.private_mint"} for name, *_ in bridge.calls)


def test_execute_treats_an_empty_nonce_as_omitted():
    bridge = _bridge()
    out = dispatch_tool(bridge, "bridge_execute", {**QUOTE, "secret_nonce": "", "confirm": True})
    assert "error" not in out
    [identity] = bridge.private_mint_identities.load()
    assert bridge.calls[-1][1]["secret_nonce"] == identity.secret_nonce
    assert identity.secret_nonce not in json.dumps(out)


@pytest.mark.parametrize("phase", ["resume", "complete"])
def test_recovery_treats_an_empty_nonce_as_omitted(phase):
    identity = _reserve(_bridge())
    bridge, checkpoint = _recovery_case(phase, identity.secret_nonce)
    out = dispatch_tool(bridge, f"bridge_{phase}", {"checkpoint": checkpoint, "secret_nonce": "", "confirm": True})
    assert "error" not in out
    assert bridge.calls[-1][1]["secret_nonce"] == identity.secret_nonce
    assert identity.secret_nonce not in json.dumps(out)


@pytest.mark.parametrize("phase", ["resume", "complete"])
def test_zero_scalar_recovery_needs_no_store(phase):
    bridge, checkpoint = _recovery_case(phase, "0scalar", store=False)
    out = dispatch_tool(bridge, f"bridge_{phase}", {"checkpoint": checkpoint, "confirm": True})
    assert "error" not in out
    assert bridge.calls[-1][1]["secret_nonce"] == "0scalar"


@pytest.mark.parametrize("phase", ["resume", "complete"])
def test_derived_nonce_submission_failure_still_requires_recovery(phase):
    identity = _reserve(_bridge())
    bridge, checkpoint = _recovery_case(phase, identity.secret_nonce)
    error = BridgeError("Transaction may already be broadcast: response lost")
    if phase == "resume":
        bridge.eth.send_error = error
    else:
        bridge.submit_error = error
    out = dispatch_tool(bridge, f"bridge_{phase}", {"checkpoint": checkpoint, "confirm": True})
    assert out["next"] == "recover" and "bridge_get_progress" in out["how_to_fix"]
    assert identity.secret_nonce not in json.dumps(out)


def test_prepared_mint_rebroadcast_needs_neither_nonce_nor_store():
    bridge, checkpoint = _recovery_case("complete", "123scalar", store=False)
    bridge.submit_error = BridgeError("Transaction may already be broadcast: response lost")
    interrupted = dispatch_tool(bridge, "bridge_complete", {
        "checkpoint": checkpoint, "secret_nonce": "123scalar", "confirm": True})
    assert interrupted["next"] == "recover"
    prepared = interrupted["checkpoint"]["destination"]["preparedTransaction"]
    bridge.submit_error = None
    out = dispatch_tool(bridge, "bridge_complete", {
        "checkpoint": interrupted["checkpoint"], "confirm": True})
    assert "error" not in out
    assert bridge.aleo.submitted == [prepared["serializedTransaction"]]
    assert sum(event[0] == "delegate_prepared" for event in bridge.events) == 1


@pytest.mark.parametrize("phase", ["resume", "complete"])
def test_preview_does_not_resolve_or_save_nonce(phase, monkeypatch):
    from aleo_bridge import private_mint as pm
    bridge, checkpoint = _recovery_case(phase, "123scalar")
    def forbidden(*args, **kwargs):
        pytest.fail("Preview must not resolve or persist a nonce")
    monkeypatch.setattr(pm, "resolve_secret_nonce", forbidden)
    out = dispatch_tool(bridge, f"bridge_{phase}", {"checkpoint": checkpoint})
    assert "error" not in out and "how_to_confirm" in out
    assert bridge.private_mint_identities.load() == []


@pytest.mark.parametrize("phase", ["resume", "complete"])
def test_wrong_phase_does_not_resolve_nonce(phase, monkeypatch):
    from aleo_bridge import private_mint as pm
    bridge, checkpoint = _recovery_case("complete" if phase == "resume" else "resume", "123scalar")
    def forbidden(*args, **kwargs):
        pytest.fail("Wrong-phase calls must not resolve a nonce")
    monkeypatch.setattr(pm, "resolve_secret_nonce", forbidden)
    out = dispatch_tool(bridge, f"bridge_{phase}", {"checkpoint": checkpoint, "confirm": True})
    assert out["error_type"] == "NotResumableError"


@pytest.mark.parametrize("broken", ["identities", "counter", "exhausted", "permissions"])
def test_execute_store_error_preserves_configuration_guidance(tmp_path, monkeypatch, broken):
    from aleo_bridge import private_mint as pm
    bridge = _bridge()
    store = pm.FilePrivateMintIdentityStore(tmp_path / "identities.json")
    bridge.private_mint_identities = store
    if broken == "identities":
        store.path.write_text("not json")
    elif broken == "permissions":
        def denied():
            raise PermissionError("Store cannot be read")
        monkeypatch.setattr(store, "load", denied)
    else:
        store.counter_path.write_text("²" if broken == "counter" else str(pm.MAX_COUNTER + 1))
    out = dispatch_tool(bridge, "bridge_execute", {**QUOTE, "confirm": True})
    assert out["error_type"] == "ConfigurationError"
    assert "next" not in out
    assert "never bridge_execute" not in out.get("how_to_fix", "")
    assert not any(name == "eth.deposit_usdc" for name, *_ in bridge.calls)
    assert bridge.events == []


@pytest.mark.parametrize("phase", ["resume", "complete"])
@pytest.mark.parametrize("broken", ["read", "lock"])
def test_recovery_storage_errors_are_structured(phase, broken, monkeypatch):
    from aleo_bridge import private_mint as pm
    identity = _reserve(_bridge())
    bridge, checkpoint = _recovery_case(phase, identity.secret_nonce)
    def denied(*args, **kwargs):
        raise PermissionError("private storage unavailable")
    if broken == "read":
        monkeypatch.setattr(bridge.private_mint_identities, "load", denied)
    else:
        monkeypatch.setattr(pm, "_exclusive", denied)
    out = dispatch_tool(bridge, f"bridge_{phase}", {"checkpoint": checkpoint, "confirm": True})
    assert out["error_type"] == "ConfigurationError" and "next" not in out
    assert bridge.events == []


@pytest.mark.parametrize("phase", ["resume", "complete"])
def test_preview_reports_missing_nonce_without_derivation(phase):
    bridge, checkpoint = _recovery_case(phase, "123scalar", store=False)
    out = dispatch_tool(bridge, f"bridge_{phase}", {"checkpoint": checkpoint})
    assert out["error_type"] == "ConfigurationError"
    assert "secret_nonce" in out["error"] and "confirmation_required" not in out


@pytest.mark.parametrize("phase", ["resume", "complete"])
def test_stored_nonce_error_never_echoes_secret(phase):
    from dataclasses import replace
    identity = _reserve(_bridge())
    bridge, checkpoint = _recovery_case(phase, identity.secret_nonce)
    bridge.private_mint_identities.save([replace(identity, secret_nonce=" " + identity.secret_nonce)])
    out = dispatch_tool(bridge, f"bridge_{phase}", {"checkpoint": checkpoint, "confirm": True})
    assert out["error_type"] == "ConfigurationError"
    assert identity.secret_nonce not in json.dumps(out)
    assert "restore" in out["error"]


def test_execute_sender_mismatch_does_not_reserve_identity():
    bridge = _bridge()
    out = dispatch_tool(bridge, "bridge_execute", {
        **QUOTE, "sender": "0x" + "98" * 20, "confirm": True})
    assert "error" in out
    assert bridge.private_mint_identities.load() == []
    assert bridge.private_mint_identities._next == 0


@pytest.mark.parametrize("phase", ["resume", "complete"])
@pytest.mark.parametrize("source", ["stored", "zero", "derived"])
def test_preview_accepts_available_nonce_without_scanning(phase, source, monkeypatch):
    from aleo_bridge import private_mint as pm
    identity = _reserve(_bridge())
    nonce = "0scalar" if source == "zero" else identity.secret_nonce
    bridge, checkpoint = _recovery_case(phase, nonce)
    if source == "stored":
        bridge.private_mint_identities.save([identity])
        bridge.aleo.default_account = None
    elif source == "zero":
        bridge.private_mint_identities = None
        bridge.aleo.default_account = None
    def forbidden(*args, **kwargs):
        pytest.fail("Preview must not scan, lock, reserve or save")
    monkeypatch.setattr(pm, "recover_identity", forbidden)
    monkeypatch.setattr(pm, "reserve_identity", forbidden)
    monkeypatch.setattr(pm, "_exclusive", forbidden)
    if bridge.private_mint_identities is not None:
        monkeypatch.setattr(bridge.private_mint_identities, "save", forbidden)
    out = dispatch_tool(bridge, f"bridge_{phase}", {"checkpoint": checkpoint})
    assert "error" not in out and out["confirmation_required"]


@pytest.mark.parametrize("phase", ["resume", "complete"])
def test_storage_failure_reports_safe_errno(phase, monkeypatch):
    import errno
    from tests.test_private_mint_lifecycle import _bridge
    identity = _reserve(_bridge())
    bridge, checkpoint = _recovery_case(phase, identity.secret_nonce)
    def failed(*args):
        raise OSError(errno.ENOSPC, "sensitive-secret-text", "/secret/path")
    monkeypatch.setattr(bridge.private_mint_identities, "load", failed)
    out = dispatch_tool(bridge, f"bridge_{phase}", {"checkpoint": checkpoint, "confirm": True})
    assert out["error_type"] == "ConfigurationError" and "ENOSPC" in out["error"]
    assert "sensitive-secret-text" not in json.dumps(out) and "/secret/path" not in json.dumps(out)
