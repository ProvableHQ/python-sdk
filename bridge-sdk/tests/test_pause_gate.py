"""quote / execute / resume / complete refuse a paused or unreadable route before anything is signed."""
import pytest

from aleo_bridge.errors import PauseCheckError, RoutePausedError, RouteUnavailableError
from aleo_bridge.lifecycle import complete, execute, quote, resume
from aleo_bridge.types import PauseStatus, PauseSwitch, to_progress
from tests.fakes.fake_bridge import ALEO_RECIPIENT, EVM_ADDRESS, FakeBridge
from tests.test_resume_complete import _prepared_progress, _ready

ALEO_ETH = "hyperlane:aleo/eth->ethereum/eth"
QUOTE = dict(source_chain="aleo", source_asset="eth", destination_chain="ethereum", destination_asset="eth",
             amount="0.000000000000000001", recipient=EVM_ADDRESS)


def paused(rid, side="source"):
    return PauseStatus(rid, (PauseSwitch("aleo", side, "hyp_warp_token_eth_v2.aleo", "pause[true]", "paused", "true"),))


def unknown(rid):
    return PauseStatus(rid, (PauseSwitch("ethereum", "destination", "0x8888888199b2Df864bf678259607d6D5EBb4e3Ce", "paused()",
                                         "unknown", "ConnectionError: down after 7 attempts"),))


def test_quote_refuses_paused_before_any_module_call():
    b = FakeBridge(ethereum=False)
    b.pause_statuses[ALEO_ETH] = paused(ALEO_ETH)
    with pytest.raises(RoutePausedError, match=r"pause\[true\] on hyp_warp_token_eth_v2.aleo") as info:
        quote(b, **QUOTE)
    assert b.calls == [] and b.pause_checks == [(ALEO_ETH, ("source", "destination"))]
    assert isinstance(info.value, RouteUnavailableError) and info.value.status.paused


def test_quote_and_execute_refuse_unknown_fail_closed():
    b = FakeBridge(ethereum=False)
    plan = quote(b, **QUOTE).plan                                     # open: quote works
    b.pause_statuses[ALEO_ETH] = unknown(ALEO_ETH)
    with pytest.raises(PauseCheckError, match="after 7 attempts"):
        quote(b, **QUOTE)
    with pytest.raises(PauseCheckError, match="never proceeds"):
        execute(b, plan)
    assert b.events == [] and all(c[0] != "hyperlane.transfer_remote" for c in b.calls)


def test_execute_checks_both_sides_after_require_active():
    b = FakeBridge(ethereum=False)
    plan = quote(b, **QUOTE).plan
    b.pause_checks.clear()
    b.pause_statuses[ALEO_ETH] = paused(ALEO_ETH, side="destination")
    with pytest.raises(RoutePausedError, match="destination side"):
        execute(b, plan)
    assert b.events == [] and b.pause_checks == [(ALEO_ETH, ("source", "destination"))]
    # a metadata-required route is refused by availability first; no pause read happens for it
    b.pause_checks.clear()
    with pytest.raises(RouteUnavailableError, match="metadata-required"):
        quote(b, route="hyperlane:aleo/aleo->ethereum/aleo", amount="1", recipient=EVM_ADDRESS)
    assert b.pause_checks == []


def test_resume_checks_both_sides_and_refuses_paused():
    b = FakeBridge(ethereum=False)
    plan, _serialized, progress = _prepared_progress(b)
    b.pause_statuses[plan.route_id] = paused(plan.route_id)
    with pytest.raises(RoutePausedError):
        resume(b, progress)
    assert b.submitted == [] and b.pause_checks == [(plan.route_id, ("source", "destination"))]


def test_complete_checks_destination_only():
    b = FakeBridge(environment="testnet")
    plan, _payload, _hash, ready = _ready(b)
    b.xreserve.expected_secret_nonce = "7scalar"
    complete(b, to_progress(plan, ready), secret_nonce="7scalar")
    assert b.pause_checks[-1] == (plan.route_id, ("destination",))


def test_complete_ignores_source_side_switches():
    b = FakeBridge(environment="testnet")
    plan, _payload, _hash, ready = _ready(b)
    b.xreserve.expected_secret_nonce = "7scalar"
    rid = plan.route_id
    b.pause_statuses[rid] = PauseStatus(rid, (
        PauseSwitch("sepolia", "source", "0x008888878f94C0d87defdf0B07f46B93C1934442", "paused()", "paused", "true"),
        PauseSwitch("aleo-testnet", "destination", "test_usdcx_bridge_v2.aleo", "paused[true]", "open", "false")))
    out = complete(b, to_progress(plan, ready), secret_nonce="7scalar")   # only the requested side is returned
    assert out.next == "wait" and len(b.submitted) == 1
    b.pause_statuses[rid] = PauseStatus(rid, (
        PauseSwitch("aleo-testnet", "destination", "test_usdcx_bridge_v2.aleo", "paused[true]", "paused", "true"),))
    with pytest.raises(RoutePausedError, match="test_usdcx_bridge_v2.aleo"):
        complete(b, to_progress(plan, ready), secret_nonce="7scalar")
    assert len(b.submitted) == 1


def test_gate_runs_once_per_verb():
    b = FakeBridge(ethereum=False)
    plan = quote(b, **QUOTE).plan
    assert len(b.pause_checks) == 1
    execute(b, plan)
    assert len(b.pause_checks) == 2
