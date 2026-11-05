"""Verulink routes compose with the shared pause gate after PR 86."""
import pytest
import requests

from aleo_bridge import PauseCheckError, RoutePausedError
from aleo_bridge.pause import PauseReader
from tests.fakes.fake_verulink import Harness, M


@pytest.mark.parametrize("outbound", [True, False])
def test_verulink_pause_status_reads_both_chains(outbound):
    h = Harness(outbound)
    status = h.bridge.pause_status(h.plan)
    assert len(status.switches) == 6
    assert not status.paused and not status.unknown
    aleo_side = "source" if outbound else "destination"
    assert len([s for s in status.switches if s.side == aleo_side and s.chain_id == "aleo"]) == 4
    assert {s.locator for s in status.switches if s.chain_id == "bsc"} == {M["evmTokenService"], M["evmBridge"]}


@pytest.mark.parametrize("raw,state", [(None, "unknown"), ("true", "unknown"), ("0u8", "paused")])
def test_aleo_active_setting_is_not_a_boolean_pause_flag(raw, state):
    h = Harness()
    h.aleo.mappings[M["aleoBridge"]]["bridge_settings"]["3u8"] = raw
    status = h.bridge.pause_status(h.plan)
    assert next(s for s in status.switches if s.switch == "bridge_settings[3u8]").state == state
    with pytest.raises(RoutePausedError if state == "paused" else PauseCheckError):
        h.quote()


def test_destination_only_status_skips_source_evm_reads():
    h = Harness(False)
    status = h.bridge.pause_status(h.plan, sides=("destination",))
    assert len(status.switches) == 4
    assert {s.side for s in status.switches} == {"destination"}
    assert not h.rpc.calls


def test_agent_pause_schema_and_dispatch_include_verulink():
    from aleo_bridge import bridge_tools, dispatch_tool
    h = Harness()
    schemas = {t["name"]: t["input_schema"] for t in bridge_tools()}
    assert "verulink" in schemas["bridge_pause_status"]["properties"]["bridge_protocol"]["enum"]
    result = dispatch_tool(h.bridge, "bridge_pause_status", {
        "source_chain": "aleo", "source_asset": "aleo", "destination_chain": "bsc", "bridge_protocol": "verulink"})
    assert "error" not in result
    assert len(result["switches"]) == 6
    assert not result["paused"] and not result["unknown"]


def test_verulink_pause_reads_are_fresh_and_block_execution():
    h = Harness()
    quote = h.quote()
    h.rpc.values["paused"] = True
    with pytest.raises(RoutePausedError):
        h.bridge.execute(quote.plan)
    assert not h.sent and not h.aleo.submitted


def test_verulink_mapping_failure_retries_and_stays_unknown(monkeypatch):
    h = Harness()
    original = h.bridge.mapping_value
    calls = []

    def mapping(program, name, key):
        if name == "bridge_settings":
            calls.append(key)
            raise requests.ConnectionError("unavailable")
        return original(program, name, key)

    monkeypatch.setattr(h.bridge, "mapping_value", mapping)
    status = PauseReader(h.bridge, sleep=lambda _: None).status(h.bridge.registry.route(h.plan.route_id))
    assert status.unknown and len(calls) == 7
    assert not h.sent and not h.aleo.submitted
