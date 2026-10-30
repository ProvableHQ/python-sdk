"""Aleo-side pause switches: boolean mappings and the enrolled remote-router table, with the declared-mapping guard."""
import pytest
import requests

from aleo_bridge.errors import ConfigurationError
from aleo_bridge.pause import read_aleo_flag, read_aleo_remote_router
from tests.fakes.fake_web3 import make_bridge

WARP = "hyp_warp_token_eth_v2.aleo"


def _bridge(mappings, **kw):
    return make_bridge(ethereum=None, mappings=mappings, **kw)


def _flag(b, **kw):
    return read_aleo_flag(b, chain_id="aleo", side="source", program=WARP, mapping="pause", key="true", **kw)


def test_paused_open_and_missing_key():
    assert _flag(_bridge({WARP: {"pause": {"true": "true"}}})).state == "paused"
    s = _flag(_bridge({WARP: {"pause": {"true": "false"}}}))
    assert (s.state, s.switch, s.locator, s.detail, s.chain_id, s.side) == ("open", "pause[true]", WARP, "false", "aleo", "source")
    assert _flag(_bridge({WARP: {"pause": {}}})).state == "open"           # declared, key never set


def test_undeclared_mapping_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="does not declare mapping 'pause'"):
        _flag(_bridge({WARP: {"remote_routers": {}}}))


def test_missing_program_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="is not served by this node"):
        _flag(_bridge({}, missing_programs={WARP}))


def test_malformed_value_is_a_configuration_error():
    with pytest.raises(ConfigurationError, match="expected a boolean"):
        _flag(_bridge({WARP: {"pause": {"true": "7u8"}}}))


def test_remote_router_present_is_open_and_missing_is_paused():
    b = _bridge({WARP: {"remote_routers": {"1u32": "{ domain: 1u32, recipient: [0u8], gas: 1u128 }"}}})
    assert read_aleo_remote_router(b, chain_id="aleo", side="source", program=WARP, domain=1).state == "open"
    s = read_aleo_remote_router(b, chain_id="aleo", side="source", program=WARP, domain=1399811149)
    assert s.state == "paused" and s.switch == "remote_routers[1399811149u32]"


def test_transient_read_errors_become_unknown_after_retries(monkeypatch):
    b = _bridge({WARP: {"pause": {"true": "false"}}})

    def boom(*a, **k):
        raise requests.ConnectionError("rpc down")
    monkeypatch.setattr(b, "mapping_value", boom)
    sleeps = []
    s = _flag(b, sleep=sleeps.append)
    assert s.state == "unknown" and "after 7 attempts" in s.detail and sleeps == [0, 0, 0.5, 1.0, 2.0, 4.0]
