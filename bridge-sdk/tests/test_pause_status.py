"""``Bridge.pause_status``: every switch on both ends of a route, per protocol family, read live."""
import base64
import json
from pathlib import Path

import pytest
import requests

from aleo_bridge import Bridge, Ethereum, Solana
from aleo_bridge.errors import ConfigurationError
from aleo_bridge.lifecycle import prepare
from aleo_bridge.registry import DEFAULT_REGISTRY, Registry
from tests.conftest import FakeAleo
from tests.fakes.fake_solana import FakeSolanaClient
from tests.fakes.fake_web3 import fake_web3, make_bridge
from tests.test_pause_evm import ISM_PAUSABLE, MAILBOX, XRESERVE, _hook_tree, addr, boolean, sel, u8

FIXTURES = Path(__file__).parent / "fixtures"
ALEO_DOMAIN = DEFAULT_REGISTRY.chain("aleo").protocol_domains["hyperlane"]
SOL_PDA = "JDkpV5CsSbhyGhHhirC5DjGPTcuKWUVHtBZ5MFsgu3ZW"
TRANSMITTER = "0x81D40F21F12A8F0E3252Bccb954D722d4c464B64"
ALEO_RECIPIENT = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
EVM_RECIPIENT = "0x00000000000000000000000000000000000000ee"


def _ism_tree(paused: bool) -> dict:
    return {(MAILBOX, sel("defaultIsm()")): addr(ISM_PAUSABLE), (ISM_PAUSABLE, sel("moduleType()")): u8(6),
            (ISM_PAUSABLE, sel("paused()")): boolean(paused)}


def _warp(program: str, far_domain: int, *, paused: bool = False, enrolled: bool = True) -> dict:
    routers = {f"{far_domain}u32": "{ domain: 1u32, recipient: [0u8], gas: 1u128 }"} if enrolled else {}
    return {program: {"pause": {"true": str(paused).lower()}, "remote_routers": routers, "app_metadata": {}}}


def _sol_pda_bytes() -> bytes:
    return base64.b64decode(json.loads((FIXTURES / "sealevel-token-pda-sol.json").read_text())["data_base64"])


def _keyed(status):
    return {(s.chain_id, s.side, s.switch): s.state for s in status.switches}


def test_aleo_to_ethereum_hyperlane_reads_both_ends():
    b = make_bridge(ethereum=Ethereum(w3=fake_web3(calls=_ism_tree(False))),
                    mappings={**_warp("hyp_warp_token_eth_v2.aleo", 1), "arc20_eth.aleo": {"pause__": {}}})
    status = b.pause_status("hyperlane:aleo/eth->ethereum/eth")
    assert _keyed(status) == {("aleo", "source", "pause[true]"): "open", ("aleo", "source", "remote_routers[1u32]"): "open",
                              ("aleo", "source", "pause__[false]"): "open",
                              ("ethereum", "destination", "PausableIsm.paused()"): "open"}
    assert not status.paused and not status.unknown


def test_ethereum_to_aleo_hyperlane_source_hook_tree_and_destination_aleo():
    b = make_bridge(ethereum=Ethereum(w3=fake_web3(calls=_hook_tree(False))),
                    mappings={**_warp("hyp_warp_token_eth_v2.aleo", 1, paused=True), "arc20_eth.aleo": {"pause__": {}}})
    status = b.pause_status("hyperlane:ethereum/eth->aleo/eth")
    keyed = _keyed(status)
    assert keyed[("ethereum", "source", "PausableHook.paused()")] == "open"
    assert keyed[("aleo", "destination", "pause[true]")] == "paused" and status.paused
    assert "hyp_warp_token_eth_v2.aleo" in status.reason


def test_aleo_to_solana_uses_the_paired_routes_token_pda():
    sol = Solana(client=FakeSolanaClient(accounts={SOL_PDA: _sol_pda_bytes()}))
    b = make_bridge(ethereum=None, mappings={**_warp("hyp_warp_token_sol_v2.aleo", 1399811149), "arc20_sol.aleo": {"pause__": {}}})
    b.solana = sol
    status = b.pause_status("hyperlane:aleo/sol->solana/sol")
    assert _keyed(status)[("solana", "destination", f"remote_routers[{ALEO_DOMAIN}]")] == "open" and not status.paused
    b.solana = Solana(client=FakeSolanaClient(accounts={}))                   # account gone → un-enrolled
    assert _keyed(b.pause_status("hyperlane:aleo/sol->solana/sol"))[("solana", "destination", f"remote_routers[{ALEO_DOMAIN}]")] == "paused"


def test_solana_to_aleo():
    b = make_bridge(ethereum=None, mappings={**_warp("hyp_warp_token_sol_v2.aleo", 1399811149), "arc20_sol.aleo": {"pause__": {}}})
    b.solana = Solana(client=FakeSolanaClient(accounts={SOL_PDA: _sol_pda_bytes()}))
    keyed = _keyed(b.pause_status("hyperlane:solana/sol->aleo/sol"))
    assert keyed[("solana", "source", f"remote_routers[{ALEO_DOMAIN}]")] == "open"
    assert keyed[("aleo", "destination", "pause[true]")] == "open" and keyed[("aleo", "destination", "remote_routers[1399811149u32]")] == "open"


def test_xreserve_both_directions_and_arc():
    aleo = {"usdcx_bridge_v2.aleo": {"paused": {}}, "usdcx_stablecoin.aleo": {"pause": {"true": "false"}}}
    eth = Ethereum(w3=fake_web3(calls={(XRESERVE, sel("paused()")): boolean(False)}))
    arc = Ethereum(w3=fake_web3(calls={(XRESERVE, sel("paused()")): boolean(True)}))
    b = make_bridge(ethereum=eth, evm={"arc": arc}, mappings=aleo)
    out = _keyed(b.pause_status("xreserve:aleo/usdcx->ethereum/usdc"))
    assert out == {("aleo", "source", "paused[true]"): "open", ("aleo", "source", "pause[true]"): "open",
                   ("ethereum", "destination", "paused()"): "open"}
    inbound = _keyed(b.pause_status("xreserve:ethereum/usdc->aleo/usdcx"))
    assert inbound == {("ethereum", "source", "paused()"): "open", ("aleo", "destination", "paused[true]"): "open",
                       ("aleo", "destination", "pause[true]"): "open"}
    assert b.pause_status("xreserve:aleo/usdcx->arc/usdc").paused          # Arc's XReserve is paused in this fake


def test_cctp_reads_message_transmitter_on_both_chains():
    eth = Ethereum(w3=fake_web3(calls={(TRANSMITTER, sel("paused()")): boolean(False)}))
    arc = Ethereum(w3=fake_web3(calls={(TRANSMITTER, sel("paused()")): boolean(True)}))
    b = make_bridge(ethereum=eth, evm={"arc": arc})
    status = b.pause_status("cctp:ethereum/usdc->arc/usdc")
    assert _keyed(status) == {("ethereum", "source", "paused()"): "open", ("arc", "destination", "paused()"): "paused"}


def test_sides_destination_only_skips_source_reads():
    w3 = fake_web3(calls=_hook_tree(False))
    b = make_bridge(ethereum=Ethereum(w3=w3), mappings={**_warp("hyp_warp_token_eth_v2.aleo", 1), "arc20_eth.aleo": {"pause__": {}}})
    status = b.pause_status("hyperlane:ethereum/eth->aleo/eth", sides=("destination",))
    assert {s.side for s in status.switches} == {"destination"} and w3.provider.call_log == []


def test_destination_evm_falls_back_to_registry_public_rpc(monkeypatch):
    urls = []

    def factory(url):
        urls.append(url)
        return Ethereum(w3=fake_web3(calls={(XRESERVE, sel("paused()")): boolean(False)}))
    monkeypatch.setattr("aleo_bridge.pause.Ethereum", factory)
    b = make_bridge(ethereum=None, mappings={"usdcx_bridge_v2.aleo": {"paused": {}}, "usdcx_stablecoin.aleo": {"pause": {}}})
    status = b.pause_status("xreserve:aleo/usdcx->ethereum/usdc")
    assert not status.paused and not status.unknown
    assert urls == [DEFAULT_REGISTRY.public_read_rpc_urls("ethereum")[0]]


def test_public_rpc_rotation_on_transient_errors(monkeypatch):
    urls = []
    public = DEFAULT_REGISTRY.public_read_rpc_urls("ethereum")
    assert len(public) >= 2

    def factory(url):
        urls.append(url)
        scripted = requests.ConnectionError("down") if url == public[0] else boolean(False)
        return Ethereum(w3=fake_web3(calls={(XRESERVE, sel("paused()")): scripted}))
    monkeypatch.setattr("aleo_bridge.pause.Ethereum", factory)
    b = make_bridge(ethereum=None, mappings={"usdcx_bridge_v2.aleo": {"paused": {}}, "usdcx_stablecoin.aleo": {"pause": {}}})
    status = b.pause_status("xreserve:aleo/usdcx->ethereum/usdc")       # first retry is immediate, so no real wait
    assert not status.unknown and urls == [public[0], public[1]]


def test_no_connection_and_no_public_urls_is_a_configuration_error():
    r = DEFAULT_REGISTRY
    bare = Registry(r.version, r.chains(), r.assets(), r.routes(include_unavailable=True))
    b = Bridge(FakeAleo(mappings={"usdcx_bridge_v2.aleo": {"paused": {}}, "usdcx_stablecoin.aleo": {"pause": {}}}), registry=bare)
    with pytest.raises(ConfigurationError, match="ethereum=.*set_rpc_url.*ETHEREUM_RPC_URL"):
        b.pause_status("xreserve:aleo/usdcx->ethereum/usdc")


def test_bridge_pause_status_accepts_route_id_route_and_plan():
    b = make_bridge(ethereum=Ethereum(w3=fake_web3(calls=_ism_tree(False))),
                    mappings={**_warp("hyp_warp_token_eth_v2.aleo", 1), "arc20_eth.aleo": {"pause__": {}}})
    route = b.registry.route("hyperlane:aleo/eth->ethereum/eth")
    plan = prepare(b.registry, source_chain="aleo", source_asset="eth", destination_chain="ethereum", destination_asset="eth",
                   amount="0.000000000000000001", recipient=EVM_RECIPIENT)
    assert b.pause_status(route.id) == b.pause_status(route) == b.pause_status(plan)


def test_pause_status_never_caches(monkeypatch):
    b = make_bridge(ethereum=Ethereum(w3=fake_web3(calls=_ism_tree(False))),
                    mappings={**_warp("hyp_warp_token_eth_v2.aleo", 1), "arc20_eth.aleo": {"pause__": {}}})
    reads = []
    original = b.mapping_value
    monkeypatch.setattr(b, "mapping_value", lambda *a: (reads.append(a), original(*a))[1])
    b.pause_status("hyperlane:aleo/eth->ethereum/eth")
    first = len(reads)
    b.pause_status("hyperlane:aleo/eth->ethereum/eth")
    assert first == 3 and len(reads) == 2 * first
