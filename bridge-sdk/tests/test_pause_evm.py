"""EVM-side pause switches: ``paused()`` and the Hyperlane Mailbox hook / ISM trees."""
import pytest
import requests
from eth_abi import encode
from web3 import Web3

from aleo_bridge._keccak import keccak256
from aleo_bridge.errors import ConfigurationError
from aleo_bridge.pause import read_evm_paused, read_mailbox_hook_pauses, read_mailbox_ism_pauses
from tests.fakes.fake_web3 import fake_web3

MAILBOX = "0xc005dc82818d67AF737725bD4bf75435d065D239"
XRESERVE = "0x8888888199b2Df864bf678259607d6D5EBb4e3Ce"
FALLBACK, AGG, PAUSABLE, MERKLE, IGP, FEE = (f"0x{n:040x}" for n in range(0xA1, 0xA7))
ISM_AGG, ISM_NULL_PLAIN, ISM_ROUTING, ISM_AGG2, ISM_PAUSABLE = (f"0x{n:040x}" for n in range(0xB1, 0xB6))
ALEO_DOMAIN = 1634493807


def sel(sig: str) -> str:
    return "0x" + keccak256(sig.encode())[:4].hex()


def addr(a: str) -> bytes:
    return encode(["address"], [a])


def addrs(*a: str) -> bytes:
    return encode(["address[]"], [list(a)])


def u8(n: int) -> bytes:
    return encode(["uint8"], [n])


def boolean(b: bool) -> bytes:
    return encode(["bool"], [b])


def test_paused_true_false_and_unknown():
    w3 = fake_web3(calls={(XRESERVE, sel("paused()")): boolean(True)})
    s = read_evm_paused(w3, chain_id="ethereum", side="destination", address=XRESERVE)
    assert (s.state, s.switch, s.locator, s.chain_id, s.side) == ("paused", "paused()", XRESERVE, "ethereum", "destination")
    w3 = fake_web3(calls={(XRESERVE, sel("paused()")): boolean(False)})
    assert read_evm_paused(w3, chain_id="ethereum", side="source", address=XRESERVE).state == "open"
    w3 = fake_web3(calls={(XRESERVE, sel("paused()")): requests.ConnectionError("rpc down")})
    sleeps = []
    s = read_evm_paused(w3, chain_id="ethereum", side="source", address=XRESERVE, sleep=sleeps.append)
    assert s.state == "unknown" and "after 7 attempts" in s.detail and sleeps == [0, 0, 0.5, 1.0, 2.0, 4.0]


def _hook_tree(pausable: bool) -> dict:
    return {
        (MAILBOX, sel("defaultHook()")): addr(FALLBACK), (MAILBOX, sel("requiredHook()")): addr(FEE),
        (FALLBACK, sel("hookType()")): u8(5), (FALLBACK, sel("hooks(uint32)")): addr(AGG),
        (AGG, sel("hookType()")): u8(2), (AGG, sel("hooks(bytes)")): addrs(PAUSABLE, MERKLE, IGP),
        (PAUSABLE, sel("hookType()")): u8(7), (PAUSABLE, sel("paused()")): boolean(pausable),
        (MERKLE, sel("hookType()")): u8(3), (IGP, sel("hookType()")): u8(4), (FEE, sel("hookType()")): u8(8),
    }


def test_hook_tree_finds_the_pausable_member():
    w3 = fake_web3(calls=_hook_tree(False))
    switches = read_mailbox_hook_pauses(w3, chain_id="ethereum", side="source", mailbox=MAILBOX, far_domain=ALEO_DOMAIN)
    assert [(s.state, s.switch, s.locator) for s in switches] == [("open", "PausableHook.paused()", Web3.to_checksum_address(PAUSABLE))]


def test_hook_tree_pausable_true_is_paused():
    w3 = fake_web3(calls=_hook_tree(True))
    switches = read_mailbox_hook_pauses(w3, chain_id="ethereum", side="source", mailbox=MAILBOX, far_domain=ALEO_DOMAIN)
    assert [s.state for s in switches] == ["paused"]


def test_ism_tree_descends_routing_then_aggregation_and_probes_null():
    calls = {
        (MAILBOX, sel("defaultIsm()")): addr(ISM_AGG),
        (ISM_AGG, sel("moduleType()")): u8(2),
        (ISM_AGG, sel("modulesAndThreshold(bytes)")): encode(["address[]", "uint8"], [[ISM_NULL_PLAIN, ISM_ROUTING], 2]),
        (ISM_NULL_PLAIN, sel("moduleType()")): u8(6), (ISM_NULL_PLAIN, sel("paused()")): "revert",
        (ISM_ROUTING, sel("moduleType()")): u8(1), (ISM_ROUTING, sel("module(uint32)")): addr(ISM_AGG2),
        (ISM_AGG2, sel("moduleType()")): u8(2),
        (ISM_AGG2, sel("modulesAndThreshold(bytes)")): encode(["address[]", "uint8"], [[ISM_PAUSABLE], 1]),
        (ISM_PAUSABLE, sel("moduleType()")): u8(6), (ISM_PAUSABLE, sel("paused()")): boolean(True),
    }
    w3 = fake_web3(calls=calls)
    switches = read_mailbox_ism_pauses(w3, chain_id="ethereum", side="destination", mailbox=MAILBOX, far_domain=ALEO_DOMAIN)
    assert [(s.state, s.switch, s.locator) for s in switches] == [("paused", "PausableIsm.paused()", ISM_PAUSABLE)]


def test_depth_cap_stops_a_cyclic_aggregation():
    calls = {(MAILBOX, sel("defaultHook()")): addr(AGG), (MAILBOX, sel("requiredHook()")): addr(FEE),
             (AGG, sel("hookType()")): u8(2), (AGG, sel("hooks(bytes)")): addrs(AGG), (FEE, sel("hookType()")): u8(8)}
    w3 = fake_web3(calls=calls)
    switches = read_mailbox_hook_pauses(w3, chain_id="ethereum", side="source", mailbox=MAILBOX, far_domain=ALEO_DOMAIN)
    assert [s.state for s in switches] == ["unknown"] and "depth" in switches[0].detail


def test_revert_on_hook_type_is_a_configuration_error():
    calls = {(MAILBOX, sel("defaultHook()")): addr(FALLBACK), (FALLBACK, sel("hookType()")): "revert"}
    w3 = fake_web3(calls=calls)
    with pytest.raises(ConfigurationError, match="reverted"):
        read_mailbox_hook_pauses(w3, chain_id="ethereum", side="source", mailbox=MAILBOX, far_domain=ALEO_DOMAIN)
