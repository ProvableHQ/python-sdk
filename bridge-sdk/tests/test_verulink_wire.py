"""Known-answer packets/signatures from Veil PR 217 at a2071c0b.

Fixture source: packages/bridge/test/fixtures/verulink-mainnet.json, copied verbatim.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aleo_bridge import BridgeError

FIXTURE = json.loads((Path(__file__).parent / "fixtures/verulink-mainnet.json").read_text())


def packet(outbound):
    from aleo_bridge._verulink_wire import collector_packet
    return collector_packet(FIXTURE["toBsc" if outbound else "toAleo"])


@pytest.mark.parametrize("outbound", [True, False])
def test_real_attestor_signatures(outbound):
    from aleo_bridge._verulink_attestation import verified_signatures
    signatures = FIXTURE["toBsc" if outbound else "toAleo"]["status"]["signatures"]
    c = SimpleNamespace(outbound=outbound, aleo_bridge="bridge.aleo", evm_bridge="0x" + "01" * 20,
                        network="mainnet", read=lambda *args: 3 if args[1] == "quorumRequired" else True,
                        mapping=lambda *args: 3 if args[1] == "bridge_settings" else False)
    # Aleo's membership uses contains: even a present false entry is a member.
    assert len(verified_signatures(c, packet(outbound), signatures)) == 3
    tampered = {**packet(outbound), "amount": "10000001"}
    with pytest.raises(BridgeError, match="signature"):
        verified_signatures(c, tampered, signatures)


@pytest.mark.parametrize("outbound", [True, False])
def test_duplicate_unknown_and_zero_quorum_rejected(outbound):
    from aleo_bridge._verulink_attestation import verified_signatures
    signatures = copy.deepcopy(FIXTURE["toBsc" if outbound else "toAleo"]["status"]["signatures"])
    c = SimpleNamespace(outbound=outbound, aleo_bridge="bridge.aleo", evm_bridge="0x" + "01" * 20,
                        network="mainnet", read=lambda *args: 3 if args[1] == "quorumRequired" else True,
                        mapping=lambda *args: 3 if args[1] == "bridge_settings" else True)
    with pytest.raises(BridgeError, match="Duplicate"):
        verified_signatures(c, packet(outbound), [signatures[0]] * 3)
    c.read = lambda *args: 3 if args[1] == "quorumRequired" else False
    c.mapping = lambda *args: 3 if args[1] == "bridge_settings" else None
    with pytest.raises(BridgeError, match="Unknown"):
        verified_signatures(c, packet(outbound), signatures)
    c.read = lambda *args: 0
    c.mapping = lambda *args: 0
    with pytest.raises(BridgeError, match="quorum"):
        verified_signatures(c, packet(outbound), [])


@pytest.mark.parametrize("value", [True, -1, 1.0, "1e2", "-1", str(2**64)])
def test_wire_integers_reject_lossy_or_out_of_range_values(value):
    from aleo_bridge._verulink_wire import decimal
    with pytest.raises(BridgeError):
        decimal(value)


def test_evm_calldata_decodes_historical_transfer():
    from web3 import Web3
    from aleo_bridge._verulink_abi import ABI
    function, args = Web3().eth.contract(abi=ABI).decode_function_input(FIXTURE["bscSource"]["tx"]["input"])
    assert function.fn_name == "publicTokenSend"
    assert args["tokenAddress"].lower() == "0x6cfff a5bfd4277a04d83307feedfe2d18d944dd2".replace(" ", "")
    assert args["receiver"] == FIXTURE["toAleo"]["message"]["receiver"]


def test_old_registry_checkpoints_keep_only_preexisting_routes():
    from aleo_bridge import DEFAULT_REGISTRY
    from aleo_bridge._registry_compatibility import is_registry_version_compatible
    for route in DEFAULT_REGISTRY.routes():
        assert is_registry_version_compatible(DEFAULT_REGISTRY, "2026-09-30.hyperlane-bat-usdg-zec.1", route.id) is (route.protocol != "verulink")


@pytest.mark.parametrize("outbound", [True, False])
def test_reconstructs_historical_source_packet(outbound):
    from tests.fakes.fake_verulink import Harness
    from aleo_bridge._verulink_source import source_packet
    from aleo_bridge._verulink_wire import same_packet
    h = Harness(outbound)
    tx_id = FIXTURE["aleoSource"]["transaction"]["id"] if outbound else FIXTURE["bscSource"]["tx"]["hash"]
    assert same_packet(source_packet(h.bridge.verulink.context(h.plan), h.plan, tx_id), packet(outbound))


def test_unrelated_bridge_event_does_not_hide_source_packet():
    from tests.fakes.fake_verulink import Harness, M
    from aleo_bridge._verulink_source import source_packet
    from aleo_bridge._verulink_wire import same_packet
    h = Harness(False)
    tx_id = FIXTURE["bscSource"]["tx"]["hash"]
    log = copy.deepcopy(h.rpc.receipts[tx_id]["logs"][0])
    log.update(address=M["evmBridge"], topics=["0x" + "ab" * 32])
    h.rpc.receipts[tx_id]["logs"].append(log)
    assert same_packet(source_packet(h.bridge.verulink.context(h.plan), h.plan, tx_id), packet(False))


def test_negative_and_mixed_screening_votes_never_authorize_claim():
    from eth_account import Account
    from eth_account.messages import encode_defunct
    from eth_utils.crypto import keccak
    from aleo_bridge._verulink_attestation import verified_signatures
    from aleo_bridge._verulink_wire import packet_hash
    c = SimpleNamespace(outbound=True, evm_bridge="0x" + "01" * 20,
                        read=lambda *args: 3 if args[1] == "quorumRequired" else True)
    p = packet(True)

    def signed(votes):
        result = []
        for i, vote in enumerate(votes):
            account = Account.from_key(bytes([i + 1]) * 32)
            sign = account.sign_message(encode_defunct(keccak(packet_hash(p) + bytes([vote]))))
            result.append({"signer": account.address, "sign": "0x" + sign.signature.hex()})
        return result

    with pytest.raises(BridgeError, match="negatively"):
        verified_signatures(c, p, signed([2, 2, 2]))
    assert verified_signatures(c, p, signed([1, 1, 2])) is None


def test_collector_pagination_and_packet_substitution():
    from aleo_bridge._verulink_attestation import fetch_signatures
    c = SimpleNamespace(collector_url="https://collector.invalid/v2")
    calls = []
    row = copy.deepcopy(FIXTURE["toBsc"])

    def get(url, **kwargs):
        calls.append(url)
        page = int(url.rsplit("=", 1)[1])
        return SimpleNamespace(status_code=200, json=lambda: {
            "data": {"data": [] if page == 1 else [row], "currentPage": page, "totalPages": 2}})

    session = SimpleNamespace(get=get)
    assert fetch_signatures(c, session, packet(True)) == row["status"]["signatures"]
    assert len(calls) == 2
    row["message"]["amount"] = "1"
    with pytest.raises(BridgeError, match="source data"):
        fetch_signatures(c, session, packet(True))


def test_source_validation_rejects_changed_inputs_and_ambiguous_packets():
    from tests.fakes.fake_verulink import Harness, M
    from aleo_bridge._verulink_source import source_packet
    h = Harness()
    tx_id = FIXTURE["aleoSource"]["transaction"]["id"]
    c = h.bridge.verulink.context(h.plan)
    root = h.aleo.confirmed_transactions[tx_id]["transaction"]["execution"]["transitions"][-1]
    root["inputs"][1]["value"] = "20000000u64"
    with pytest.raises(BridgeError, match="inputs"):
        source_packet(c, h.plan, tx_id)
    root["inputs"][1]["value"] = "10000000u64"
    maps = h.aleo.mappings[M["aleoBridge"]]
    maps["sequences"][M["evmDomain"] + "u128"] = "3u64"
    maps["out_packets"][f"{{ chain_id: {M['evmDomain']}u128, sequence: 2u64 }}"] = FIXTURE["aleoPacket"].replace("sequence: 1u64", "sequence: 2u64")
    with pytest.raises(BridgeError, match="Ambiguous"):
        source_packet(c, h.plan, tx_id)
