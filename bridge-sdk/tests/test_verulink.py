"""Public Verulink routes and fee/peer validation (Veil PR 217)."""
import pytest

from aleo_bridge import BridgeError, DEFAULT_REGISTRY, PauseCheckError, RoutePausedError, prepare

ALEO = "aleo19lu7tcg5v3c7ke5gn98h0v7crsn4jcct4uck0u0q9ewuhtc0hc9s0rygds"
EVM = "0x78b4873616922c84159188e9f7098b6d57d777c3"


@pytest.mark.parametrize("source,destination,recipient,executor", [
    ("aleo", "bsc", EVM, "evm-wallet"), ("bsc", "aleo", ALEO, "aleo-wallet"),
])
def test_public_routes_require_destination_claim(source, destination, recipient, executor):
    plan = prepare(DEFAULT_REGISTRY, source_chain=source, source_asset="aleo",
                   destination_chain=destination, amount="20", recipient=recipient)
    assert plan.protocol == "verulink"
    assert plan.amount_atomic == 20_000_000
    assert [s.executor for s in plan.steps if s.id == "destination-claim"] == [executor]
    assert sum(s.irreversible for s in plan.steps) == 1


@pytest.mark.parametrize("mode", ["record", "private"])
def test_no_private_verulink_delivery(mode):
    with pytest.raises(BridgeError, match="xReserve"):
        prepare(DEFAULT_REGISTRY, source_chain="bsc", source_asset="aleo", destination_chain="aleo",
                amount="20", recipient=ALEO, mint_mode=mode)


@pytest.mark.parametrize("outbound", [True, False])
def test_live_fee_truncation_and_net_limits(outbound):
    from tests.fakes.fake_verulink import Harness
    h = Harness(outbound)
    q = h.quote()
    assert q.amount_out == "10.989001"
    assert q.fee_atomic == 11_000
    assert q.rate == 100
    assert q.plan.amount_out == q.amount_out
    assert len([f for f in q.fees if f.kind == "network" and f.amount is None]) == 2
    assert not h.sent and not h.aleo.submitted
    with pytest.raises(BridgeError, match="minimum"):
        h.quote("10")
    assert h.quote("10.01001").amount_out == "10"


def test_quote_rejects_wrong_chain_peers_pause_and_overflow():
    from tests.fakes.fake_verulink import Harness
    h = Harness()
    h.rpc.chain_id = 1
    with pytest.raises(BridgeError, match="chain 56"):
        h.quote()
    h.rpc.chain_id = 56
    h.rpc.values["paused"] = True
    with pytest.raises(BridgeError, match="paused"):
        h.quote()
    h.rpc.values["paused"] = False
    with pytest.raises(BridgeError, match="u64"):
        h.quote("184467440737.095517")
    h.rpc.values["erc20Bridge"] = "0x" + "11" * 20
    with pytest.raises(BridgeError, match="peers"):
        h.quote()


@pytest.mark.parametrize("value", [None, "true", "0u8"])
def test_missing_or_malformed_aleo_active_flag_fails_closed(value):
    from tests.fakes.fake_verulink import Harness, M
    h = Harness()
    h.aleo.mappings[M["aleoBridge"]]["bridge_settings"]["3u8"] = value
    with pytest.raises(RoutePausedError if value == "0u8" else PauseCheckError):
        h.quote()
