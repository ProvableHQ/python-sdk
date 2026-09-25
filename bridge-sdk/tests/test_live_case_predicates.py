"""Hermetic checks of the live-case route predicates (no network)."""
from aleo_bridge.registry import DEFAULT_REGISTRY
from tests.live import cases as live_cases


def test_aleo_origin_legs_complete_without_a_destination_id():
    """Aleo-origin legs (both protocols) prove delivery by a balance rise and record no message or
    destination id; every other leg must carry one of them when it completes (the 2026-09-22
    mainnet aleo/eth->ethereum/eth leg completed with +1 wei and no ids, as veil's own
    aleo-hyperlane test expects)."""
    no_id = {r.id for r in DEFAULT_REGISTRY.routes(environment="mainnet")
             if live_cases.completion_has_no_destination_id(r, DEFAULT_REGISTRY)}
    assert "hyperlane:aleo/eth->ethereum/eth" in no_id
    assert "hyperlane:aleo/sol->solana/sol" in no_id
    assert "xreserve:aleo/usdcx->ethereum/usdc" in no_id
    assert "hyperlane:ethereum/eth->aleo/eth" not in no_id
    assert "hyperlane:solana/sol->aleo/sol" not in no_id
    assert "xreserve:ethereum/usdc->aleo/usdcx" not in no_id


def test_only_the_xreserve_withdrawal_skips_the_wait_loop():
    """The drive-loop predicate stays narrower than the no-id predicate: an Aleo-origin Hyperlane
    leg is still driven through ``wait`` (the SDK terminates it from the balance rise)."""
    registry = DEFAULT_REGISTRY
    assert live_cases.delivery_is_a_balance_rise(registry.route("xreserve:aleo/usdcx->ethereum/usdc"), registry)
    assert not live_cases.delivery_is_a_balance_rise(registry.route("hyperlane:aleo/eth->ethereum/eth"), registry)
    assert not live_cases.delivery_is_a_balance_rise(registry.route("hyperlane:ethereum/eth->aleo/eth"), registry)
