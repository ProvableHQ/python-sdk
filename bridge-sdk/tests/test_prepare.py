import pytest

from aleo_bridge._plan import build_plan
from aleo_bridge.errors import (AmbiguousRouteError, ConfigurationError, InvalidAmountError,
                                InvalidRecipientError, RouteNotFoundError,
                                RegistryVersionMismatchError, CheckpointInvalidError)
from aleo_bridge.lifecycle import prepare, resolve_route
from aleo_bridge.registry import DEFAULT_REGISTRY
from aleo_bridge.types import Plan

ALEO = "aleo1" + "a" * 58
EVM1 = "0x0000000000000000000000000000000000000001"
SOLANA1 = "11111111111111111111111111111111"


def test_resolves_route_from_chain_and_asset_keywords():
    plan = prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc",
                   destination_chain="aleo", destination_asset="usdcx", amount="25", recipient=ALEO)
    assert plan.route_id == "xreserve:ethereum/usdc->aleo/usdcx"
    assert plan.protocol == "xreserve" and plan.environment == "mainnet"
    assert plan.registry_version == DEFAULT_REGISTRY.version
    assert (plan.amount, plan.amount_atomic) == ("25", 25_000_000)
    assert plan.mint_mode == "public" and plan.sender is None
    # case-insensitive, and destination_asset is inferred when the pair leaves one route
    assert prepare(DEFAULT_REGISTRY, source_chain="Ethereum", source_asset="USDC", destination_chain="aleo",
                   amount="25", recipient=ALEO).route_id == plan.route_id
    with pytest.raises(TypeError):
        prepare(DEFAULT_REGISTRY, "ethereum/usdc", "aleo/usdcx", amount="25", recipient=ALEO)   # keyword-only


def test_route_keyword_takes_a_route_or_its_id():
    route = DEFAULT_REGISTRY.route("xreserve:ethereum/usdc->aleo/usdcx")
    by_obj = prepare(DEFAULT_REGISTRY, route=route, amount="25", recipient=ALEO)
    by_id = prepare(DEFAULT_REGISTRY, route=route.id, amount="25", recipient=ALEO)
    by_filters = prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc",
                         destination_chain="aleo", amount="25", recipient=ALEO)
    assert by_obj == by_id == by_filters
    with pytest.raises(RouteNotFoundError):
        prepare(DEFAULT_REGISTRY, route="hyperlane:aleo/doge->ethereum/doge", amount="1", recipient=ALEO)
    # route= and the chain/asset filters are two ways to say the same thing — never both
    with pytest.raises(ConfigurationError, match="route="):
        prepare(DEFAULT_REGISTRY, route=route, source_chain="ethereum", amount="25", recipient=ALEO)


def test_missing_selectors_name_what_is_needed():
    with pytest.raises(ConfigurationError, match="destination_chain"):
        prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", amount="25", recipient=ALEO)
    with pytest.raises(ConfigurationError, match="source_asset"):
        prepare(DEFAULT_REGISTRY, source_chain="ethereum", destination_chain="aleo", amount="25", recipient=ALEO)
    with pytest.raises(ConfigurationError, match="route=.*bridge_protocol="):
        prepare(DEFAULT_REGISTRY, amount="25", recipient=ALEO)


def test_protocol_filter_disambiguates():
    from aleo_bridge.registry import Registry, Route
    dup = Route("xreserve:ethereum/wbtc->aleo/wbtc", "xreserve", "mainnet", "ethereum/wbtc", "aleo/wbtc",
                "active", None, None, {})
    reg = Registry(DEFAULT_REGISTRY.version, DEFAULT_REGISTRY.chains(), DEFAULT_REGISTRY.assets(),
                   [*DEFAULT_REGISTRY.routes(include_unavailable=True), dup])
    with pytest.raises(AmbiguousRouteError):
        prepare(reg, source_chain="ethereum", source_asset="wbtc", destination_chain="aleo", amount="1", recipient=ALEO)
    plan = prepare(reg, source_chain="ethereum", source_asset="wbtc", destination_chain="aleo", amount="1",
                   recipient=ALEO, bridge_protocol="hyperlane")
    assert plan.route_id == "hyperlane:ethereum/wbtc->aleo/wbtc"
    with pytest.raises(RouteNotFoundError):
        prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo",
                amount="25", recipient=ALEO, bridge_protocol="hyperlane")
    with pytest.raises(TypeError):                                   # veil's quote key is bridgeProtocol
        prepare(reg, source_chain="ethereum", source_asset="wbtc", destination_chain="aleo", amount="1",
                recipient=ALEO, protocol="hyperlane")


def test_xreserve_deposit_steps():
    plan = prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx",
                   amount="25.5", recipient=ALEO)
    assert [s.id for s in plan.steps] == ["source-approval", "source-deposit",
                                          "deposit-attestation", "destination-mint"]
    assert [s.kind for s in plan.steps] == ["approve", "deposit", "wait-attestation", "mint"]
    assert [s.id for s in plan.steps if s.irreversible] == ["source-deposit"]
    assert [s.executor for s in plan.steps] == ["evm-wallet", "evm-wallet", "protocol", "protocol"]


def test_xreserve_burn_steps():
    plan = prepare(DEFAULT_REGISTRY, source_chain="aleo", source_asset="usdcx", destination_chain="ethereum", destination_asset="usdc",
                   amount="10", recipient=EVM1)
    assert [s.id for s in plan.steps] == ["source-burn", "withdrawal-attestation",
                                          "destination-withdrawal", "destination-confirmation"]
    assert [s.kind for s in plan.steps] == ["burn", "wait-attestation", "withdraw", "confirm-delivery"]
    assert [s.executor for s in plan.steps] == ["aleo-wallet", "protocol", "protocol", "protocol"]
    assert [s.id for s in plan.steps if s.irreversible] == ["source-burn"]


def test_mint_modes():
    record = prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx",
                     amount="25", recipient=ALEO, mint_mode="record")
    private = prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx",
                      amount="25", recipient=ALEO, mint_mode="private")
    assert record.mint_mode == "record" and record.steps[-1].executor == "protocol"
    assert private.mint_mode == "private" and private.steps[-1].executor == "aleo-wallet"
    with pytest.raises(ConfigurationError, match="only valid.*Aleo"):
        prepare(DEFAULT_REGISTRY, source_chain="aleo", source_asset="usdcx", destination_chain="ethereum", destination_asset="usdc",
                amount="10", recipient=EVM1, mint_mode="private")
    with pytest.raises(ConfigurationError, match="xReserve"):
        prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="wbtc", destination_chain="aleo", destination_asset="wbtc",
                amount="0.1", recipient=ALEO, mint_mode="record")
    with pytest.raises(ConfigurationError, match="mint_mode"):
        prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx",
                amount="25", recipient=ALEO, mint_mode="secret")


def test_hyperlane_steps_approval_only_on_non_aleo_token_sources():
    inbound = prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="wbtc", destination_chain="aleo", destination_asset="wbtc",
                      amount="0.1", recipient=ALEO)
    assert [s.id for s in inbound.steps] == ["source-approval", "source-dispatch",
                                             "message-delivery", "destination-confirmation"]
    assert [s.kind for s in inbound.steps] == ["approve", "dispatch", "wait-delivery", "confirm-delivery"]
    assert inbound.steps[0].executor == "evm-wallet" and inbound.steps[-1].executor == "protocol"
    assert [s.id for s in inbound.steps if s.irreversible] == ["source-dispatch"]

    native = prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="eth", destination_chain="aleo", destination_asset="eth",
                     amount="0.000000000000000001", recipient=ALEO)
    assert [s.id for s in native.steps] == ["source-dispatch", "message-delivery", "destination-confirmation"]
    assert native.amount_atomic == 1

    outbound = prepare(DEFAULT_REGISTRY, source_chain="aleo", source_asset="wbtc", destination_chain="ethereum", destination_asset="wbtc",
                       amount="0.1", recipient=EVM1)
    assert [s.id for s in outbound.steps] == ["source-dispatch", "message-delivery", "destination-confirmation"]
    assert outbound.steps[0].executor == "aleo-wallet"

    sol = prepare(DEFAULT_REGISTRY, source_chain="solana", source_asset="sol", destination_chain="aleo", destination_asset="sol",
                  amount="0.000000001", recipient=ALEO, sender="11111111111111111111111111111111")
    assert [s.id for s in sol.steps] == ["source-dispatch", "message-delivery", "destination-confirmation"]
    assert sol.steps[0].executor == "solana-wallet" and sol.sender == "11111111111111111111111111111111"


def test_amount_forms_and_precision():
    by_atomic = prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="wbtc", destination_chain="aleo", destination_asset="wbtc",
                        amount_atomic=100_000, recipient=ALEO)
    assert (by_atomic.amount, by_atomic.amount_atomic) == ("0.001", 100_000)
    with pytest.raises(InvalidAmountError, match="greater than zero"):
        prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx",
                amount="0", recipient=ALEO)
    with pytest.raises(InvalidAmountError):                       # 7 fractional digits on a 6-decimal asset
        prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx",
                amount="0.0000001", recipient=ALEO)
    with pytest.raises(InvalidAmountError):                       # exactly one of amount / amount_atomic
        prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx",
                amount="1", amount_atomic=1_000_000, recipient=ALEO)
    with pytest.raises(InvalidAmountError):
        prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx", recipient=ALEO)


def test_destination_decimals_are_checked_too():
    class _Reg:
        version = DEFAULT_REGISTRY.version
        chain = DEFAULT_REGISTRY.chain
        find_route = DEFAULT_REGISTRY.find_route

        def asset(self, ref):
            asset = DEFAULT_REGISTRY.asset(ref)
            if asset.id == "aleo/usdcx":
                import dataclasses
                return dataclasses.replace(asset, decimals=2)   # coarser destination
            return asset

    with pytest.raises(InvalidAmountError):
        prepare(_Reg(), source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx", amount="1.001", recipient=ALEO)
    assert prepare(_Reg(), source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx", amount="1.5",
                   recipient=ALEO).amount_atomic == 1_500_000


def test_recipient_regex():
    with pytest.raises(InvalidRecipientError, match="aleo address format"):
        prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx",
                amount="1", recipient="not-an-aleo-address")
    with pytest.raises(InvalidRecipientError, match="ethereum address format"):
        prepare(DEFAULT_REGISTRY, source_chain="aleo", source_asset="wbtc", destination_chain="ethereum", destination_asset="wbtc",
                amount="0.1", recipient=ALEO)


def test_metadata_required_routes_still_plan():
    # veil's prepare only excludes *disabled* routes; quote/execute refuse metadata-required ones.
    plan = prepare(DEFAULT_REGISTRY, source_chain="aleo", source_asset="aleo", destination_chain="ethereum", destination_asset="aleo",
                   amount="1", recipient=EVM1)
    assert plan.route_id == "hyperlane:aleo/aleo->ethereum/aleo"


def test_plan_roundtrip_and_resolve_route():
    plan = prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx",
                   amount="25", recipient=ALEO, mint_mode="private", sender=EVM1)
    again = Plan.from_dict(plan.to_dict())
    assert again == plan
    resolved = resolve_route(DEFAULT_REGISTRY, plan)
    assert (resolved.route.id, resolved.source_chain.family, resolved.destination_chain.family) == (
        plan.route_id, "evm", "aleo")
    import dataclasses
    with pytest.raises(RegistryVersionMismatchError):
        resolve_route(DEFAULT_REGISTRY, dataclasses.replace(plan, registry_version="old"))
    with pytest.raises(CheckpointInvalidError):
        resolve_route(DEFAULT_REGISTRY, dataclasses.replace(plan, protocol="hyperlane"))


def _recipient_for(chain_family: str) -> str:
    return {"aleo": ALEO, "evm": EVM1, "solana": SOLANA1}[chain_family]


def test_prepare_equals_build_plan_for_every_active_route():
    # Controller ruling (task-1-controller-notes.md #4): prepare() is only ever a thin
    # validating wrapper around the shared _plan.build_plan — for every active route this
    # must hold field-by-field, with amount_atomic=1 and a recipient valid for the destination.
    for route in DEFAULT_REGISTRY.routes(include_unavailable=True):
        if not route.active:
            continue
        destination = DEFAULT_REGISTRY.asset(route.destination_asset_id)
        destination_chain = DEFAULT_REGISTRY.chain(destination.chain_id)
        recipient = _recipient_for(destination_chain.family)
        prepared = prepare(DEFAULT_REGISTRY, route=route, amount_atomic=1, recipient=recipient)
        expected = build_plan(DEFAULT_REGISTRY, route, amount_atomic=1, recipient=recipient, sender=None)
        assert prepared == expected, route.id
