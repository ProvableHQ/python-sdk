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


def test_resolves_route_from_asset_refs():
    plan = prepare(DEFAULT_REGISTRY, source="ethereum/usdc", destination="aleo/usdcx",
                   amount="25", recipient=ALEO)
    assert plan.route_id == "xreserve:ethereum/usdc->aleo/usdcx"
    assert plan.protocol == "xreserve" and plan.environment == "mainnet"
    assert plan.registry_version == DEFAULT_REGISTRY.version
    assert (plan.amount, plan.amount_atomic) == ("25", 25_000_000)
    assert plan.mint_mode == "public" and plan.sender is None
    # tuple refs and case-insensitive keys work too
    assert prepare(DEFAULT_REGISTRY, source=("ethereum", "USDC"), destination=("aleo", "usdcx"),
                   amount="25", recipient=ALEO).route_id == plan.route_id


def test_protocol_filter_is_forwarded_and_disambiguates():
    class _Ambiguous:
        version = DEFAULT_REGISTRY.version
        asset = DEFAULT_REGISTRY.asset
        chain = DEFAULT_REGISTRY.chain
        seen = []

        def find_route(self, source, destination, protocol=None):
            self.seen.append(protocol)
            if protocol is None:
                raise AmbiguousRouteError("Multiple bridge routes match; specify protocol")
            return DEFAULT_REGISTRY.find_route(source, destination, protocol)

    reg = _Ambiguous()
    with pytest.raises(AmbiguousRouteError):
        prepare(reg, source="ethereum/usdc", destination="aleo/usdcx", amount="25", recipient=ALEO)
    plan = prepare(reg, source="ethereum/usdc", destination="aleo/usdcx", amount="25",
                   recipient=ALEO, protocol="xreserve")
    assert plan.route_id == "xreserve:ethereum/usdc->aleo/usdcx" and reg.seen == [None, "xreserve"]
    with pytest.raises(RouteNotFoundError):
        prepare(DEFAULT_REGISTRY, source="ethereum/usdc", destination="aleo/usdcx",
                amount="25", recipient=ALEO, protocol="hyperlane")


def test_xreserve_deposit_steps():
    plan = prepare(DEFAULT_REGISTRY, source="ethereum/usdc", destination="aleo/usdcx",
                   amount="25.5", recipient=ALEO)
    assert [s.id for s in plan.steps] == ["source-approval", "source-deposit",
                                          "deposit-attestation", "destination-mint"]
    assert [s.kind for s in plan.steps] == ["approve", "deposit", "wait-attestation", "mint"]
    assert [s.id for s in plan.steps if s.irreversible] == ["source-deposit"]
    assert [s.executor for s in plan.steps] == ["evm-wallet", "evm-wallet", "protocol", "protocol"]


def test_xreserve_burn_steps():
    plan = prepare(DEFAULT_REGISTRY, source="aleo/usdcx", destination="ethereum/usdc",
                   amount="10", recipient=EVM1)
    assert [s.id for s in plan.steps] == ["source-burn", "withdrawal-attestation",
                                          "destination-withdrawal", "destination-confirmation"]
    assert [s.kind for s in plan.steps] == ["burn", "wait-attestation", "withdraw", "confirm-delivery"]
    assert [s.executor for s in plan.steps] == ["aleo-wallet", "protocol", "protocol", "protocol"]
    assert [s.id for s in plan.steps if s.irreversible] == ["source-burn"]


def test_mint_modes():
    record = prepare(DEFAULT_REGISTRY, source="ethereum/usdc", destination="aleo/usdcx",
                     amount="25", recipient=ALEO, mint_mode="record")
    private = prepare(DEFAULT_REGISTRY, source="ethereum/usdc", destination="aleo/usdcx",
                      amount="25", recipient=ALEO, mint_mode="private")
    assert record.mint_mode == "record" and record.steps[-1].executor == "protocol"
    assert private.mint_mode == "private" and private.steps[-1].executor == "aleo-wallet"
    with pytest.raises(ConfigurationError, match="only valid.*Aleo"):
        prepare(DEFAULT_REGISTRY, source="aleo/usdcx", destination="ethereum/usdc",
                amount="10", recipient=EVM1, mint_mode="private")
    with pytest.raises(ConfigurationError, match="xReserve"):
        prepare(DEFAULT_REGISTRY, source="ethereum/wbtc", destination="aleo/wbtc",
                amount="0.1", recipient=ALEO, mint_mode="record")
    with pytest.raises(ConfigurationError, match="mint_mode"):
        prepare(DEFAULT_REGISTRY, source="ethereum/usdc", destination="aleo/usdcx",
                amount="25", recipient=ALEO, mint_mode="secret")


def test_hyperlane_steps_approval_only_on_non_aleo_token_sources():
    inbound = prepare(DEFAULT_REGISTRY, source="ethereum/wbtc", destination="aleo/wbtc",
                      amount="0.1", recipient=ALEO)
    assert [s.id for s in inbound.steps] == ["source-approval", "source-dispatch",
                                             "message-delivery", "destination-confirmation"]
    assert [s.kind for s in inbound.steps] == ["approve", "dispatch", "wait-delivery", "confirm-delivery"]
    assert inbound.steps[0].executor == "evm-wallet" and inbound.steps[-1].executor == "protocol"
    assert [s.id for s in inbound.steps if s.irreversible] == ["source-dispatch"]

    native = prepare(DEFAULT_REGISTRY, source="ethereum/eth", destination="aleo/eth",
                     amount="0.000000000000000001", recipient=ALEO)
    assert [s.id for s in native.steps] == ["source-dispatch", "message-delivery", "destination-confirmation"]
    assert native.amount_atomic == 1

    outbound = prepare(DEFAULT_REGISTRY, source="aleo/wbtc", destination="ethereum/wbtc",
                       amount="0.1", recipient=EVM1)
    assert [s.id for s in outbound.steps] == ["source-dispatch", "message-delivery", "destination-confirmation"]
    assert outbound.steps[0].executor == "aleo-wallet"

    sol = prepare(DEFAULT_REGISTRY, source="solana/sol", destination="aleo/sol",
                  amount="0.000000001", recipient=ALEO, sender="11111111111111111111111111111111")
    assert [s.id for s in sol.steps] == ["source-dispatch", "message-delivery", "destination-confirmation"]
    assert sol.steps[0].executor == "solana-wallet" and sol.sender == "11111111111111111111111111111111"


def test_amount_forms_and_precision():
    by_atomic = prepare(DEFAULT_REGISTRY, source="ethereum/wbtc", destination="aleo/wbtc",
                        amount_atomic=100_000, recipient=ALEO)
    assert (by_atomic.amount, by_atomic.amount_atomic) == ("0.001", 100_000)
    with pytest.raises(InvalidAmountError, match="greater than zero"):
        prepare(DEFAULT_REGISTRY, source="ethereum/usdc", destination="aleo/usdcx",
                amount="0", recipient=ALEO)
    with pytest.raises(InvalidAmountError):                       # 7 fractional digits on a 6-decimal asset
        prepare(DEFAULT_REGISTRY, source="ethereum/usdc", destination="aleo/usdcx",
                amount="0.0000001", recipient=ALEO)
    with pytest.raises(InvalidAmountError):                       # exactly one of amount / amount_atomic
        prepare(DEFAULT_REGISTRY, source="ethereum/usdc", destination="aleo/usdcx",
                amount="1", amount_atomic=1_000_000, recipient=ALEO)
    with pytest.raises(InvalidAmountError):
        prepare(DEFAULT_REGISTRY, source="ethereum/usdc", destination="aleo/usdcx", recipient=ALEO)


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
        prepare(_Reg(), source="ethereum/usdc", destination="aleo/usdcx", amount="1.001", recipient=ALEO)
    assert prepare(_Reg(), source="ethereum/usdc", destination="aleo/usdcx", amount="1.5",
                   recipient=ALEO).amount_atomic == 1_500_000


def test_recipient_regex():
    with pytest.raises(InvalidRecipientError, match="aleo address format"):
        prepare(DEFAULT_REGISTRY, source="ethereum/usdc", destination="aleo/usdcx",
                amount="1", recipient="not-an-aleo-address")
    with pytest.raises(InvalidRecipientError, match="ethereum address format"):
        prepare(DEFAULT_REGISTRY, source="aleo/wbtc", destination="ethereum/wbtc",
                amount="0.1", recipient=ALEO)


def test_metadata_required_routes_still_plan():
    # veil's prepare only excludes *disabled* routes; quote/execute refuse metadata-required ones.
    plan = prepare(DEFAULT_REGISTRY, source="aleo/aleo", destination="ethereum/aleo",
                   amount="1", recipient=EVM1)
    assert plan.route_id == "hyperlane:aleo/aleo->ethereum/aleo"


def test_plan_roundtrip_and_resolve_route():
    plan = prepare(DEFAULT_REGISTRY, source="ethereum/usdc", destination="aleo/usdcx",
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
        prepared = prepare(DEFAULT_REGISTRY, source=route.source_asset_id,
                           destination=route.destination_asset_id, amount_atomic=1,
                           recipient=recipient, protocol=route.protocol)
        expected = build_plan(DEFAULT_REGISTRY, route, amount_atomic=1, recipient=recipient, sender=None)
        assert prepared == expected, route.id
