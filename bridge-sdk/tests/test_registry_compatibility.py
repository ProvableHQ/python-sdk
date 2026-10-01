"""Legacy deployment fingerprints must survive additions, but never deployment changes."""
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from aleo_bridge.registry import DEFAULT_REGISTRY, Registry, Route, validate_registry
from aleo_bridge.errors import ConfigurationError

LEGACY = json.loads((Path(__file__).parent / 'fixtures/registry-2026-08-31-python.json').read_text())


@pytest.mark.parametrize('route_id', [r['id'] for r in LEGACY['routes']])
def test_legacy_routes_remain_compatible(route_id):
    from aleo_bridge._registry_compatibility import is_registry_version_compatible
    assert is_registry_version_compatible(DEFAULT_REGISTRY, LEGACY['version'], route_id)


@pytest.mark.parametrize('route_id', [r['id'] for r in LEGACY['routes']])
def test_legacy_label_cannot_hide_changed_deployment(route_id):
    from aleo_bridge._registry_compatibility import is_registry_version_compatible
    r = DEFAULT_REGISTRY
    changed = Registry(r.version, r.chains(), r.assets(), [
        replace(v, metadata={**v.metadata, 'changedDeployment': 'unreviewed'}) if v.id == route_id else v
        for v in r.routes(include_unavailable=True)])
    assert not is_registry_version_compatible(changed, LEGACY['version'], route_id)


@pytest.mark.parametrize('field,value', [('decimals', 18), ('availability', 'disabled'), ('domain', 99)])
def test_legacy_route_rejects_asset_chain_or_availability_change(field, value):
    from aleo_bridge._registry_compatibility import is_registry_version_compatible
    r = DEFAULT_REGISTRY
    route_id = 'xreserve:ethereum/usdc->aleo/usdcx'
    assets = [replace(a, decimals=value) if field == 'decimals' and a.id == 'ethereum/usdc' else a for a in r.assets()]
    routes = [replace(v, availability=value) if field == 'availability' and v.id == route_id else v
              for v in r.routes(include_unavailable=True)]
    chains = [replace(c, protocol_domains={**c.protocol_domains, 'xreserve': value})
              if field == 'domain' and c.id == 'ethereum' else c for c in r.chains()]
    assert not is_registry_version_compatible(Registry(r.version, chains, assets, routes), LEGACY['version'], route_id)


def test_unknown_versions_and_new_routes_cannot_use_legacy_label():
    from aleo_bridge._registry_compatibility import is_registry_version_compatible
    assert not is_registry_version_compatible(DEFAULT_REGISTRY, 'unknown', LEGACY['routes'][0]['id'])
    assert not is_registry_version_compatible(DEFAULT_REGISTRY, LEGACY['version'], 'cctp:ethereum/usdc->arc/usdc')


@pytest.mark.parametrize('chain,domain,chain_id', [('ethereum', 0, 1), ('base', 6, 8453), ('arbitrum', 3, 42161)])
def test_cctp_routes_bind_both_chain_domains(chain, domain, chain_id):
    r = DEFAULT_REGISTRY
    inbound = r.find_route(source_chain=chain, source_asset='usdc', destination_chain='arc')
    outbound = r.find_route(source_chain='arc', source_asset='usdc', destination_chain=chain)
    assert inbound.protocol == outbound.protocol == 'cctp'
    assert (inbound.meta_int('sourceDomain'), inbound.meta_int('destinationDomain')) == (domain, 26)
    assert (outbound.meta_int('sourceChainId'), outbound.meta_int('destinationChainId')) == (5042, chain_id)
    assert r.asset('arc/usdc').decimals == 6
    assert r.asset('arc/usdc').locator.value == '0x3600000000000000000000000000000000000000'


@pytest.mark.parametrize('domain', [None, True, 99])
def test_cctp_registry_refuses_missing_bool_or_mismatched_domain(domain):
    r = DEFAULT_REGISTRY
    chains = [replace(c, protocol_domains={'cctp': domain, 'xreserve': 26}) if c.id == 'arc' else c
              for c in r.chains()]
    with pytest.raises(ConfigurationError, match='domain'):
        validate_registry(Registry(r.version, chains, r.assets(), r.routes(include_unavailable=True)))


def test_legacy_prepared_checkpoint_recovers_without_network_or_signing():
    from aleo_bridge.lifecycle import recover
    from tests.test_recover import _aleo_eth_checkpoint
    from tests.fakes.fake_bridge import FakeBridge
    b = FakeBridge(ethereum=False)
    serialized = json.dumps({'type':'execute','id':'at1prepared','fee':{}})
    _, cp = _aleo_eth_checkpoint(b,preparedTransaction={'transactionId':'at1prepared','serializedTransaction':serialized})
    cp['route']['registryVersion'] = LEGACY['version']
    result = recover(b,cp)
    assert result.next == 'resume'
    assert result.receipt.protocol_state['preparedTransaction'] == serialized
    assert b.calls == [] and b.events == []


def test_legacy_plan_still_tracks_terminal_delivery_without_provider_reads():
    from aleo_bridge.lifecycle import get_status, prepare
    from aleo_bridge.types import Receipt, Status
    from tests.fakes.fake_bridge import FakeBridge, EVM_ADDRESS
    b = FakeBridge()
    plan = prepare(b.registry,source_chain='aleo',source_asset='eth',destination_chain='ethereum',
                   amount='0.1',recipient=EVM_ADDRESS)
    plan = replace(plan,registry_version=LEGACY['version'])
    receipt = Receipt('old','hyperlane',Status.COMPLETED,protocol_state={'routeId':plan.route_id})
    assert get_status(b,plan,receipt) is receipt
    assert b.calls == []


# Captured from the Python registry at master dba135b (the snapshot immediately preceding the BAT/USDG/ZEC
# rollout, veil PR #169). Never regenerate it from current data.
PREVIOUS = json.loads((Path(__file__).parent / 'fixtures/registry-2026-09-28-python.json').read_text())


def _registry_from_snapshot(snapshot: dict) -> Registry:
    from aleo_bridge.registry import Asset, Chain, Locator, Privacy
    chains = [Chain(**c) for c in snapshot['chains']]
    assets = [Asset(**{**a, 'locator': Locator(**a['locator']) if a['locator'] else None,
                       'privacy': Privacy(**a['privacy']) if a['privacy'] else None}) for a in snapshot['assets']]
    routes = [Route(**r) for r in snapshot['routes']]
    return validate_registry(Registry(snapshot['version'], chains, assets, routes))


def test_previous_snapshot_fixture_is_the_registry_the_hashes_were_taken_from():
    from aleo_bridge._registry_compatibility import PREVIOUS_ROUTE_HASHES, PREVIOUS_VERSION, route_fingerprint
    previous = _registry_from_snapshot(PREVIOUS)
    assert previous.version == PREVIOUS_VERSION == '2026-09-28.cctp-arc.1'
    assert {r.id for r in previous.routes(include_unavailable=True)} == set(PREVIOUS_ROUTE_HASHES)
    assert all(route_fingerprint(previous, rid) == digest for rid, digest in PREVIOUS_ROUTE_HASHES.items())
    assert not any(re.search(r'/(bat|usdg|zec)(?:->|$)', rid) for rid in PREVIOUS_ROUTE_HASHES)


@pytest.mark.parametrize('route_id', [r['id'] for r in PREVIOUS['routes']])
def test_every_previous_route_remains_compatible_with_the_current_registry(route_id):
    """veil: 'accepts every previously executable route' — and the discovery-only ones, which are unchanged too."""
    from aleo_bridge._registry_compatibility import is_registry_version_compatible
    assert is_registry_version_compatible(DEFAULT_REGISTRY, PREVIOUS['version'], route_id)


@pytest.mark.parametrize('route_id', [r['id'] for r in PREVIOUS['routes']])
def test_previous_label_cannot_hide_changed_deployment(route_id):
    from aleo_bridge._registry_compatibility import is_registry_version_compatible
    r = DEFAULT_REGISTRY
    changed = Registry(r.version, r.chains(), r.assets(), [
        replace(v, metadata={**v.metadata, 'routerAddress': '0x0000000000000000000000000000000000000001'}) if v.id == route_id else v
        for v in r.routes(include_unavailable=True)])
    assert not is_registry_version_compatible(changed, PREVIOUS['version'], route_id)


@pytest.mark.parametrize('route_id', [
    'hyperlane:ethereum/bat->aleo/bat', 'hyperlane:aleo/bat->ethereum/bat', 'hyperlane:ethereum/usdg->aleo/usdg',
    'hyperlane:aleo/usdg->ethereum/usdg', 'hyperlane:solana/bat->aleo/bat', 'hyperlane:aleo/bat->solana/bat',
    'hyperlane:solana/usdg->aleo/usdg', 'hyperlane:aleo/usdg->solana/usdg', 'hyperlane:solana/zec->aleo/zec',
    'hyperlane:aleo/zec->solana/zec'])
def test_a_previous_version_plan_cannot_name_a_route_this_snapshot_added(route_id):
    """veil: 'rejects a preceding-version plan for a route activated by this snapshot'."""
    from aleo_bridge._registry_compatibility import is_registry_version_compatible
    from aleo_bridge.errors import RegistryVersionMismatchError
    from aleo_bridge.lifecycle import prepare, resolve_route
    assert not is_registry_version_compatible(DEFAULT_REGISTRY, PREVIOUS['version'], route_id)
    assert not is_registry_version_compatible(DEFAULT_REGISTRY, LEGACY['version'], route_id)
    route = DEFAULT_REGISTRY.route(route_id)
    recipient = {'aleo': 'aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n',
                 'evm': '0x0000000000000000000000000000000000000001', 'solana': '11111111111111111111111111111111'}[
        DEFAULT_REGISTRY.chain(DEFAULT_REGISTRY.asset(route.destination_asset_id).chain_id).family]
    plan = prepare(DEFAULT_REGISTRY, route=route, amount='0.0001', recipient=recipient)
    with pytest.raises(RegistryVersionMismatchError):
        resolve_route(DEFAULT_REGISTRY, replace(plan, registry_version=PREVIOUS['version']))


def test_legacy_routes_remain_compatible_with_the_pinned_previous_registry():
    """The legacy → September path survives for callers that pin that reviewed registry."""
    from aleo_bridge._registry_compatibility import is_registry_version_compatible
    previous = _registry_from_snapshot(PREVIOUS)
    for route in LEGACY['routes']:
        assert is_registry_version_compatible(previous, LEGACY['version'], route['id'])
    assert not is_registry_version_compatible(previous, LEGACY['version'], 'cctp:ethereum/usdc->arc/usdc')


def test_an_old_registry_rejects_a_new_checkpoint_even_on_unchanged_routes():
    """Upgrade recovery services first: a 2026-09-28 client never accepts a 2026-09-30 checkpoint."""
    from aleo_bridge._registry_compatibility import is_registry_version_compatible
    previous = _registry_from_snapshot(PREVIOUS)
    assert not is_registry_version_compatible(previous, DEFAULT_REGISTRY.version, 'xreserve:ethereum/usdc->aleo/usdcx')


@pytest.mark.parametrize('source_chain,source_asset,destination_chain,destination_asset,amount,recipient', [
    ('ethereum', 'usdc', 'aleo', 'usdcx', '25', 'aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n'),
    ('ethereum', 'usdc', 'arc', 'usdc', '25', '0x0000000000000000000000000000000000000001'),
    ('ethereum', 'wbtc', 'aleo', 'wbtc', '1', 'aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n'),
    ('solana', 'sol', 'aleo', 'sol', '1', 'aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n'),
])
def test_a_plan_prepared_against_the_previous_snapshot_resolves_on_the_current_registry(
        source_chain, source_asset, destination_chain, destination_asset, amount, recipient):
    """veil: 'accepts an unchanged %s route from the preceding snapshot'."""
    from aleo_bridge.lifecycle import prepare, resolve_route
    previous = _registry_from_snapshot(PREVIOUS)
    plan = prepare(previous, source_chain=source_chain, source_asset=source_asset, destination_chain=destination_chain,
                   destination_asset=destination_asset, amount=amount, recipient=recipient)
    assert plan.registry_version == PREVIOUS['version']
    assert resolve_route(DEFAULT_REGISTRY, plan).route.id == plan.route_id


def test_a_previous_prepared_checkpoint_recovers_without_network_or_signing():
    from aleo_bridge.lifecycle import recover
    from tests.test_recover import _aleo_eth_checkpoint
    from tests.fakes.fake_bridge import FakeBridge
    b = FakeBridge(ethereum=False)
    serialized = json.dumps({'type': 'execute', 'id': 'at1prepared', 'fee': {}})
    _, cp = _aleo_eth_checkpoint(b, preparedTransaction={'transactionId': 'at1prepared', 'serializedTransaction': serialized})
    cp['route']['registryVersion'] = PREVIOUS['version']
    result = recover(b, cp)
    assert result.next == 'resume' and cp['route']['registryVersion'] == PREVIOUS['version']
    assert b.calls == [] and b.events == []
