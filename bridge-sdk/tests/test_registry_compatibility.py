"""Legacy deployment fingerprints must survive additions, but never deployment changes."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from aleo_bridge.registry import DEFAULT_REGISTRY, Registry, validate_registry
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
