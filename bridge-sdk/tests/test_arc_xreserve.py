"""Arc xReserve binds the selected domain and rechecks live withdrawal coverage."""
from dataclasses import replace

import pytest
from eth_account import Account

from aleo_bridge import Ethereum
from aleo_bridge.errors import BridgeError, ConfigurationError, InvalidAmountError
from aleo_bridge.registry import DEFAULT_REGISTRY
from tests.fakes.fake_web3 import fake_web3, make_bridge
from tests.test_eth_xreserve_quote import ALEO, KEY, MAINNET_XRESERVE

ARC_TOKEN = '0x3600000000000000000000000000000000000000'
OUT = DEFAULT_REGISTRY.route('xreserve:aleo/usdcx->arc/usdc')
IN = DEFAULT_REGISTRY.route('xreserve:arc/usdc->aleo/usdcx')


class FeeSession:
    def __init__(self, fee='16400', status=200):
        self.fee, self.status_code, self.calls = fee, status, []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self

    def json(self):
        return {'withdrawalFeeBaseUnits': self.fee}


@pytest.mark.parametrize('mode,index', [('private', 2), ('public', 1), ('public-as-signer', 1)])
def test_arc_burn_uses_domain_26(mode, index):
    b = make_bridge()
    _, _, inputs = b.xreserve.build_burn_inputs(OUT, mode=mode, amount_atomic=2_000_000,
                       recipient='0x'+'11'*20, record='record', merkle_proof='[proof]')
    assert inputs[index] == '26u32'


def test_minimum_burn_even_when_fee_is_smaller():
    with pytest.raises(InvalidAmountError, match='minimum'):
        make_bridge().xreserve.build_burn_inputs(OUT, mode='public', amount_atomic=1_999_999,
                          recipient='0x'+'11'*20, record=None, merkle_proof=None)


def test_quote_uses_live_fee_and_selected_route():
    b = make_bridge()
    b.xreserve.circle_session = session = FeeSession()
    q = b.quote(route=OUT, amount='2', recipient='0x'+'11'*20)
    assert q.status == 'quoted' and q.withdrawal_fee_atomic == 16400 and q.amount_out == '1.9836'
    assert session.calls[0][1]['json'] == {'evmChain': 'arc', 'amountUsdc': '2'}


@pytest.mark.parametrize('fee,status', [(True, 200), ('-1', 200), (None, 200), ('1.5', 200), ('2000000', 200), ('16400', 503)])
def test_bad_or_unaffordable_live_fee_refuses_quote(fee, status):
    b = make_bridge()
    b.xreserve.circle_session = FeeSession(fee, status)
    with pytest.raises(BridgeError):
        b.quote(route=OUT, amount='2', recipient='0x'+'11'*20)


def test_execute_rechecks_fee_before_proving(monkeypatch):
    b = make_bridge()
    b.xreserve.circle_session = session = FeeSession()
    q = b.quote(route=OUT, amount='2', recipient='0x'+'11'*20)
    session.fee = '2000000'
    def fail_prove(*args, **kwargs):
        pytest.fail('Unaffordable burn reached proving')
    monkeypatch.setattr(b, '_call', fail_prove)
    with pytest.raises(InvalidAmountError):
        b.execute(q.plan, mode='public')
    assert len(session.calls) == 2


@pytest.mark.parametrize('mode', ['public', 'record', 'private'])
def test_arc_deposit_preserves_mint_modes(mode):
    address = Account.from_key(KEY).address
    w3 = fake_web3(chain_id=5042, token_balances={(ARC_TOKEN, address): 3_000_000},
                   allowances={(ARC_TOKEN, address, MAINNET_XRESERVE): 0})
    b = make_bridge(evm={'arc': Ethereum(w3=w3, private_key=KEY)})
    q = b.quote(route=IN, amount='2', recipient=ALEO, sender=address, mint_mode=mode, secret_nonce='7scalar')
    assert q.plan.route_id == IN.id and q.plan.mint_mode == mode
    assert q.balance_atomic == 3_000_000


def test_missing_destination_domain_refuses_arc_burn():
    from aleo_bridge.registry import Registry
    b = make_bridge()
    r = b.registry
    b.registry = Registry(r.version, [replace(c, protocol_domains={}) if c.id == 'arc' else c for c in r.chains()],
                          r.assets(), r.routes(include_unavailable=True))
    with pytest.raises(ConfigurationError, match='domain'):
        b.xreserve.build_burn_inputs(OUT, mode='public', amount_atomic=2_000_000,
                                     recipient='0x'+'11'*20, record=None, merkle_proof=None)


def test_execute_uses_one_live_fee_for_validation_and_delivery(monkeypatch):
    from aleo_bridge import Receipt, Status
    from aleo_bridge import lifecycle
    recipient = '0x' + '11' * 20
    b = make_bridge(evm={'arc': Ethereum(w3=fake_web3(chain_id=5042,
        token_balances={(ARC_TOKEN, recipient): 100}))})
    session = FeeSession()
    b.xreserve.circle_session = session
    plan = b.quote(route=OUT, amount='2', recipient=recipient).plan
    session.calls.clear()
    captured = {}
    def prove(bridge, plan, call, *, extra_state, **kwargs):
        captured.update(extra_state)
        return Receipt('test-burn', 'xreserve', Status.SOURCE_CONFIRMING,
                       source_tx_id='test-burn', protocol_state={'routeId': OUT.id, **extra_state})
    monkeypatch.setattr(lifecycle, '_run_aleo_leg', prove)
    b.execute(plan, mode='public')
    assert len(session.calls) == 1
    assert captured['expectedDestinationIncreaseAtomic'] == '1983600'
