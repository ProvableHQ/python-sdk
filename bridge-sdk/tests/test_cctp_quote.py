"""CCTP estimates preserve the approved ceiling and exact six-decimal accounting."""
from dataclasses import replace

import pytest

from aleo_bridge.errors import BridgeError
from aleo_bridge.lifecycle import prepare
from aleo_bridge.registry import DEFAULT_REGISTRY
from tests.fakes.fake_web3 import make_bridge

ROUTES = [f'cctp:{a}/usdc->{b}/usdc' for other in ('ethereum', 'base', 'arbitrum')
          for a, b in ((other, 'arc'), ('arc', other))]
SENDER, RECIPIENT = '0x'+'11'*20, '0x'+'22'*20


class CircleSession:
    def __init__(self, minimum='1.3', forward=1000):
        self.minimum, self.forward = minimum, forward
        self.urls = []
        self.status_code = 200
        self.body = None

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self

    def json(self):
        if self.body is not None:
            return self.body
        return [{'finalityThreshold': 1000, 'minimumFee': self.minimum, 'forwardFee': {'medium': self.forward}},
                {'finalityThreshold': 2000, 'minimumFee': '0', 'forwardFee': {'med': self.forward}}]


def bridge():
    b = make_bridge()
    b.cctp.circle_session = CircleSession()
    return b


@pytest.mark.parametrize('route', ROUTES)
def test_fast_quote_deducts_full_forwarding_ceiling(route):
    b = bridge()
    q = b.quote(route=route, amount='5', sender=SENDER, recipient=RECIPIENT,
                cctp={'speed': 'fast', 'max_fee': '0.1'})
    assert q.kind == 'evm-cctp' and q.protocol_fee_atomic == 650
    assert q.forwarding_fee_atomic == 1000 and q.max_fee_atomic == 100_000
    assert q.amount_out == '4.9' and q.amount_out_atomic == 4_900_000
    assert q.plan.cctp.max_fee == '0.1' and q.min_finality_threshold == 1000


def test_standard_defaults_and_plan_serialization():
    from aleo_bridge import Plan
    b = bridge()
    q = b.quote(route=ROUTES[0], amount='5', recipient=RECIPIENT)
    assert q.plan.cctp.speed == 'standard' and q.plan.cctp.forwarding is True
    assert q.plan.cctp.max_fee == '0.0011' and q.amount_out == '4.9989'
    assert Plan.from_dict(q.plan.to_dict()) == q.plan
    assert b.cctp.circle_session.urls[-1].endswith('/fees/0/26?forward=true')


def test_manual_quote_deducts_estimate_and_requote_never_raises_cap():
    b = bridge()
    q = b.quote(route=ROUTES[0], amount='5', recipient=RECIPIENT,
                cctp={'speed': 'fast', 'forwarding': False, 'max_fee': '0.1'})
    assert q.amount_out == '4.99935' and q.forwarding_fee_atomic == 0
    assert q.plan.steps[-1].executor == 'evm-wallet'
    b.cctp.circle_session.minimum = '1000'
    with pytest.raises(BridgeError, match='exceed'):
        b.cctp.quote(q.plan)


def test_fractional_basis_points_round_up():
    b = bridge()
    b.cctp.circle_session.minimum = '0.0001'
    q = b.quote(route=ROUTES[0], amount='5', recipient=RECIPIENT,
                cctp={'speed': 'fast', 'forwarding': False})
    assert q.protocol_fee_atomic == 1 and q.amount_out == '4.999999'


@pytest.mark.parametrize('options', [{'speed': 'warp'}, {'forwarding': 1}, {'max_fee': '-1'},
                                     {'max_fee': True}, {'max_fee': '5'}, {'unknown': 1}])
def test_invalid_options_cannot_prepare_or_quote(options):
    b = bridge()
    with pytest.raises(BridgeError):
        b.quote(route=ROUTES[0], amount='5', recipient=RECIPIENT, cctp=options)


@pytest.mark.parametrize('minimum,forward', [(True, 1000), ('-1', 1000), ('nan', 1000), ('1e2', 1000),
                                            ('1', True), ('1', -1), ('1', '1.5')])
def test_malformed_provider_fees_refused(minimum, forward):
    b = bridge()
    b.cctp.circle_session = CircleSession(minimum, forward)
    with pytest.raises(BridgeError):
        b.quote(route=ROUTES[0], amount='5', recipient=RECIPIENT, cctp={'speed': 'fast'})


def test_cctp_options_refused_on_other_protocols():
    with pytest.raises(BridgeError, match='CCTP'):
        prepare(DEFAULT_REGISTRY, route='xreserve:ethereum/usdc->aleo/usdcx', amount='2',
                recipient='aleo1'+'a'*58, cctp={})


def test_changed_plan_amount_and_mislabeled_asset_refused():
    b = bridge()
    q = b.quote(route=ROUTES[0], amount='5', recipient=RECIPIENT)
    with pytest.raises(BridgeError):
        b.cctp.quote(replace(q.plan, amount_atomic=6_000_000))
