"""Funding failures stop the example before any swap work."""
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from aleo_shield_swap import ConfirmAirdropResult

EXAMPLE = Path(__file__).parents[1] / 'examples/first-swap/swap.py'


@pytest.mark.parametrize('status', ['rate_limited', 'failed', 'rejected', 'pending', 'empty', 'accepted', 'funded'])
def test_funding_gate(monkeypatch, status):
    monkeypatch.setenv('SHIELD_SWAP_PRIVATE_KEY', 'test-key')
    monkeypatch.setattr('aleo.testnet.PrivateKey', Mock())
    aleo = Mock()
    aleo.records.register.return_value = {'ok': True}
    monkeypatch.setattr('aleo.Aleo', Mock(return_value=aleo))
    client = Mock()
    result = SimpleNamespace(symbol='USDCx', status=status, tx_id='at1funding', error=None)
    client.confirm_airdrop.return_value = ConfirmAirdropResult(
        status='rate_limited' if status == 'rate_limited' else 'settled',
        message='Try later',
        job=None if status == 'rate_limited' else SimpleNamespace(results=[] if status == 'empty' else [result]),
    )
    client.has_swap_balance.return_value = status == 'funded'
    client.api.get_token.return_value = SimpleNamespace(address='1field')
    client.quote.side_effect = RuntimeError('reached quote')
    monkeypatch.setattr('aleo_shield_swap.ShieldSwap', Mock(return_value=client))
    with pytest.raises(RuntimeError) as caught:
        runpy.run_path(str(EXAMPLE), run_name='__main__')
    if status in ('accepted', 'funded'):
        assert str(caught.value) == 'reached quote'
    else:
        client.quote.assert_not_called()
        assert 'airdrop' in str(caught.value).lower()
        if status in ('failed', 'rejected', 'pending'):
            assert status in str(caught.value) and 'at1funding' in str(caught.value)
    client.swap.assert_not_called()
    client.has_swap_balance.assert_called_once_with('1field', '1.5')
    if status == 'funded':
        client.confirm_airdrop.assert_not_called()
    else:
        client.confirm_airdrop.assert_called_once()


@pytest.mark.parametrize('statuses', [[], ['accepted'], ['accepted', 'failed'], ['rejected'], ['pending']])
def test_airdrop_success_and_error(statuses):
    funding = ConfirmAirdropResult('settled', job=SimpleNamespace(results=[
        SimpleNamespace(symbol='USDCx', status=status, tx_id='at1funding', error='transfer detail')
        for status in statuses
    ]))
    assert funding.success is (statuses == ['accepted'])
    if funding.success:
        assert funding.error is None
    elif statuses:
        assert 'at1funding' in funding.error and 'transfer detail' in funding.error
    else:
        assert 'no token results' in funding.error


def test_airdrop_rate_limit_exposes_reason():
    funding = ConfirmAirdropResult('rate_limited', message='Try again tomorrow')
    assert not funding.success
    assert 'Try again tomorrow' in funding.error


def test_airdrop_missing_job_is_not_success():
    funding = ConfirmAirdropResult('settled')
    assert not funding.success
    assert 'no token results' in funding.error
