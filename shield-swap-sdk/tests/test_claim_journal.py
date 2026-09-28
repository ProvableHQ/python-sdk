"""Confirmed single claims update the journal; uncertain submissions stay pending."""
from unittest.mock import Mock
import pytest
from aleo_shield_swap import ShieldSwap, Journal
from .test_claim import _stub, _handle, SWAP_OUTPUT_TEXT


@pytest.fixture
def journalled(tmp_path):
    stub = _stub({'77field': SWAP_OUTPUT_TEXT})
    dex = ShieldSwap(stub)
    dex.journal = Journal(tmp_path / 'claims.jsonl')
    dex.journal.record_swap(_handle(), 0)
    dex._is_wrapped = Mock(return_value=False)
    dex._amm_token_program = Mock(return_value='tok.aleo')
    dex._token_program = Mock(return_value='tok.aleo')
    dex._ensure = Mock()
    return dex, stub


@pytest.mark.parametrize('method', ['delegate', 'transact'])
@pytest.mark.parametrize('outcome', ['confirmed', 'timeout', 'rejected', 'submit_failed', 'no_wait'])
def test_single_claim_journals_only_confirmation(journalled, method, outcome):
    dex, stub = journalled
    call = dex.claim_swap_output(_handle())
    # Full payload avoids the mandatory lookup wait for ID-only responses.
    call._bound.delegate = Mock(return_value={'transaction': {
        'id': 'at1claim', 'execution': {'transitions': []}}})
    def confirm(*args, **kwargs):
        assert dex.journal.pending_claims() == [_handle()]
        if outcome == 'timeout':
            raise TimeoutError('pending')
        if outcome == 'rejected':
            raise RuntimeError('rejected')
    stub.network.wait_for_transaction = Mock(side_effect=confirm)
    if outcome == 'submit_failed':
        if method == 'delegate':
            call._bound.delegate.side_effect = RuntimeError('submission failed')
        else:
            stub.network.submit_transaction = Mock(side_effect=RuntimeError('submission failed'))
    if outcome in ('timeout', 'rejected', 'submit_failed'):
        with pytest.raises((TimeoutError, RuntimeError)):
            getattr(call, method)(wait=True)
    else:
        result = getattr(call, method)(wait=outcome != 'no_wait')
    claims = [e for e in dex.journal.events() if e['type'] == 'claim']
    assert len(claims) == (1 if outcome == 'confirmed' else 0)
    if outcome == 'confirmed':
        assert claims[0]['transaction_id'] == result.transaction_id
        assert claims[0]['amount_out'] == result.amount_out
        assert dex.journal.pending_claims() == []
    else:
        assert dex.journal.pending_claims() == [_handle()]


def test_collect_all_records_claim_once(journalled):
    dex, stub = journalled
    dex.get_positions = Mock(return_value=[])
    report = dex.collect_all()
    assert len(report.claimed) == 1
    assert len([e for e in dex.journal.events() if e['type'] == 'claim']) == 1
    assert dex.collect_all().claimed == []


def test_simulate_does_not_record_claim(journalled):
    dex, stub = journalled
    dex.claim_swap_output(_handle()).simulate()
    assert dex.journal.pending_claims() == [_handle()]
