"""An uncertain submission must never trigger another deposit."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from aleo_bridge import PollingTimeoutError
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))

@pytest.mark.parametrize('error,code', [
    (PollingTimeoutError('pending', status='DELIVERY_PENDING'), 2),
    (RuntimeError('secret-key'), 1),
])
def test_entrypoint_errors_do_not_retry_or_expose_credentials(error, code, monkeypatch, capsys):
    import runpy
    import aleo_bridge
    bridge = Mock()
    bridge.quote.side_effect = error
    monkeypatch.setattr(aleo_bridge, 'Bridge', Mock(return_value=bridge))
    monkeypatch.setattr(sys, 'argv', ['quote_transfer.py', '--sender', 'sender', '--recipient', 'recipient'])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / 'examples' / 'quote_transfer.py'), run_name='__main__')
    assert exc.value.code == code
    bridge.quote.assert_called_once()
    output = capsys.readouterr().err
    assert 'secret-key' not in output
    assert 'recover' in output.lower()


def configure_example(module, bridge, monkeypatch):
    monkeypatch.setattr(module, 'Bridge', Mock(return_value=bridge))
    monkeypatch.setattr(module, 'Aleo', Mock())
    if hasattr(module, 'Ethereum'):
        monkeypatch.setattr(module, 'Ethereum', Mock())
    monkeypatch.setenv('EVM_PRIVATE_KEY', 'test-only-key')
    monkeypatch.setenv('ALEO_PRIVATE_KEY', 'test-only-key')
    monkeypatch.setenv('BRIDGE_MINT_SECRET_NONCE', '123scalar')


@pytest.mark.parametrize('module_name', ['bridge_wbtc', 'bridge_usdc_private_balance', 'bridge_usdc_private_recipient'])
@pytest.mark.parametrize('execute', [False, True])
def test_transfer_examples_quote_or_submit_once(module_name, execute, monkeypatch, tmp_path):
    module = importlib.import_module(module_name)
    bridge = Mock()
    quote = SimpleNamespace(amount_out='2', fees=[], plan=object())
    bridge.quote.return_value = quote
    progress = SimpleNamespace(next='done', error=None, receipt=SimpleNamespace(id='receipt', source_tx_id='source'))
    bridge.execute.return_value = progress
    bridge.aleo_address.return_value = 'recipient'
    configure_example(module, bridge, monkeypatch)
    args = ['--sender', 'sender', '--recipient', 'recipient', '--journal', str(tmp_path)]
    if execute:
        args += ['--execute']
    assert module.main(args) == 0
    if execute:
        assert bridge.execute.call_count == 1
        assert bridge.execute.call_args.args == (quote.plan,)
    else:
        bridge.execute.assert_not_called()
    bridge.resume.assert_not_called()


def test_private_example_claims_once_when_ready(monkeypatch, tmp_path):
    module = importlib.import_module('bridge_usdc_private_recipient')
    bridge = Mock()
    bridge.aleo_address.return_value = 'recipient'
    bridge.quote.return_value = SimpleNamespace(amount_out='2', fees=[], plan=object())
    ready = SimpleNamespace(next='complete')
    done = SimpleNamespace(next='done', receipt=SimpleNamespace(id='claim', source_tx_id='deposit'))
    bridge.execute.return_value = SimpleNamespace(next='wait')
    bridge.wait.return_value = ready
    bridge.complete.return_value = done
    configure_example(module, bridge, monkeypatch)
    assert module.main(['--recipient', 'recipient', '--execute', '--journal', str(tmp_path)]) == 0
    bridge.complete.assert_called_once()
    assert bridge.complete.call_args.kwargs['secret_nonce'] == '123scalar'


def test_no_file_recovery_uses_actual_sdk_without_submitting(monkeypatch):
    from aleo_bridge import Bridge, Ethereum
    from tests.fakes.fake_web3 import fake_web3, make_bridge, dispatch_id_log
    from tests.test_eth_status import H2, MAILBOX, MESSAGE_ID, ACCT, WBTC_ROUTER, ALEO
    module = importlib.import_module('recover_without_files')
    w3 = fake_web3()
    w3.provider.add_receipt(H2, logs=[dispatch_id_log(MAILBOX, MESSAGE_ID, tx_hash=H2)],
                            sender=ACCT.address, to=WBTC_ROUTER)
    bridge = Bridge(make_bridge().aleo, ethereum=Ethereum(w3=w3))
    monkeypatch.setattr(module, 'Bridge', Mock(return_value=bridge))
    assert module.main(['--transaction', H2, '--sender', ACCT.address, '--recipient', ALEO,
                        '--amount', '0.001']) == 2
    assert bridge.checkpoints is None
    assert not any(method.startswith('eth_send') for method in w3.provider.methods)


def test_shield_example_submits_once_and_reports_unconfirmed(monkeypatch):
    module = importlib.import_module('shield_assets')
    bridge = Mock()
    bridge.shield.return_value.delegate.return_value.transaction_id = 'at1submitted'
    configure_example(module, bridge, monkeypatch)
    assert module.main([]) == 0
    bridge.shield.assert_not_called()
    assert module.main(['--execute']) == 2
    bridge.shield.assert_called_once_with('aleo/sol', amount='0.01')
    bridge.shield.return_value.delegate.assert_called_once()


def test_journal_example_reads_bundled_entry_without_resubmitting(monkeypatch):
    module = importlib.import_module('recover_from_journal')
    directory = Path(__file__).resolve().parents[1] / 'examples' / 'checkpoints'
    checkpoint = next(directory.glob('*.json'))
    original = checkpoint.read_bytes()
    bridge = Mock()
    bridge.recover.return_value = SimpleNamespace(next='wait', receipt=SimpleNamespace(id='message', source_tx_id='signature'))
    configure_example(module, bridge, monkeypatch)
    assert module.main(['--journal', str(directory), '--id', checkpoint.stem]) == 2
    bridge.execute.assert_not_called()
    bridge.resume.assert_not_called()
    bridge.complete.assert_not_called()
    assert checkpoint.read_bytes() == original


@pytest.mark.parametrize('next_step,code', [('done', 0), ('failed', 1), ('resume', 2), ('complete', 2)])
def test_recovery_status_does_not_submit(next_step, code, monkeypatch):
    module = importlib.import_module('recover_from_journal')
    directory = Path(__file__).resolve().parents[1] / 'examples' / 'checkpoints'
    checkpoint = next(directory.glob('*.json'))
    bridge = Mock()
    bridge.recover.return_value = SimpleNamespace(next=next_step, error=None,
        receipt=SimpleNamespace(id='receipt', source_tx_id='source'))
    configure_example(module, bridge, monkeypatch)
    assert module.main(['--journal', str(directory), '--id', checkpoint.stem]) == code
    bridge.execute.assert_not_called()
    bridge.resume.assert_not_called()
    bridge.complete.assert_not_called()


@pytest.mark.parametrize('name,source,destination,asset,mode', [
    ('bridge_wbtc_to_ethereum', 'wbtc', 'ethereum', 'wbtc', 'signer'),
    ('bridge_sol_to_solana', 'sol', 'solana', 'sol', 'signer'),
    ('bridge_usdcx_to_ethereum', 'usdcx', 'ethereum', 'usdc', 'public'),
])
@pytest.mark.parametrize('execute', [False, True])
def test_outbound_tutorials_spend_public_balance_once(name, source, destination, asset, mode, execute, monkeypatch, tmp_path):
    module = importlib.import_module(name)
    bridge = Mock()
    quote = SimpleNamespace(amount_out='1', fees=[], plan=object())
    bridge.quote.return_value = quote
    bridge.execute.return_value = SimpleNamespace(next='done', receipt=SimpleNamespace(id='receipt', source_tx_id='at1source'))
    configure_example(module, bridge, monkeypatch)
    if hasattr(module, 'Solana'):
        monkeypatch.setattr(module, 'Solana', Mock())
    args = ['--sender', 'aleo-sender', '--recipient', 'destination', '--journal', str(tmp_path)]
    if execute:
        args += ['--execute']
    assert module.main(args) == 0
    call = bridge.quote.call_args.kwargs
    assert (call['source_chain'], call['source_asset'], call['destination_chain'], call['destination_asset']) == ('aleo', source, destination, asset)
    assert call['recipient'] == 'destination'
    if execute:
        bridge.execute.assert_called_once()
        assert bridge.execute.call_args.kwargs['mode'] == mode
    else:
        bridge.execute.assert_not_called()
    bridge.resume.assert_not_called()
