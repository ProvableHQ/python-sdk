import importlib
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from aleo_bridge.errors import BridgeError
from tests.fakes.fake_cctp import Harness, RECIPIENT

sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))


def test_interrupted_leg_recovers_checkpoint_without_executing_again(tmp_path):
    workflow = importlib.import_module('_arc_workflow')
    h = Harness()
    real_execute = h.bridge.execute
    def interrupted(*args, **kwargs):
        real_execute(*args, **kwargs)
        raise RuntimeError('interrupted after checkpoint')
    h.bridge.execute = interrupted
    kwargs = dict(route=h.route.id, amount='5', recipient=RECIPIENT, state_path=tmp_path/'leg.json', execute=True,
                  cctp={'speed':'fast','max_fee':'0.1'})
    with pytest.raises(RuntimeError): workflow.run_leg(h.bridge, **kwargs)
    h.bridge.execute = Mock(side_effect=AssertionError('duplicate burn'))
    assert workflow.run_leg(h.bridge, **kwargs) == 4_990_000
    h.bridge.execute.assert_not_called()
    assert workflow.run_leg(h.bridge, **kwargs) == 4_990_000


def test_arc_reserve_is_integer_and_cannot_exhaust_receipts():
    workflow = importlib.import_module('_arc_workflow')
    assert workflow.spendable(4_990_000,'0.10') == '4.89'
    for reserve in ('5','-1','0.1000001'):
        with pytest.raises((BridgeError, ValueError)): workflow.spendable(4_990_000,reserve)


def test_provider_handoff_never_claims_received_funds(tmp_path):
    workflow = importlib.import_module('_arc_workflow')
    h = Harness()
    h.circle.forward_hash = None
    h.destination.used = False
    with pytest.raises(BridgeError):
        workflow.run_leg(h.bridge,route=h.route.id,amount='5',recipient=RECIPIENT,
                         state_path=tmp_path/'leg.json',execute=True,timeout=0,
                         cctp={'speed':'fast','max_fee':'0.1'})
    assert (tmp_path/'leg.json').exists()
    assert len(h.source.sent) == 1


def test_xreserve_repricing_returns_observed_net_not_initial_quote(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from aleo_bridge import Receipt, Status
    from aleo_bridge.lifecycle import prepare
    from aleo_bridge.types import to_progress
    from aleo_bridge.registry import DEFAULT_REGISTRY
    workflow = importlib.import_module('_arc_workflow')
    route = 'xreserve:aleo/usdcx->arc/usdc'
    plan = prepare(DEFAULT_REGISTRY,route=route,amount='5',recipient=RECIPIENT)
    receipt = Receipt('at1burn','xreserve',Status.COMPLETED,source_tx_id='at1burn',
                      protocol_state={'routeId':route,'expectedDestinationIncreaseAtomic':'4980000',
                                      'destinationBalanceBeforeAtomic':'10000000'})
    bridge = Mock(registry=DEFAULT_REGISTRY)
    bridge.quote.return_value = SimpleNamespace(plan=plan,amount_out='4.99',fees=[])
    bridge.execute.return_value = to_progress(plan,receipt)
    balances = iter([10_000_000,14_980_000])
    monkeypatch.setattr(workflow,'_balance',lambda *args: next(balances),raising=False)
    assert workflow.run_leg(bridge,route=route,amount='5',recipient=RECIPIENT,
                             state_path=tmp_path/'leg.json',execute=True) == 4_980_000


@pytest.mark.parametrize('execute',[False,True])
@pytest.mark.parametrize('module_name',['bridge_arc_to_aleo','bridge_ethereum_arc_aleo'])
def test_arc_example_commands_preview_and_execute(module_name,execute,monkeypatch,tmp_path):
    module = importlib.import_module(module_name)
    build = Mock(return_value=Mock())
    run = Mock(return_value=4_990_000 if execute else None)
    monkeypatch.setattr(module,'build_bridge',build)
    monkeypatch.setattr(module,'run_leg',run)
    args = ['--sender',RECIPIENT,'--recipient','aleo1recipient','--journal',str(tmp_path)]
    if execute: args.append('--execute')
    assert module.main(args) == 0
    assert all(c.kwargs['execute'] is execute for c in run.call_args_list)
    assert run.call_count == (2 if execute and module_name == 'bridge_ethereum_arc_aleo' else 1)
    if run.call_count == 2:
        assert run.call_args.kwargs['amount'] == '4.89'


@pytest.mark.parametrize('l2',['base','arbitrum'])
@pytest.mark.parametrize('step',[1,2,3,4])
@pytest.mark.parametrize('execute',[False,True])
def test_each_l2_roundtrip_step_uses_previous_received_budget(l2,step,execute,monkeypatch,tmp_path):
    import json
    module = importlib.import_module('l2_arc_aleo_roundtrip')
    routes = [f'cctp:{l2}/usdc->arc/usdc','xreserve:arc/usdc->aleo/usdcx',
              'xreserve:aleo/usdcx->arc/usdc',f'cctp:arc/usdc->{l2}/usdc']
    if step > 1:
        root = tmp_path/l2
        root.mkdir()
        (root/f'leg-{step-1}.json').write_text(json.dumps({'done':True,'received_atomic':4_990_000,
                       'request':{'route':routes[step-2]}}))
    monkeypatch.setattr(module,'build_bridge',Mock(return_value=Mock()))
    run = Mock(return_value=4_000_000 if execute else None)
    monkeypatch.setattr(module,'run_leg',run)
    args = ['--sender',RECIPIENT,'--recipient','aleo1recipient','--journal',str(tmp_path),
            '--l2',l2,'--step',str(step)]
    if execute: args.append('--execute')
    assert module.main(args) == 0
    run.assert_called_once()
    assert run.call_args.kwargs['route'] == routes[step-1]
    assert run.call_args.kwargs['execute'] is execute
    assert run.call_args.kwargs['amount'] == ('5' if step == 1 else '4.89' if step in (2,4) else '4.99')
