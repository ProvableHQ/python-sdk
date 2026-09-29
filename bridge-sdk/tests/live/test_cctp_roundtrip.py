"""Funded received-only roundtrips. No gate is set or inferred by this module."""
from pathlib import Path
import sys

import pytest
from aleo_bridge.units import format_decimal_amount
from . import config
from .helpers import build_bridge

sys.path.insert(0,str(Path(__file__).parents[2]/'examples'))
from _arc_workflow import run_leg, spendable

pytestmark = [pytest.mark.live, pytest.mark.slow]


def workflow_gate(case):
    if not config.mainnet_case_enabled(case):
        pytest.skip(f'enable the existing live-funds/state/mainnet gates and case {case}')
    return config.state_dir()/'mainnet'/case, config.mainnet_execution_enabled()


@pytest.mark.parametrize('other',['ethereum','base','arbitrum'])
def test_cctp_received_only_roundtrip(other):
    root, execute = workflow_gate('cctp-roundtrip')
    bridge = build_bridge('mainnet')
    address = bridge.evm(other).conn.require_address()
    first = run_leg(bridge,route=f'cctp:{other}/usdc->arc/usdc',amount='5',recipient=address,sender=address,
                    state_path=root/other/'out.json',execute=execute,timeout=1200)
    if not execute:
        return
    assert first is not None and 0 < first <= 5_000_000
    returned = run_leg(bridge,route=f'cctp:arc/usdc->{other}/usdc',amount=spendable(first),recipient=address,
                       sender=address,state_path=root/other/'return.json',execute=True,timeout=1200)
    assert returned is not None and 0 < returned < first


@pytest.mark.parametrize('other',['base','arbitrum'])
def test_public_l2_arc_aleo_four_leg_journey(other):
    root, execute = workflow_gate('arc-journey')
    bridge = build_bridge('mainnet')
    evm = bridge.evm(other).conn.require_address()
    aleo = bridge.aleo_address()
    legs = [(f'cctp:{other}/usdc->arc/usdc',evm,evm),('xreserve:arc/usdc->aleo/usdcx',evm,aleo),
            ('xreserve:aleo/usdcx->arc/usdc',aleo,evm),(f'cctp:arc/usdc->{other}/usdc',evm,evm)]
    amount = '5'
    for i,(route,sender,recipient) in enumerate(legs):
        received = run_leg(bridge,route=route,amount=amount,recipient=recipient,sender=sender,
                           state_path=root/other/f'leg-{i}.json',execute=execute,timeout=1200)
        if not execute:
            return
        assert received is not None and received > 0
        amount = spendable(received) if i in (0,2) else format_decimal_amount(received,6)
