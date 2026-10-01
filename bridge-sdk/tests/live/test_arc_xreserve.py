"""Named Arc xReserve cases, using the existing funded gates and recovery engine."""
import pytest
import json
import requests
from web3 import Web3
from eth_utils.crypto import keccak
from aleo_bridge import Bridge, Ethereum
from aleo_bridge._cctp_abi import TOKEN_ABI
from aleo_bridge.errors import BridgeError
from aleo_bridge.privacy import record_amount
from aleo_bridge.client import build_aleo
from aleo_bridge.registry import DEFAULT_REGISTRY
from .test_lifecycle_live import _mainnet, mainnet_client, _register_record_scanner
from . import config
from .helpers import wait_for, wait_for_aleo_transaction
from .test_cctp_roundtrip import workflow_gate
from _arc_workflow import _save

pytestmark = [pytest.mark.live, pytest.mark.slow]


def test_arc_to_aleo_private_mint(mainnet_client,record_property):
    _mainnet('evm-xreserve',DEFAULT_REGISTRY.route('xreserve:arc/usdc->aleo/usdcx'),mainnet_client,record_property)


def exact_arc_delivery(conn, *, recipient, start_block, before, expected):
    """Independent live-test proof: recent exact transfer, successful receipt, exact balance delta."""
    token = conn.w3.eth.contract(address='0x3600000000000000000000000000000000000000',abi=TOKEN_ABI)
    topics = ['0x'+keccak(text='Transfer(address,address,uint256)').hex(),None,
              '0x'+bytes.fromhex(recipient[2:]).rjust(32,b'\0').hex()]
    head = conn.w3.eth.block_number
    for start in range(start_block,head+1,1000):
        for log in conn.w3.eth.get_logs({'address':token.address,'topics':topics,
                                        'fromBlock':start,'toBlock':min(head,start+999)}):
            if not log['topics'] or Web3.to_hex(log['topics'][0]) != topics[0]:
                continue
            event = token.events.Transfer().process_log(log)['args']
            if event['to'].lower() != recipient.lower() or event['value'] != expected:
                continue
            tx_hash = Web3.to_hex(log['transactionHash'])
            receipt = conn.get_receipt(tx_hash)
            if receipt is None:
                continue
            assert receipt['status'] == 1 and Web3.to_hex(receipt['transactionHash']) == tx_hash
            matching = False
            for entry in receipt['logs']:
                if (entry['address'].lower() != token.address.lower() or not entry['topics']
                        or Web3.to_hex(entry['topics'][0]) != topics[0]):
                    continue
                decoded = token.events.Transfer().process_log(entry)['args']
                matching |= decoded['to'].lower() == recipient.lower() and decoded['value'] == expected
            assert matching, 'Destination receipt must contain the exact token transfer'
            assert token.functions.balanceOf(Web3.to_checksum_address(recipient)).call()-before == expected
            return tx_hash
    return None


class BudgetedFeeSession:
    def __init__(self,state,path,session=None):
        self.state,self.path,self.session = state,path,session or requests.Session()

    def post(self,*args,**kwargs):
        response = self.session.post(*args,**kwargs)
        if response.status_code == 200:
            fee = response.json().get('withdrawalFeeBaseUnits')
            if not isinstance(fee,str) or not fee.isascii() or not fee.isdigit() or int(fee)>100_000:
                raise BridgeError('Withdrawal estimate exceeds the live-test 0.10-USDC fee budget')
            self.state['expected_atomic'] = 2_000_000-int(fee)
            if self.state.get('started'):
                _save(self.path,self.state)
        return response


def test_aleo_private_burn_to_arc(record_property):
    root,execute = workflow_gate('aleo-arc')
    path = root/'withdrawal.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    route = DEFAULT_REGISTRY.route('xreserve:aleo/usdcx->arc/usdc')
    recipient = config.required('BRIDGE_LIVE_ARC_RECIPIENT')
    arc = Ethereum(config.evm_chain_rpc_url('arc'))  # No destination signer is created.
    assert arc.chain_id == 5042 and not arc.can_sign
    aleo = build_aleo(config.aleo_endpoint(),'mainnet',config.aleo_private_key('mainnet'))
    bridge = Bridge(aleo,evm={'arc':arc})
    bridge.xreserve.circle_session = BudgetedFeeSession(state,path)
    if state and (state.get('route') != route.id or state.get('recipient') != recipient):
        raise BridgeError('Saved withdrawal belongs to a different intent')
    def save(cp):
        state['checkpoint'] = cp.to_dict()
        state['source_tx_id'] = (cp.source or {}).get('transactionId')
        _save(path,state)
    if state.get('checkpoint'):
        progress = bridge.recover(state['checkpoint'])
        if progress.next == 'resume':
            if not execute:
                return
            bridge.resume(progress,on_checkpoint=save)
        elif progress.next == 'failed':
            raise BridgeError('Saved Aleo burn failed')
    else:
        if state.get('started'):
            raise BridgeError('Burn started without checkpoint; inspect source history before retrying')
        quote = bridge.quote(route=route,amount='2',recipient=recipient,sender=bridge.aleo_address())
        assert state['expected_atomic'] >= 1_900_000
        if not execute:
            return
        _register_record_scanner(bridge,'mainnet')
        record = bridge.privacy.select_record(route.meta_str('remoteToken'),2_000_000)
        assert record_amount(record) == 2_000_000, 'Use an unspent two-USDCx record'
        proof = bridge.freezelist.exclusion_proof(bridge.aleo_address(),route.meta_str('remoteToken'))
        state.update(route=route.id,recipient=recipient,started=True,start_block=arc.w3.eth.block_number,
                     before=bridge.evm('arc').balance('arc/usdc',address=recipient))
        _save(path,state)
        bridge.execute(quote.plan,mode='private',record=record,merkle_proof=proof,on_checkpoint=save)
    assert state.get('source_tx_id')
    wait_for_aleo_transaction(bridge,state['source_tx_id'])
    tx_hash = wait_for(lambda: exact_arc_delivery(arc,recipient=recipient,start_block=state['start_block'],
                             before=state['before'],expected=state['expected_atomic']))
    state.update(destination_tx_id=tx_hash,done=True)
    _save(path,state)
    record_property('destination_tx_id',tx_hash)
