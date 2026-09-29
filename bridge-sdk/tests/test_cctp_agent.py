from aleo_bridge.agent import bridge_tools, dispatch_tool
from tests.fakes.fake_cctp import Harness, SENDER, RECIPIENT


def test_cctp_schemas_and_quote_dispatch():
    tools = {t['name']:t['input_schema'] for t in bridge_tools(include_writes=True)}
    assert 'cctp' in tools['bridge_quote']['properties']
    assert 'manual_mint' in tools['bridge_complete']['properties']
    assert 'approval_replacement' in tools['bridge_get_progress']['properties']
    h = Harness()
    result = dispatch_tool(h.bridge, 'bridge_quote', {'source_chain':'ethereum','source_asset':'usdc',
             'destination_chain':'arc','amount':'5','sender':SENDER,'recipient':RECIPIENT,
             'cctp':{'speed':'fast','forwarding':True,'max_fee':'0.1'}})
    assert result['kind'] == 'evm-cctp'
    assert result['plan']['cctp']['max_fee'] == '0.1'
    assert not h.source.sent


def test_manual_mint_confirm_gate_needs_no_aleo_secret():
    h = Harness(forwarding=False)
    h.destination.used = False
    h.circle.forward_hash = None
    h.execute()
    args = {'checkpoint':h.saved[-1].to_dict()}
    preview = dispatch_tool(h.bridge,'bridge_complete',args)
    assert 'error' not in preview
    assert not h.destination.sent
    result = dispatch_tool(h.bridge,'bridge_complete',{**args,'confirm':True})
    assert 'error' not in result
    assert len(h.destination.sent) == 1
    assert 'attestation' not in result['progress']['receipt']['protocol_state']
    assert 'message' not in result['progress']['receipt']['protocol_state']


def test_store_failure_returns_the_broadcast_checkpoint_to_agent():
    from types import SimpleNamespace
    from unittest.mock import Mock
    h = Harness(forwarding=False)
    h.destination.used = False
    h.circle.forward_hash = None
    h.execute()
    h.bridge.checkpoints = SimpleNamespace(save=Mock(side_effect=[None,OSError('disk unavailable')]))
    result = dispatch_tool(h.bridge,'bridge_complete',{'checkpoint':h.saved[-1].to_dict(),'confirm':True})
    assert result['checkpoint']['destination']['transactionId'] == h.destination.hash_at(1)
    assert result['next'] == 'recover'
