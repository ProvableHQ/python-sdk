import pytest
from tests.live import config


@pytest.mark.parametrize('case',['cctp-roundtrip','arc-journey','aleo-arc','evm-cctp','evm-xreserve','aleo-xreserve'])
def test_every_arc_funded_case_needs_all_gates(case):
    env = {config.FUNDS_VAR:'1',config.STATE_DIR_VAR:'/tmp/arc-live-state',
           config.MAINNET_ACK_VAR:config.MAINNET_ACK,config.MAINNET_CASES_VAR:case}
    assert config.mainnet_case_enabled(case,env)
    assert not config.mainnet_execution_enabled(env)
    for required in list(env):
        missing = {k:v for k,v in env.items() if k != required}
        assert not config.mainnet_case_enabled(case,missing)
    assert config.mainnet_execution_enabled({**env,config.MAINNET_EXECUTE_VAR:config.MAINNET_EXECUTE_ACK})


@pytest.mark.parametrize('failure',[None,'wrong_amount','old_block','reverted','wrong_balance','unrelated_event'])
def test_arc_live_delivery_requires_exact_recent_successful_mint(failure):
    from tests.live.test_arc_xreserve import exact_arc_delivery
    from tests.fakes.fake_cctp import Harness,RECIPIENT,DEST_HASH
    from web3 import Web3
    h = Harness()
    log = h.destination_logs(amount=1 if failure == 'wrong_amount' else 4_990_000)[1]
    log['blockNumber'] = hex(10 if failure == 'old_block' else 100)
    h.destination.history_logs = [log]
    h.destination.receipts[DEST_HASH]['logs'] = [log]
    if failure == 'unrelated_event':
        from tests.fakes.fake_web3 import event_log
        from eth_utils.crypto import keccak
        other = event_log(h.destination_token,['0x'+keccak(text='Mint(address,uint256)').hex()],
                          '0x',log_index=2,tx_hash=DEST_HASH)
        h.destination.receipts[DEST_HASH]['logs'].insert(0,other)
    if failure == 'reverted': h.destination.receipts[DEST_HASH]['status'] = '0x0'
    key = (Web3.to_checksum_address(h.destination_token),Web3.to_checksum_address(RECIPIENT))
    h.destination.token_balances[key] = 4_990_000 if failure != 'wrong_balance' else 1
    conn = h.bridge.evm('arc').conn
    if failure in ('reverted','wrong_balance'):
        with pytest.raises(AssertionError): exact_arc_delivery(conn,recipient=RECIPIENT,start_block=50,before=0,expected=4_990_000)
    else:
        result = exact_arc_delivery(conn,recipient=RECIPIENT,start_block=50,before=0,expected=4_990_000)
        assert result == (DEST_HASH if failure in (None,'unrelated_event') else None)


def test_arc_live_fee_refresh_preserves_budgeted_net_amount(tmp_path):
    import json
    from unittest.mock import Mock
    from tests.live.test_arc_xreserve import BudgetedFeeSession
    from aleo_bridge.errors import BridgeError
    state = {'started':True}
    response = Mock(status_code=200)
    response.json.return_value = {'withdrawalFeeBaseUnits':'20000'}
    session = BudgetedFeeSession(state,tmp_path/'state.json',Mock(post=Mock(return_value=response)))
    session.post('unused')
    assert json.loads((tmp_path/'state.json').read_text())['expected_atomic'] == 1_980_000
    response.json.return_value = {'withdrawalFeeBaseUnits':'100001'}
    with pytest.raises(BridgeError): session.post('unused')
