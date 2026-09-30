import pytest
from eth_abi import decode
from eth_utils import keccak

from aleo_bridge.errors import BridgeError
from aleo_bridge.types import Status
from tests.fakes.fake_cctp import Harness, SENDER


@pytest.mark.parametrize('other', ['ethereum', 'base', 'arbitrum'])
@pytest.mark.parametrize('reverse', [False, True])
def test_burn_and_exact_mint_on_all_six_routes(other, reverse):
    h = Harness('arc' if reverse else other, other if reverse else 'arc')
    progress = h.execute()
    assert progress.next == 'done'
    assert len(h.source.sent) == 1 and len(h.destination.sent) == 0
    raw = bytes.fromhex(h.source.sent[0]['data'][2:])
    assert raw[:4] == keccak(text='depositForBurnWithHook(uint256,uint32,bytes32,address,bytes32,uint256,uint32,bytes)')[:4]
    args = decode(['uint256','uint32','bytes32','address','bytes32','uint256','uint32','bytes'], raw[4:])
    assert args[0] == 5_000_000 and args[1] == (h.route.metadata['destinationDomain'])
    assert args[4] == bytes(32) and args[5:7] == (100_000, 1000)
    assert args[7] == b'cctp-forward'.ljust(24,b'\0') + bytes(8)
    assert h.saved[0].source['transactionId'] == progress.receipt.source_tx_id
    assert h.saved[0].intent['cctp']['max_fee'] == '0.1'


def test_approval_and_burn_checkpoint_before_pending_return():
    h = Harness(allowance=0)
    h.source.pending_nth.add(2)
    p = h.execute()
    assert p.receipt.status == Status.SOURCE_CONFIRMING
    assert len(h.source.sent) == 2
    assert h.saved[0].source['approvalTransactionIds'] == [h.source.hash_at(1)]
    assert h.saved[-1].source['transactionId'] == h.source.hash_at(2)


def test_pending_approval_does_not_burn():
    h = Harness(allowance=0)
    h.source.pending_nth.add(1)
    assert h.execute().receipt.status == Status.SOURCE_APPROVAL_PENDING
    assert len(h.source.sent) == 1


@pytest.mark.parametrize('failure', ['balance', 'gas', 'fee', 'chain'])
def test_preflight_failure_does_not_approve_or_burn(failure):
    h = Harness(allowance=0)
    if failure == 'balance': h.source.token_balances.clear()
    if failure == 'gas': h.source.eth_balances.clear()
    if failure == 'fee': h.circle.minimum = '1000'
    if failure == 'chain': h.source.chain_id = 99
    with pytest.raises(BridgeError): h.execute()
    assert h.source.sent == []


def test_lost_burn_response_keeps_signed_hash_in_checkpoint():
    h = Harness()
    h.source.send_errors[1] = 'response lost'
    with pytest.raises(BridgeError) as caught: h.execute()
    assert h.saved[-1].source['transactionId'] == caught.value.broadcast_id
    assert h.saved[-1].intent['sender'] == SENDER


def test_default_fee_budget_tolerates_a_tick_after_approval():
    from dataclasses import replace
    from aleo_bridge import CctpOptions
    h = Harness(allowance=0)
    h.plan = replace(h.plan, cctp=CctpOptions('fast', True))
    h.source.pending_nth.add(2)
    original = h.circle.json
    reads = []
    def fee_tick():
        if '/fees/' in h.circle.urls[-1]:
            reads.append(1)
            if len(reads) == 3:
                h.circle.forward += 1
        return original()
    h.circle.json = fee_tick
    p = h.execute()
    assert p.receipt.status == Status.SOURCE_CONFIRMING
    assert len(h.source.sent) == 2
    assert p.plan.cctp.max_fee == '0.001815'
    assert h.saved[-1].intent['cctp']['max_fee'] == '0.001815'


def test_fee_above_approved_cap_returns_resumable_progress_without_reapproval():
    from dataclasses import replace
    from aleo_bridge import CctpOptions
    h = Harness(allowance=0)
    h.plan = replace(h.plan, cctp=CctpOptions('fast', True, '0.00165'))
    original = h.circle.json
    def fee_tick():
        if len(h.source.sent) == 1:
            h.circle.forward = 1001
        return original()
    h.circle.json = fee_tick
    p = h.execute()
    assert p.next == 'resume'
    assert p.receipt.source_tx_id is None and len(h.source.sent) == 1
    assert 'max_fee' in p.receipt.protocol_state['sourceError']
    from web3 import Web3
    h.source.allowances[tuple(Web3.to_checksum_address(v) for v in (h.source_token, SENDER, h.messenger))] = 5_000_000
    still_blocked = h.bridge.resume(p, timeout_seconds=0)
    assert still_blocked.next == 'resume' and len(h.source.sent) == 1
    assert still_blocked.plan.cctp.max_fee == '0.00165'
    h.circle.json = original
    h.circle.forward = 1000
    h.source.pending_nth.add(2)
    result = h.bridge.resume(still_blocked, timeout_seconds=0)
    assert result.receipt.status == Status.SOURCE_CONFIRMING
    assert len(h.source.sent) == 2
    assert h.saved[0].intent['cctp']['max_fee'] == '0.00165'
