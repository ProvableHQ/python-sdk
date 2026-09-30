from dataclasses import replace

import pytest
from eth_abi import decode
from eth_utils import keccak

from aleo_bridge import Status, create_checkpoint
from aleo_bridge.errors import BridgeError
from tests.fakes.fake_cctp import Harness, SENDER, DEST_HASH


def test_pending_burn_checkpoint_roundtrip_never_resends():
    h = Harness()
    h.source.pending_nth.add(1)
    p = h.execute()
    recovered = h.bridge.recover(h.saved[-1].to_json())
    assert recovered.plan.cctp == p.plan.cctp
    assert recovered.receipt.status == Status.SOURCE_CONFIRMING
    assert len(h.source.sent) == 1
    h.source.pending.clear()
    assert h.bridge.recover(h.saved[-1]).next == 'done'
    assert len(h.source.sent) == 1


def test_approval_recovery_and_resume_burn_once():
    h = Harness(allowance=0)
    h.source.pending_nth.add(1)
    h.execute()
    cp = h.saved[-1]
    assert h.bridge.recover(cp).receipt.status == Status.SOURCE_APPROVAL_PENDING
    h.source.pending.clear()
    from web3 import Web3
    h.source.allowances[tuple(Web3.to_checksum_address(v) for v in (h.source_token, SENDER, h.messenger))] = 5_000_000
    recovered = h.bridge.recover(cp)
    assert recovered.next == 'resume'
    result = h.bridge.resume(recovered, timeout_seconds=0, on_checkpoint=h.saved.append)
    assert result.next == 'done'
    assert len(h.source.sent) == 2
    assert h.saved[-1].intent['cctp']['max_fee'] == '0.1'


def test_manual_completion_and_second_call_do_not_resend():
    h = Harness(forwarding=False)
    h.destination.used = False
    h.circle.forward_hash = None
    p = h.execute()
    assert p.next == 'complete'
    minted = h.bridge.complete(p, on_checkpoint=h.saved.append)
    assert minted.receipt.status == Status.DESTINATION_CONFIRMING
    data = bytes.fromhex(h.destination.sent[0]['data'][2:])
    assert data[:4] == keccak(text='receiveMessage(bytes,bytes)')[:4]
    assert decode(['bytes','bytes'],data[4:]) == (h.attested, bytes.fromhex('abcd'))
    assert h.saved[-1].destination['transactionId'] == minted.receipt.destination_tx_id
    assert 'message' not in h.saved[-1].to_json()
    h.bridge.complete(minted)
    assert len(h.destination.sent) == 1
    h.destination.used = True
    assert h.bridge.recover(h.saved[-1]).next == 'done'


def test_forwarding_fallback_requires_explicit_manual_mint():
    h = Harness()
    h.circle.forward_hash = None
    h.destination.used = False
    p = h.execute()
    with pytest.raises(BridgeError): h.bridge.complete(p)
    assert not h.destination.sent
    assert h.bridge.complete(p, manual_mint=True).receipt.status == Status.DESTINATION_CONFIRMING


def test_used_nonce_cannot_be_manually_minted_without_evidence():
    h = Harness()
    h.circle.forward_hash = None
    with pytest.raises(BridgeError): h.bridge.complete(h.execute(), manual_mint=True)
    assert not h.destination.sent


def test_lost_destination_response_keeps_checkpoint():
    h = Harness(forwarding=False)
    h.circle.forward_hash = None
    h.destination.used = False
    h.destination.send_errors[1] = 'response lost'
    with pytest.raises(BridgeError) as caught:
        h.bridge.complete(h.execute(), on_checkpoint=h.saved.append)
    assert h.saved[-1].destination['transactionId'] == caught.value.broadcast_id


@pytest.mark.parametrize('field', ['sender','amount','spender','value'])
def test_approval_recovery_rejects_mismatched_transaction(field):
    h = Harness(allowance=0)
    h.source.pending_nth.add(1)
    h.execute()
    cp = h.saved[-1]
    h.source.pending.clear()
    tx = h.source.sent[0]
    if field == 'sender': tx['from'] = '0x'+'33'*20
    elif field == 'value': tx['value'] = 1
    else:
        from eth_abi import encode
        tx['data'] = '0x'+(keccak(text='approve(address,uint256)')[:4] + encode(['address','uint256'],
              ['0x'+'33'*20 if field == 'spender' else h.messenger, 1 if field == 'amount' else 5_000_000])).hex()
    with pytest.raises(BridgeError): h.bridge.recover(cp)
    assert len(h.source.sent) == 1


def replacement_case():
    from eth_abi import encode
    h = Harness(allowance=0)
    h.source.pending_nth.add(1)
    h.execute()
    cp = h.saved[-1]
    original = cp.source['approvalTransactionIds'][0]
    replacement = '0x'+'44'*32
    h.source.tx_not_found.add(original)
    h.source.add_transaction(replacement, sender=SENDER, to=h.source_token)
    h.source.transactions[replacement]['input'] = '0x'+(keccak(text='approve(address,uint256)')[:4]+
                     encode(['address','uint256'], [h.messenger,5_000_000])).hex()
    h.source.add_receipt(replacement)
    return h, cp, {'original_transaction_id': original, 'replacement_transaction_id': replacement}


def test_explicit_approval_replacement_preserves_history_and_fee_cap():
    h, cp, selection = replacement_case()
    recovered = h.bridge.recover(cp, approval_replacement=selection)
    assert recovered.next == 'resume'
    saved = create_checkpoint(recovered.plan, recovered.receipt, h.bridge.registry)
    assert saved.source['approvalTransactionIds'] == [selection['replacement_transaction_id']]
    assert saved.source['replacedApprovalTransactionIds'] == [selection['original_transaction_id']]
    assert saved.intent['cctp']['max_fee'] == '0.1'
    assert len(h.source.sent) == 1


@pytest.mark.parametrize('failure',['visible','same','not_saved','pending','bad_amount','after_burn'])
def test_replacement_requires_absent_original_and_bound_confirmed_approval(failure):
    h, cp, selection = replacement_case()
    if failure == 'visible': h.source.tx_not_found.clear()
    if failure == 'same': selection['replacement_transaction_id'] = selection['original_transaction_id']
    if failure == 'not_saved': selection['original_transaction_id'] = DEST_HASH
    if failure == 'pending': h.source.pending.add(selection['replacement_transaction_id'])
    if failure == 'bad_amount': h.source.transactions[selection['replacement_transaction_id']]['input'] = '0x'
    if failure == 'after_burn': cp = replace(cp, source={**cp.source,'transactionId':DEST_HASH})
    with pytest.raises(BridgeError): h.bridge.recover(cp, approval_replacement=selection)
    assert len(h.source.sent) == 1


@pytest.mark.parametrize('failure',['gas','signer'])
def test_manual_destination_requires_wallet_and_gas(failure):
    h = Harness(forwarding=False)
    h.destination.used = False
    h.circle.forward_hash = None
    p = h.execute()
    if failure == 'gas': h.destination.eth_balances.clear()
    else:
        from aleo_bridge import Ethereum
        from web3 import Web3
        h.bridge.evm('arc').conn = Ethereum(w3=Web3(h.destination))
    with pytest.raises(BridgeError): h.bridge.complete(p)
    assert not h.destination.sent


def test_offline_checkpoint_reconstruction_excludes_attestation():
    from aleo_bridge.lifecycle import progress_from_checkpoint
    h = Harness(forwarding=False)
    h.circle.forward_hash = None
    h.destination.used = False
    p = h.execute()
    cp = create_checkpoint(p.plan,p.receipt,h.bridge.registry)
    h.source.methods.clear(); h.destination.methods.clear(); h.circle.urls.clear()
    offline = progress_from_checkpoint(h.bridge.registry,cp)
    assert offline.plan.cctp == p.plan.cctp
    assert 'attestation' not in cp.to_json()
    assert h.source.methods == h.destination.methods == h.circle.urls == []


def test_callback_failure_preserves_burn_in_store_and_error(tmp_path):
    from aleo_bridge import FileCheckpointStore
    store = FileCheckpointStore(tmp_path)
    h = Harness(allowance=0,checkpoints=store)
    def callback(cp):
        if cp.source.get('transactionId'):
            raise OSError('callback disk unavailable')
    with pytest.raises(BridgeError) as caught:
        h.bridge.execute(h.plan,on_checkpoint=callback,timeout_seconds=0)
    assert caught.value.broadcast_id == h.source.hash_at(2)
    assert caught.value.checkpoint.source['transactionId'] == h.source.hash_at(2)
    assert any(cp.source.get('transactionId') == h.source.hash_at(2) for cp in store.list())
    assert h.bridge.recover(caught.value.checkpoint).next == 'done'
    assert len(h.source.sent) == 2


def test_stale_approval_recovers_existing_burn_before_resuming():
    h = Harness(allowance=0)
    h.execute()
    approval = h.saved[0]
    burn = h.source.hash_at(2)
    h.source.history_logs = h.source_logs(burn)
    h.source.history_logs[0]['blockNumber'] = hex(101)
    # Receipt and event providers now share a consistent mined head.
    h.source.block_number = 101
    recovered = h.bridge.recover(approval)
    assert recovered.next == 'done'
    assert recovered.receipt.source_tx_id == burn
    assert len(h.source.sent) == 2


def test_stale_approval_with_unresolved_later_nonce_cannot_resume():
    h = Harness(allowance=0)
    h.source.pending_nth.add(2)
    h.execute()
    h.source.nonce_latest = 1  # Only the approval is mined; the later burn remains pending.
    recovered = h.bridge.recover(h.saved[0])
    assert recovered.next == 'wait'
    assert recovered.receipt.status == Status.SOURCE_APPROVAL_PENDING
    assert len(h.source.sent) == 2


def test_confirmed_unrelated_activity_after_approval_does_not_block_resume():
    h = Harness(allowance=0)
    h.source.pending_nth.add(1)
    h.execute()
    h.source.pending.clear()
    h.bridge.evm('ethereum').conn.send_transaction({'to': '0x0000000000000000000000000000000000000022', 'value': 0})
    h.source.nonce_latest = 2
    result = h.bridge.recover(h.saved[0])
    assert result.next == 'resume'
    assert result.receipt.source_tx_id is None
    assert len(h.source.sent) == 2  # Recovery itself cannot send a burn.


def test_activity_after_scanned_head_still_blocks_approval_resume():
    h = Harness(allowance=0)
    h.source.pending_nth.add(1)
    h.execute()
    h.source.pending.clear()
    h.source.nonce_latest = 1
    h.source.nonce_pending = 2
    requested_tags = []
    eth = h.bridge.evm('ethereum').conn.w3.eth
    original = eth.get_transaction_count

    def track(address, block_identifier='latest'):
        requested_tags.append(block_identifier)
        return original(address, block_identifier)

    eth.get_transaction_count = track
    assert h.bridge.recover(h.saved[0]).next == 'wait'
    assert h.source.block_number in requested_tags


def test_approval_recovery_rejects_a_changed_scanned_head():
    h = Harness(allowance=0)
    h.source.pending_nth.add(1)
    h.execute()
    h.source.pending.clear()
    h.source.nonce_latest = h.source.nonce_pending = 2
    eth = h.bridge.evm('ethereum').conn.w3.eth
    original = eth.get_block
    calls = []

    def changed_head(*args, **kwargs):
        block = dict(original(*args, **kwargs))
        calls.append(1)
        block['hash'] = bytes([len(calls)]) * 32
        return block

    eth.get_block = changed_head
    with pytest.raises(BridgeError, match='head block changed'):
        h.bridge.recover(h.saved[0])
    assert len(h.source.sent) == 1


def test_approval_does_not_adopt_identical_burn_before_its_nonce():
    h = Harness(allowance=0)
    h.execute()
    cp = h.saved[0]
    approval,burn = h.source.sent
    # Model an older identical burn and a subsequent approval in the same block.
    approval['nonce'],burn['nonce'] = 2,1
    h.source.nonce_pending = 3
    h.source.nonce_latest = 3
    h.source.history_logs = h.source_logs(burn['hash'])
    h.source.history_logs[0]['blockNumber'] = hex(h.source.block_number)
    result = h.bridge.recover(cp)
    assert result.next == 'resume'
    assert result.receipt.source_tx_id is None


def test_old_approval_recovery_batches_history_before_allowing_resume():
    h = Harness(allowance=0)
    h.source.pending_nth.add(1)
    h.execute()
    cp = h.saved[0]
    h.source.pending.clear()
    # Pin the mined approval so advancing the provider head does not move it.
    h.source.receipts[h.source.hash_at(1)] = h.source._receipt(h.source.hash_at(1))
    h.source.block_number = 25100
    result = h.bridge.recover(cp)
    assert result.next == 'wait'
    assert len(h.source.log_filters) <= 10
    for _ in range(3):
        before = len(h.source.log_filters)
        result = h.bridge.recover(cp)
        assert len(h.source.log_filters) - before <= 10
        if result.next == 'resume':
            break
    assert result.next == 'resume'
    assert len(h.source.sent) == 1
    spans = [(int(f['fromBlock'],16),int(f['toBlock'],16)) for f in h.source.log_filters]
    assert spans[0][0] == 16 and spans[-1][1] == 25100
    assert all(spans[i+1][0] == spans[i][1]+1 for i in range(len(spans)-1))


def test_old_approval_finds_later_burn_without_repeating_it():
    h = Harness(allowance=0)
    h.execute()
    cp = h.saved[0]
    h.source.receipts[h.source.hash_at(1)] = h.source._receipt(h.source.hash_at(1))
    burn = h.source.hash_at(2)
    h.source.history_logs = h.source_logs(burn)
    h.source.history_logs[0]['blockNumber'] = hex(20100)
    h.source.block_number = 25100
    result = h.bridge.recover(cp)
    assert result.next == 'wait'
    for _ in range(3):
        result = h.bridge.recover(cp)
        if result.next == 'done':
            break
    assert result.next == 'done' and result.receipt.source_tx_id == burn
    assert len(h.source.sent) == 2


def test_partial_source_scan_discards_progress_after_reorg():
    h = Harness(allowance=0)
    h.source.pending_nth.add(1)
    h.execute()
    cp = h.saved[0]
    h.source.pending.clear()
    h.source.receipts[h.source.hash_at(1)] = h.source._receipt(h.source.hash_at(1))
    h.source.block_number = 25100
    assert h.bridge.recover(cp).next == 'wait'
    eth = h.bridge.evm('ethereum').conn.w3.eth
    original = eth.get_block
    def reorg(*args, **kwargs):
        block = dict(original(*args, **kwargs))
        block['hash'] = b'\xcd' * 32
        return block
    eth.get_block = reorg
    with pytest.raises(BridgeError, match='head block changed'):
        h.bridge.recover(cp)
    before = len(h.source.log_filters)
    assert h.bridge.recover(cp).next == 'wait'
    assert int(h.source.log_filters[before]['fromBlock'], 16) == 16
    assert len(h.source.sent) == 1


def test_new_client_restarts_partial_source_scan_safely():
    from aleo_bridge.cctp import CctpModule
    h = Harness(allowance=0)
    h.source.pending_nth.add(1)
    h.execute()
    cp = h.saved[0]
    h.source.pending.clear()
    h.source.receipts[h.source.hash_at(1)] = h.source._receipt(h.source.hash_at(1))
    h.source.block_number = 25100
    assert h.bridge.recover(cp).next == 'wait'
    # Exercise a new protocol module directly: no in-memory scan can be trusted
    # merely because a caller claims to have scanned earlier blocks.
    module = CctpModule(h.bridge)
    before = len(h.source.log_filters)
    state = module.recover(h.plan, cp)
    assert state.status == Status.SOURCE_APPROVAL_PENDING
    assert int(h.source.log_filters[before]['fromBlock'], 16) == 16
    assert len(h.source.sent) == 1


def test_extending_source_scan_rechecks_previous_anchor_before_resuming():
    h = Harness(allowance=0)
    h.execute()
    cp = h.saved[0]
    h.source.receipts[h.source.hash_at(1)] = h.source._receipt(h.source.hash_at(1))
    h.source.block_number = 100
    assert h.bridge.recover(cp).next == 'resume'
    h.source.block_number = 101
    eth = h.bridge.evm('ethereum').conn.w3.eth
    original = eth.get_block
    switched = [False]
    def reorg_on_extension(number, *args, **kwargs):
        block = dict(original(number, *args, **kwargs))
        if number > 100:
            switched[0] = True
            # The replacement chain contains a burn inside the cached prefix.
            h.source.history_logs = h.source_logs(h.source.hash_at(2))
            h.source.history_logs[0]['blockNumber'] = hex(90)
        if switched[0]:
            block['hash'] = b'\xcd' * 32
        return block
    eth.get_block = reorg_on_extension
    with pytest.raises(BridgeError, match='head block changed'):
        h.bridge.recover(cp)
    recovered = h.bridge.recover(cp)
    assert recovered.next == 'done'
    assert recovered.receipt.source_tx_id == h.source.hash_at(2)
    assert len(h.source.sent) == 2
