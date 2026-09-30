import pytest

from aleo_bridge import Receipt, Status
from aleo_bridge.errors import BridgeError
from tests.fakes.fake_cctp import Harness, DEST_HASH


@pytest.mark.parametrize('failure', ['mint_amount', 'missing_mint', 'missing_message', 'nonce', 'source_message', 'attested_fee', 'recipient'])
def test_unrelated_or_invalid_evidence_never_completes(failure):
    h = Harness()
    if failure == 'mint_amount': h.destination.receipts[DEST_HASH]['logs'] = h.destination_logs(amount=1)
    if failure == 'missing_mint': h.destination.receipts[DEST_HASH]['logs'].pop()
    if failure == 'missing_message': h.destination.receipts[DEST_HASH]['logs'].pop(0)
    if failure == 'nonce': h.destination.used = False
    if failure == 'source_message': h.burned = h.burned[:216]+(1).to_bytes(32,'big')+h.burned[248:]
    if failure == 'attested_fee': h.attested = h.attested[:312]+(100001).to_bytes(32,'big')+h.attested[344:]
    if failure == 'recipient': h.attested = h.attested[:184]+bytes(32)+h.attested[216:]
    if failure == 'recipient':
        # Veil filters unrelated attestation messages, including those from a batch.
        assert h.execute().receipt.status == Status.ATTESTATION_PENDING
    elif failure == 'nonce':
        assert h.execute().next == 'wait'
    else:
        with pytest.raises(BridgeError): h.execute()


def test_used_nonce_without_destination_proof_stays_pending():
    h = Harness()
    h.circle.forward_hash = None
    p = h.execute()
    assert p.receipt.status == Status.DELIVERY_PENDING
    assert h.destination.log_filters
    assert all(f['topics'][2] == '0x'+(123).to_bytes(32,'big').hex() for f in h.destination.log_filters)


def test_missing_forward_hash_recovers_from_matching_event():
    h = Harness()
    h.circle.forward_hash = None
    h.destination.history_logs = h.destination_logs()
    assert h.execute().next == 'done'


@pytest.mark.parametrize('head', [0, 999, 1000, 15000])
def test_scan_is_bounded_and_covers_contiguous_ranges(head):
    h = Harness()
    h.circle.forward_hash = None
    h.destination.block_number = head
    assert h.execute().next == 'wait'
    spans = [(int(f['fromBlock'],16),int(f['toBlock'],16)) for f in h.destination.log_filters]
    assert spans[0][1] == head and spans[-1][0] == max(0,head-9999)
    assert all(end-start < 1000 for start,end in spans)
    assert all(spans[i+1][1] == spans[i][0]-1 for i in range(len(spans)-1))


def test_terminal_receipt_is_not_polled():
    h = Harness()
    p = h.execute()
    h.source.methods.clear(); h.destination.methods.clear(); h.circle.urls.clear()
    assert h.bridge.get_status(p.plan,p.receipt) is p.receipt
    assert h.source.methods == h.destination.methods == h.circle.urls == []


def test_reverted_burn_recovery_fails_without_mint():
    h = Harness()
    h.source.pending_nth.add(1)
    p = h.execute()
    h.source.pending.clear()
    h.source.reverted.add(p.receipt.source_tx_id)
    assert h.bridge.get_status(p.plan, p.receipt).status == Status.FAILED
    assert h.destination.sent == []


@pytest.mark.parametrize('variant', ['pending', 'duplicate', 'wrong_hash', 'rpc_error'])
def test_attestation_response_and_transport_failures(variant):
    h = Harness()
    if variant == 'pending':
        h.circle.attestation_status = 'pending_confirmations'
        assert h.execute().receipt.status == Status.ATTESTATION_PENDING
        return
    old_json = h.circle.json
    def response():
        result = old_json()
        if '/fees/' in h.circle.urls[-1]: return result
        if variant == 'rpc_error': raise ConnectionError('offline')
        if variant == 'duplicate': result['messages'] *= 2
        else: result['sourceTxHash'] = DEST_HASH
        return result
    h.circle.json = response
    with pytest.raises((BridgeError, ConnectionError)): h.execute()
    assert len(h.source.sent) == 1


def test_reverted_forwarding_permits_manual_action():
    h = Harness()
    h.destination.used = False
    h.destination.receipts[DEST_HASH]['status'] = '0x0'
    p = h.execute()
    assert p.next == 'complete'
    assert p.receipt.destination_tx_id is None


@pytest.mark.parametrize('failure', ['connection', 'timeout'])
def test_wait_retries_circle_transport_failure_without_resubmission(failure):
    import requests
    h = Harness()
    h.circle.attestation_status = 'pending'
    progress = h.execute()
    h.circle.attestation_status = 'complete'
    original = h.circle.get
    calls = []

    def get(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            cls = requests.ConnectionError if failure == 'connection' else requests.Timeout
            raise cls('temporary transport failure')
        return original(*args, **kwargs)

    h.circle.get = get
    result = h.bridge.wait(progress, poll_seconds=0.001, timeout_seconds=2)
    assert result.next == 'done'
    assert len(calls) == 2 and len(h.source.sent) == 1


def test_wait_does_not_retry_malformed_circle_json():
    import requests
    h = Harness()
    h.circle.attestation_status = 'pending'
    progress = h.execute()
    calls = []

    def invalid_json():
        calls.append(1)
        raise requests.exceptions.JSONDecodeError('invalid JSON', 'not json', 0)

    h.circle.json = invalid_json
    with pytest.raises(BridgeError, match='Circle CCTP request failed'):
        h.bridge.wait(progress, poll_seconds=0.001, timeout_seconds=2)
    assert len(calls) == 1 and len(h.source.sent) == 1


@pytest.mark.parametrize('failure', [429, 500, 503, 'chunked'])
def test_wait_retries_circle_service_failures(failure):
    import requests
    h = Harness()
    h.circle.attestation_status = 'pending'
    progress = h.execute()
    h.circle.attestation_status = 'complete'
    original = h.circle.get
    calls = []
    def get(*args, **kwargs):
        calls.append(1)
        h.circle.status_code = 200
        if len(calls) == 1:
            if failure == 'chunked':
                raise requests.exceptions.ChunkedEncodingError('truncated')
            h.circle.status_code = failure
        return original(*args, **kwargs)
    h.circle.get = get
    result = h.bridge.wait(progress, poll_seconds=0.001, timeout_seconds=2)
    assert result.next == 'done'
    assert len(calls) == 2 and len(h.source.sent) == 1


def test_old_delivery_is_found_across_bounded_polls():
    h = Harness()
    h.circle.forward_hash = None
    h.destination.block_number = 25000
    h.destination.history_logs = h.destination_logs()
    p = h.execute()
    assert p.next == 'wait'
    assert len(h.destination.log_filters) <= 10
    for _ in range(3):
        before = len(h.destination.log_filters)
        receipt = h.bridge.get_status(p.plan, p.receipt)
        assert len(h.destination.log_filters) - before <= 10
        if receipt.status == Status.COMPLETED:
            break
    assert receipt.status == Status.COMPLETED
    assert receipt.destination_tx_id == DEST_HASH
    assert len(h.source.sent) == 1 and not h.destination.sent


def test_partial_destination_scan_discards_progress_after_reorg():
    h = Harness()
    h.circle.forward_hash = None
    h.destination.block_number = 25000
    p = h.execute()
    eth = h.bridge.evm('arc').conn.w3.eth
    original = eth.get_block
    def reorg(*args, **kwargs):
        block = dict(original(*args, **kwargs))
        block['hash'] = b'\xcd' * 32
        return block
    eth.get_block = reorg
    with pytest.raises(BridgeError, match='head block changed'):
        h.bridge.get_status(p.plan, p.receipt)
    before = len(h.destination.log_filters)
    assert h.bridge.get_status(p.plan, p.receipt).status == Status.DELIVERY_PENDING
    assert int(h.destination.log_filters[before]['toBlock'], 16) == 25000
    assert not h.destination.sent


@pytest.mark.parametrize('code', [400, 401, 403])
def test_circle_client_errors_are_not_retried(code):
    from aleo_bridge.lifecycle import _is_transient_error
    h = Harness()
    h.circle.status_code = code
    with pytest.raises(BridgeError) as caught:
        h.bridge.cctp._json('https://unused.invalid')
    assert not _is_transient_error(caught.value)
