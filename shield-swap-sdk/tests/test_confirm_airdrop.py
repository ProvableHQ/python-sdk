"""Faucet confirmation behavior shared by sync and async clients."""
import asyncio
from unittest.mock import AsyncMock, Mock
import pytest
from aleo_shield_swap.api import ApiClient, AsyncApiClient
from aleo_shield_swap._api_models import AirdropJob, AirdropResult, AirdropStartResult
from aleo_shield_swap.errors import AirdropPendingError, AirdropRateLimitedError, DexApiError


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def faucet(request, monkeypatch):
    async_mode = request.param
    client = object.__new__(AsyncApiClient if async_mode else ApiClient)
    mock = AsyncMock if async_mode else Mock
    client.request_airdrop = mock(return_value=AirdropStartResult("job-1", "running"))
    client.get_airdrop_job = mock()
    monkeypatch.setattr("time.sleep", Mock())
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    def confirm(**kwargs):
        result = client.confirm_airdrop("aleo1test", **kwargs)
        return asyncio.run(result) if async_mode else result
    return client, confirm


def test_waits_for_settlement_and_preserves_token_failures(faucet):
    client, confirm = faucet
    failed = AirdropResult("token.aleo", "0", "failed", "USDCx", error="rejected")
    job = AirdropJob([failed], "complete", 1)
    client.get_airdrop_job.side_effect = [AirdropJob([], "running", 1), job]
    result = confirm(poll_interval=0)
    assert result.status == "settled" and result.job is job
    assert result.job.results[0].error == "rejected"
    client.request_airdrop.assert_called_once_with("aleo1test")
    assert client.get_airdrop_job.call_count == 2


def test_rate_limit_does_not_poll(faucet):
    client, confirm = faucet
    client.request_airdrop.side_effect = AirdropRateLimitedError("wait")
    result = confirm()
    assert result.status == "rate_limited" and result.message
    assert result.job is None
    client.get_airdrop_job.assert_not_called()


def test_timeout_retains_job_id(faucet):
    client, confirm = faucet
    client.get_airdrop_job.return_value = AirdropJob([], "running", 1)
    with pytest.raises(AirdropPendingError) as caught:
        confirm(timeout=0)
    assert caught.value.job_id == "job-1"


def test_already_settled_job_returns_immediately(faucet):
    client, confirm = faucet
    client.get_airdrop_job.return_value = AirdropJob([], "complete", 0)
    assert confirm(timeout=0).status == "settled"
    client.get_airdrop_job.assert_called_once_with("job-1")


@pytest.mark.parametrize("stage", ["request_airdrop", "get_airdrop_job"])
def test_other_api_errors_propagate(faucet, stage):
    client, confirm = faucet
    error = DexApiError(503, "unavailable")
    getattr(client, stage).side_effect = error
    with pytest.raises(DexApiError) as caught:
        confirm()
    assert caught.value is error


def test_poll_rate_limit_propagates(faucet):
    client, confirm = faucet
    client.get_airdrop_job.side_effect = AirdropRateLimitedError("poll limit")
    with pytest.raises(AirdropRateLimitedError):
        confirm()


@pytest.fixture(params=[False, True], ids=['sync', 'async'])
def funded_client(request, monkeypatch):
    from aleo_shield_swap import ShieldSwap, AsyncShieldSwap, ConfirmAirdropResult
    from .conftest import StubAleo
    from .test_async_client import AsyncStubAleo
    is_async = request.param
    mock = AsyncMock if is_async else Mock
    stub = AsyncStubAleo() if is_async else StubAleo()
    dex = AsyncShieldSwap(stub) if is_async else ShieldSwap(stub)
    job = AirdropJob([AirdropResult('token.aleo', '1', 'accepted', 'USDCx', tx_id='at1funding')], 'complete', 1)
    dex.api.confirm_airdrop = mock(return_value=ConfirmAirdropResult('settled', job=job))
    stub.record_provider.find = mock()
    monkeypatch.setattr('time.sleep', Mock())
    monkeypatch.setattr('asyncio.sleep', AsyncMock())
    def confirm(**kwargs):
        result = dex.confirm_airdrop(**kwargs)
        return asyncio.run(result) if is_async else result
    return dex, stub, confirm


def test_account_airdrop_waits_for_its_own_decrypted_record(funded_client):
    from .conftest import RECORD_TEXT
    dex, stub, confirm = funded_client
    stub.record_provider.find.side_effect = [
        [{'transaction_id': 'at1older', 'record_plaintext': RECORD_TEXT}],
        [{'transaction_id': 'at1funding', 'record_plaintext': None}],
        [{'transaction_id': 'at1funding    ', 'record_plaintext': RECORD_TEXT}],
    ]
    assert confirm(poll_interval=0).success
    assert stub.record_provider.find.call_count == 3
    dex.api.confirm_airdrop.assert_called_once()


def test_account_airdrop_record_timeout_preserves_failure_details(funded_client):
    dex, stub, confirm = funded_client
    stub.record_provider.find.return_value = []
    result = confirm(timeout=0)
    assert not result.success
    assert 'at1funding' in result.error
    assert 'record' in result.error.lower()
    dex.api.confirm_airdrop.assert_called_once()


def test_failed_airdrop_does_not_scan(funded_client):
    from aleo_shield_swap import ConfirmAirdropResult
    dex, stub, confirm = funded_client
    dex.api.confirm_airdrop.return_value = ConfirmAirdropResult('rate_limited', message='wait')
    assert not confirm().success
    stub.record_provider.find.assert_not_called()


def test_account_airdrop_scanner_error_propagates(funded_client):
    _, stub, confirm = funded_client
    stub.record_provider.find.side_effect = RuntimeError('scanner unavailable')
    with pytest.raises(RuntimeError, match='scanner unavailable'):
        confirm()


def test_all_airdrop_transfers_must_have_available_records(funded_client):
    from .conftest import RECORD_TEXT
    dex, stub, confirm = funded_client
    dex.api.confirm_airdrop.return_value.job.results.append(
        AirdropResult('other.aleo', '1', 'accepted', 'ETH', tx_id='at1second'))
    first = {'transaction_id': 'at1funding', 'record_plaintext': RECORD_TEXT}
    second = {'transaction_id': 'at1second', 'record_plaintext': RECORD_TEXT}
    stub.record_provider.find.side_effect = [[first], [second], [first, second]]
    assert confirm(poll_interval=0).success
    assert stub.record_provider.find.call_count == 3


def test_faucet_settlement_uses_the_same_timeout_as_scanning(funded_client, monkeypatch):
    dex, stub, confirm = funded_client
    clock = [0]
    monkeypatch.setattr('time.monotonic', lambda: clock[0])
    funding = dex.api.confirm_airdrop.return_value
    def settled(*args, **kwargs):
        clock[0] = 10
        return funding
    dex.api.confirm_airdrop.side_effect = settled
    stub.record_provider.find.return_value = []
    assert not confirm(timeout=10).success
    stub.record_provider.find.assert_called_once()
