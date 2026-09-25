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
