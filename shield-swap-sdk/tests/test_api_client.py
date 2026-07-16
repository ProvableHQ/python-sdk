import json
from pathlib import Path

import pytest

from aleo_shield_swap.api import ApiClient
from aleo_shield_swap.errors import DexApiError

POOLS = json.loads((Path(__file__).parent / "fixtures" / "pools_response.json").read_text())


class _Resp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append(("GET", url, params, headers))
        return self.responses.pop(0)

    def post(self, url, json=None, timeout=None, headers=None):
        self.calls.append(("POST", url, json, headers))
        return self.responses.pop(0)


def test_get_pools_parses_models_and_token_info():
    s = _Session([_Resp(200, POOLS)])
    pools = ApiClient(base_url="https://x", session=s).get_pools()
    assert s.calls[0][1] == "https://x/pools"
    entry = pools[0]
    assert entry.key.endswith("field")            # delegation to PoolStateDoc
    assert entry.token0_info.wrapper_program.endswith(".aleo")


def test_get_route_stringifies_amount():
    payload = {"data": {"hops": [], "token_in": "1field", "token_out": "2field",
                        "estimated_amount_out": "42.5"}}
    s = _Session([_Resp(200, payload)])
    route = ApiClient(base_url="https://x", session=s).get_route(
        token_in="1field", token_out="2field", amount_in=10**18)
    assert route.estimated_amount_out == "42.5"
    _, url, params, _ = s.calls[0]
    assert url == "https://x/route"
    assert params == {"token_in": "1field", "token_out": "2field",
                      "amount_in": str(10**18)}


def test_get_public_balances_passes_user():
    payload = {"data": [{"balance": "5", "decimals": 6, "name": "USDCx",
                         "symbol": "wUSDCx", "token_address": "4field",
                         "token_id": "4field", "extra_field": "ignored"}]}
    s = _Session([_Resp(200, payload)])
    bals = ApiClient(base_url="https://x", session=s).get_public_balances("aleo1me")
    assert bals[0].balance == "5"
    assert s.calls[0][2] == {"user": "aleo1me"}


def test_non_2xx_raises_dex_api_error():
    s = _Session([_Resp(503, {"error": "down"})])
    with pytest.raises(DexApiError) as ei:
        ApiClient(base_url="https://x", session=s).get_pools()
    assert ei.value.status == 503


def test_authenticate_stores_bearer_token():
    s = _Session([
        _Resp(200, {"data": {"message": "sign me"}}),
        _Resp(200, {"data": {"token": "jwt-abc"}}),
        _Resp(200, {"data": []}),
    ])
    client = ApiClient(base_url="https://x", session=s)
    signed = []
    token = client.authenticate("aleo1me", lambda msg: signed.append(msg) or "sign1xyz")
    assert token == "jwt-abc" and signed == ["sign me"]
    method, url, body, _ = s.calls[1]
    assert (method, url) == ("POST", "https://x/auth/verify")
    assert body == {"address": "aleo1me", "signature": "sign1xyz"}
    client.get_public_balances("aleo1me")
    headers = s.calls[2][3]
    assert headers["authorization"] == "Bearer jwt-abc"


def test_401_maps_to_not_authenticated():
    from aleo_shield_swap.errors import NotAuthenticatedError
    s = _Session([_Resp(401, {"error": "missing token"})])
    with pytest.raises(NotAuthenticatedError):
        ApiClient(base_url="https://x", session=s)._get("/access/status")


def test_403_invite_maps_to_not_redeemed():
    from aleo_shield_swap.errors import NotRedeemedError
    s = _Session([_Resp(403, {"error": "redeem an invite code to unlock access"})])
    with pytest.raises(NotRedeemedError):
        ApiClient(base_url="https://x", session=s, token="t")._get("/route")


def test_other_403_stays_dex_api_error():
    s = _Session([_Resp(403, {"error": "forbidden for another reason"})])
    with pytest.raises(DexApiError):
        ApiClient(base_url="https://x", session=s, token="t")._get("/route")


def _lifecycle_client(*resps, token="t"):
    s = _Session(list(resps))
    return ApiClient(base_url="https://x", session=s, token=token), s


def test_access_status():
    api, s = _lifecycle_client(_Resp(200, {"data": {"has_access": True}}))
    assert api.access_status().has_access is True
    assert s.calls[0][:2] == ("GET", "https://x/access/status")


def test_redeem_code_adopts_new_token():
    api, s = _lifecycle_client(_Resp(200, {"data": {"code": "C", "status": "redeemed",
                                                    "token": "fresh-jwt"}}))
    out = api.redeem_code("C")
    assert out.status == "redeemed"
    assert s.calls[0][2] == {"code": "C"}
    assert api._token == "fresh-jwt"


def test_request_airdrop_and_poll():
    api, s = _lifecycle_client(
        _Resp(200, {"data": {"job_id": "j1", "status": "running"}}),
        _Resp(200, {"data": {"status": "complete", "total": 3, "results": [
            {"symbol": "wALEO", "wrapper_program": "waleo.aleo",
             "amount": "1000000", "status": "accepted",
             "tx_id": "at1...", "error": None}]}}),
    )
    start = api.request_airdrop("aleo1abc")
    assert (start.job_id, start.status) == ("j1", "running")
    assert s.calls[0][2] == {"address": "aleo1abc"}
    job = api.get_airdrop_job("j1")
    assert job.status == "complete" and job.results[0].symbol == "wALEO"
    assert s.calls[1][1] == "https://x/airdrop/j1"


def test_airdrop_429_maps_to_rate_limited():
    from aleo_shield_swap.errors import AirdropRateLimitedError
    api, _ = _lifecycle_client(_Resp(429, {"error": "already claimed"}))
    with pytest.raises(AirdropRateLimitedError):
        api.request_airdrop("aleo1abc")


def test_create_api_token():
    api, s = _lifecycle_client(_Resp(200, {"data": {
        "id": "u1", "name": "stress", "token": "sk-live", "token_prefix": "sk",
        "created_at": "2026-07-15", "expires_at": None}}))
    out = api.create_api_token("stress", expires_in_days=30)
    assert out.token == "sk-live"
    assert s.calls[0][1] == "https://x/api-tokens"
    assert s.calls[0][2] == {"name": "stress", "expires_in_days": 30}


class _AsyncResp(_Resp):
    pass


class _AsyncClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def get(self, url, params=None, headers=None):
        self.calls.append(("GET", url, params, headers))
        return self.responses.pop(0)

    async def post(self, url, json=None, headers=None):
        self.calls.append(("POST", url, json, headers))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_async_lifecycle_endpoints():
    from aleo_shield_swap.api import AsyncApiClient
    from aleo_shield_swap.errors import AirdropRateLimitedError
    c = _AsyncClient([
        _AsyncResp(200, {"data": {"has_access": False}}),
        _AsyncResp(200, {"data": {"code": "C", "status": "redeemed", "token": "t2"}}),
        _AsyncResp(200, {"data": {"job_id": "j1", "status": "running"}}),
        _AsyncResp(200, {"data": {"status": "complete", "total": 1, "results": [
            {"symbol": "wETH", "wrapper_program": "weth.aleo",
             "amount": "5", "status": "accepted"}]}}),
        _AsyncResp(429, {"error": "already claimed"}),
    ])
    api = AsyncApiClient(base_url="https://x", client=c, token="t")
    assert (await api.access_status()).has_access is False
    assert (await api.redeem_code("C")).token == "t2"
    assert api._token == "t2"
    assert (await api.request_airdrop("aleo1a")).job_id == "j1"
    job = await api.get_airdrop_job("j1")
    assert job.results[0].wrapper_program == "weth.aleo"
    with pytest.raises(AirdropRateLimitedError):
        await api.request_airdrop("aleo1a")
