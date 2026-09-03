import json
from pathlib import Path

import pytest

from aleo_shield_swap.api import ApiClient, AsyncApiClient
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
    assert entry.token0_info.amm_token_program.endswith(".aleo")


def test_get_route_stringifies_amount():
    payload = {"data": {"hops": [], "token_in": "1field", "token_out": "2field",
                        "protocol_revision": 1, "estimated_amount_out": "42.5"}}
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


def test_403_is_a_plain_dex_api_error():
    # Access is no longer invite-gated, so there is no "not redeemed" error
    # class to map a 403 onto — any 403 surfaces with its status and body.
    import aleo_shield_swap.errors as errors
    assert not hasattr(errors, "NotRedeemedError")
    s = _Session([_Resp(403, {"error": "redeem an invite code to unlock access"})])
    with pytest.raises(DexApiError) as exc:
        ApiClient(base_url="https://x", session=s, token="t")._get("/route")
    assert type(exc.value) is DexApiError and exc.value.status == 403


def _lifecycle_client(*resps, token="t"):
    s = _Session(list(resps))
    return ApiClient(base_url="https://x", session=s, token=token), s


def test_access_status():
    api, s = _lifecycle_client(_Resp(200, {"data": {"has_access": True}}))
    assert api.access_status().has_access is True
    assert s.calls[0][:2] == ("GET", "https://x/access/status")


def test_redeem_code_targets_referral_endpoint():
    # Referral codes are the only kind of code that exists now.
    api, s = _lifecycle_client(_Resp(200, {"data": {"code": "C", "status": "redeemed"}}))
    out = api.redeem_code("C")
    assert out.status == "redeemed"
    assert s.calls[0][1] == "https://x/referral/redeem"
    assert s.calls[0][2] == {"code": "C"}
    assert api._token == "t"          # sessions moved to /auth/* — no rotation


def test_referral_status():
    api, s = _lifecycle_client(_Resp(200, {"data": {
        "has_access": True, "code": None, "referred_by": None,
        "my_code": "HRBH9UVEDR"}}))
    st = api.referral_status()
    assert st.has_access is True and st.referred_by is None
    assert st.my_code == "HRBH9UVEDR"
    assert s.calls[0][:2] == ("GET", "https://x/referral/status")


def test_my_referral_code():
    # Every authenticated account owns a code to share; None until issued.
    api, s = _lifecycle_client(_Resp(200, {"data": {"code": "HRBH9UVEDR"}}))
    assert api.my_referral_code() == "HRBH9UVEDR"
    assert s.calls[0][:2] == ("GET", "https://x/referral/my-code")
    api, _ = _lifecycle_client(_Resp(200, {"data": {"code": None}}))
    assert api.my_referral_code() is None


def test_request_airdrop_and_poll():
    api, s = _lifecycle_client(
        _Resp(200, {"data": {"job_id": "j1", "status": "running"}}),
        _Resp(200, {"data": {"status": "complete", "total": 3, "results": [
            {"symbol": "wALEO", "amm_token_program": "waleo.aleo",
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
        _AsyncResp(200, {"data": {"has_access": True}}),
        _AsyncResp(200, {"data": {"has_access": True, "code": None,
                                  "referred_by": None, "my_code": "MC"}}),
        _AsyncResp(200, {"data": {"code": "MC"}}),
        _AsyncResp(200, {"data": {"code": "C", "status": "redeemed"}}),
        _AsyncResp(200, {"data": {"job_id": "j1", "status": "running"}}),
        _AsyncResp(200, {"data": {"status": "complete", "total": 1, "results": [
            {"symbol": "wETH", "amm_token_program": "weth.aleo",
             "amount": "5", "status": "accepted"}]}}),
        _AsyncResp(429, {"error": "already claimed"}),
    ])
    api = AsyncApiClient(base_url="https://x", client=c, token="t")
    assert (await api.access_status()).has_access is True
    assert (await api.referral_status()).my_code == "MC"
    assert (await api.my_referral_code()) == "MC"
    assert (await api.redeem_code("C")).status == "redeemed"
    assert api._token == "t"          # redeem no longer rotates the credential
    assert [u for _, u, *_ in c.calls[1:3]] == [
        "https://x/referral/status", "https://x/referral/my-code"]
    assert (await api.request_airdrop("aleo1a")).job_id == "j1"
    job = await api.get_airdrop_job("j1")
    assert job.results[0].amm_token_program == "weth.aleo"
    with pytest.raises(AirdropRateLimitedError):
        await api.request_airdrop("aleo1a")


def test_access_code_tier_is_not_exposed():
    # Access codes are a deprecated compatibility tier (the API marks
    # POST /access/redeem "for old clients") — the SDK neither mints nor
    # redeems them.  Referral codes are the only codes that exist.
    for cls in (ApiClient, AsyncApiClient):
        assert not hasattr(cls, "generate_access_codes"), cls.__name__
        assert not hasattr(cls, "redeem_access_code"), cls.__name__


def test_cookie_session_outranks_bearer():
    # With both credentials loaded, requests ride the cookie session (CSRF
    # header, no Authorization) — the server honors Authorization first and
    # ss_ tokens don't cover the /access tier.
    s = _Session([_Resp(200, {"data": []})])
    api = ApiClient(base_url="https://x", session=s, token="ss_durable")
    api._csrf = "csrf-1"
    api._get("/access/status")
    headers = s.calls[0][3]
    assert headers["x-csrf-token"] == "csrf-1"
    assert "authorization" not in headers


def test_expired_cookie_session_falls_back_to_bearer():
    s = _Session([
        _Resp(401, {"error": "session expired"}),
        _Resp(200, {"data": []}),
    ])
    api = ApiClient(base_url="https://x", session=s, token="ss_durable")
    api._csrf = "csrf-1"
    api._get("/route")
    assert api._csrf is None                         # session dropped
    assert s.calls[0][3]["x-csrf-token"] == "csrf-1"
    assert s.calls[1][3]["authorization"] == "Bearer ss_durable"


# ── Per-network API host ─────────────────────────────────────────────────────

def test_api_url_for_each_network():
    from aleo_shield_swap.api import SHIELD_SWAP_API_URLS, api_url_for
    assert api_url_for("mainnet") == SHIELD_SWAP_API_URLS["mainnet"]
    assert api_url_for("testnet") == SHIELD_SWAP_API_URLS["testnet"]
    # the two must never collide — a testnet pool key means nothing on mainnet
    assert api_url_for("mainnet") != api_url_for("testnet")


def test_api_url_for_unknown_network_raises():
    from aleo_shield_swap.api import api_url_for
    with pytest.raises(ValueError, match="No DEX API host known"):
        api_url_for("devnet")


def test_api_url_env_override_wins(monkeypatch):
    from aleo_shield_swap.api import api_url_for
    monkeypatch.setenv("SHIELD_SWAP_API_URL", "http://localhost:8080/")
    assert api_url_for("mainnet") == "http://localhost:8080"   # slash stripped
    assert api_url_for("devnet") == "http://localhost:8080"    # override skips lookup


def test_default_api_url_is_not_mainnet():
    # an accidental default must not reach mainnet
    from aleo_shield_swap.api import DEFAULT_API_URL, SHIELD_SWAP_API_URLS
    assert DEFAULT_API_URL == SHIELD_SWAP_API_URLS["testnet"]


def test_client_picks_the_api_for_its_network(monkeypatch):
    from aleo_shield_swap.api import SHIELD_SWAP_API_URLS
    from aleo_shield_swap.client import ShieldSwap
    monkeypatch.delenv("SHIELD_SWAP_API_URL", raising=False)

    class _Net:
        def __init__(self, name): self.network_name = name

    for net in ("mainnet", "testnet"):
        assert ShieldSwap(_Net(net)).api.base_url == SHIELD_SWAP_API_URLS[net]
    # an explicit api_url still wins
    assert ShieldSwap(_Net("mainnet"), api_url="http://x").api.base_url == "http://x"
