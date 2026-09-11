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

    def put(self, url, json=None, timeout=None, headers=None):
        self.calls.append(("PUT", url, json, headers))
        return self.responses.pop(0)

    def delete(self, url, timeout=None, headers=None):
        self.calls.append(("DELETE", url, None, headers))
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


RETIRED_ROUTES = ("access_status", "my_referral_codes", "get_tick_spacings",
                  "get_swaps", "get_swap", "get_position", "get_public_balances")


@pytest.mark.parametrize("name", RETIRED_ROUTES)
def test_retired_dex_routes_have_no_wrapper(name):
    # The API retired these routes in 2026-09 (they 404 on both networks).
    # Balances, swap detail, and position detail are chain reads on
    # ShieldSwap; the access flag rides on referral_status().
    assert not hasattr(ApiClient, name)
    assert not hasattr(AsyncApiClient, name)


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
    client.get_fee_tiers()
    headers = s.calls[2][3]
    assert headers["authorization"] == "Bearer jwt-abc"


def test_401_maps_to_not_authenticated():
    from aleo_shield_swap.errors import NotAuthenticatedError
    s = _Session([_Resp(401, {"error": "missing token"})])
    with pytest.raises(NotAuthenticatedError):
        ApiClient(base_url="https://x", session=s)._get("/fee-tiers")


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


def test_referral_status_carries_the_access_flag():
    # /access/status is gone; referral_status() is the gated liveness probe.
    api, s = _lifecycle_client(_Resp(200, {"data": {"has_access": True, "code": None,
                                                    "referred_by": None, "my_code": "MC"}}))
    assert api.referral_status().has_access is True
    assert s.calls[0][:2] == ("GET", "https://x/referral/status")


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

    async def put(self, url, json=None, headers=None):
        self.calls.append(("PUT", url, json, headers))
        return self.responses.pop(0)

    async def delete(self, url, headers=None):
        self.calls.append(("DELETE", url, None, headers))
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_async_lifecycle_endpoints():
    from aleo_shield_swap.api import AsyncApiClient
    from aleo_shield_swap.errors import AirdropRateLimitedError
    c = _AsyncClient([
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
    status = await api.referral_status()
    assert status.has_access is True and status.my_code == "MC"
    assert (await api.my_referral_code()) == "MC"
    assert (await api.redeem_code("C")).status == "redeemed"
    assert api._token == "t"          # redeem no longer rotates the credential
    assert [u for _, u, *_ in c.calls[0:2]] == [
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


# ── Endpoints added with the 2026-09 API (pool analytics, positions, routing
#    topology, rebalance state, token management, referral reporting) ─────────

def test_get_pool_and_pool_stats():
    api, s = _lifecycle_client(
        _Resp(200, {"data": {"key": "5field", "token0": "1field", "token1": "2field",
                             "fee": "3000", "stats": {"volume_24h": "10"}}}),
        _Resp(200, {"data": {"price": "1.5", "display_price": "0.66", "display_flipped": True,
                             "fee_24h": "12", "fee_7d": "80", "volume_7d": "900",
                             "reserve0": "100", "reserve1": "200",
                             "high_24h": "2", "low_24h": "1"}}),
    )
    pool = api.get_pool("5field")
    assert pool.key == "5field"
    stats = api.get_pool_stats("5field")
    assert stats.fee_24h == "12" and stats.reserve1 == "200" and stats.display_flipped is True
    assert [c[1] for c in s.calls] == ["https://x/pools/5field", "https://x/pools/5field/stats"]


def test_get_pool_trades_passes_paging_and_filter():
    api, s = _lifecycle_client(_Resp(200, {"data": [
        {"id": "t1", "pool": "5field", "amount0": "1", "amount1": "2", "executedAt": "now",
         "tradeType": "swap", "transactionHash": "at1", "legIndex": 0,
         "fee0": "3", "protocolFee0": "1", "liquidityAfter": "555", "tickAfter": 12}]}))
    trades = api.get_pool_trades("5field", limit=10, offset=20, trade_type="swap")
    assert trades[0].fee0 == "3" and trades[0].tickAfter == 12 and trades[0].legIndex == 0
    assert s.calls[0][1] == "https://x/pools/5field/trades"
    assert s.calls[0][2] == {"limit": 10, "offset": 20, "trade_type": "swap"}


def test_get_initialized_ticks():
    api, s = _lifecycle_client(_Resp(200, {"data": [-1200, 300, 4080]}))
    assert api.get_initialized_ticks("5field") == [-1200, 300, 4080]
    assert s.calls[0][1] == "https://x/pools/5field/initialized-ticks"


def test_get_positions_is_session_scoped():
    # /positions lists the authenticated account's own positions — no user
    # parameter exists; paging is the only knob.
    api, s = _lifecycle_client(
        _Resp(200, {"data": [{"token_id": "42field", "pool": "5field", "liquidity": "500"}]}),
        _Resp(200, {"data": [{"token_id": "43field", "pool": "5field", "liquidity": "1"}]}),
    )
    positions = api.get_positions()
    assert positions[0].token_id == "42field"
    assert s.calls[0][1] == "https://x/positions" and s.calls[0][2] in (None, {})
    assert api.get_positions(limit=1, offset=1)[0].token_id == "43field"
    assert s.calls[1][2] == {"limit": 1, "offset": 1}


def test_get_fee_tiers_carries_tick_spacing():
    # /tick-spacings is gone; each fee tier names its own spacing.
    api, s = _lifecycle_client(
        _Resp(200, {"data": [{"id": "f1", "fee_tier": 3000, "tick_spacing": 60,
                              "created_at": "t", "transaction": "at1"}]}),
    )
    tier = api.get_fee_tiers()[0]
    assert tier.fee_tier == 3000 and tier.tick_spacing == 60
    assert [c[1] for c in s.calls] == ["https://x/fee-tiers"]


def test_get_route_accepts_a_pool_key_pin():
    api, s = _lifecycle_client(_Resp(200, {"data": {
        "token_in": "1field", "token_out": "2field", "estimated_amount_out": "0.9",
        "hops": [], "protocol_revision": 3}}))
    api.get_route(token_in="1field", token_out="2field", amount_in="1", pool_key="5field")
    assert s.calls[0][2] == {"token_in": "1field", "token_out": "2field",
                             "amount_in": "1", "pool_key": "5field"}


def test_get_route_topology():
    api, s = _lifecycle_client(_Resp(200, {"data": {
        "edges": [{"token0": "1field", "token1": "2field"}], "max_hops": 3,
        "protocol_revision": 3, "protocol_config_observed_block": 100}}))
    topo = api.get_route_topology()
    assert topo.max_hops == 3 and topo.edges[0].token1 == "2field"
    assert s.calls[0][1] == "https://x/route/topology"


def test_get_rebalance_state_sends_every_query_param():
    api, s = _lifecycle_client(_Resp(200, {"data": {
        "tick": 5, "tick_spacing": 60, "sqrt_price_x_128": "340282366920938463463374607431768211456",
        "fee_growth_global0_x_128": "0", "fee_growth_global1_x_128": "0",
        "lower": {"tick": -60, "fee_growth_outside0_x_128": "0", "fee_growth_outside1_x_128": "0"},
        "upper": {"tick": 60, "fee_growth_outside0_x_128": "0", "fee_growth_outside1_x_128": "0"},
        "tick_lower_hint": -400001, "tick_upper_hint": -60, "observed_block": 4242}}))
    st = api.get_rebalance_state("5field", tick_lower=-60, tick_upper=60,
                                 old_liquidity=1000, mint_tick_lower=-120, mint_tick_upper=120)
    assert st.tick_lower_hint == -400001 and st.lower.tick == -60 and st.observed_block == 4242
    assert s.calls[0][1] == "https://x/pools/5field/rebalance-state"
    assert s.calls[0][2] == {"tick_lower": -60, "tick_upper": 60, "old_liquidity": "1000",
                             "mint_tick_lower": -120, "mint_tick_upper": 120}


def test_api_token_list_and_revoke():
    api, s = _lifecycle_client(
        _Resp(200, {"data": {"tokens": [{"id": "u1", "name": "bot", "token_prefix": "ss_ab",
                                         "created_at": "2026-09-01", "expires_at": None}]}}),
        _Resp(200, {"data": {"id": "u1", "revoked": True}}),
    )
    tokens = api.list_api_tokens()
    assert tokens[0].name == "bot" and tokens[0].token_prefix == "ss_ab"
    assert api.revoke_api_token("u1").revoked is True
    assert s.calls[0][:2] == ("GET", "https://x/api-tokens")
    assert s.calls[1][:2] == ("DELETE", "https://x/api-tokens/u1")


def test_referral_reporting_endpoints():
    api, s = _lifecycle_client(
        _Resp(200, {"data": {"recorded": True}}),
        _Resp(200, {"data": {"recorded": True}}),
        _Resp(200, {"data": {"recorded": 2, "duplicate": 1, "conflict": 0}}),
    )
    assert api.report_referral_activity(action="create_pool", tx_id="at1x",
                                        metadata={"pool": "5field"}).recorded is True
    assert s.calls[0][1] == "https://x/referral/activity"
    assert s.calls[0][2] == {"action": "create_pool", "tx_id": "at1x", "metadata": {"pool": "5field"}}
    assert api.report_referral_swap_claim(code="ABC", blinded_address="aleo1b").recorded is True
    assert s.calls[1][2] == {"code": "ABC", "blinded_address": "aleo1b"}
    batch = api.report_referral_address_batch(code="ABC", blinded_addresses=["aleo1b", "aleo1c", "aleo1d"])
    assert (batch.recorded, batch.duplicate, batch.conflict) == (2, 1, 0)
    assert s.calls[2][2] == {"code": "ABC", "blinded_addresses": ["aleo1b", "aleo1c", "aleo1d"]}


def test_dex_api_error_carries_the_machine_code():
    s = _Session([_Resp(409, {"error": "stale quote", "code": "protocol_revision_mismatch",
                              "ref": "r-1"})])
    with pytest.raises(DexApiError) as exc:
        ApiClient(base_url="https://x", session=s, token="t")._get("/route")
    assert exc.value.status == 409
    assert exc.value.code == "protocol_revision_mismatch"
    # A body that is not the API's JSON envelope still parses.
    s = _Session([_Resp(502, {"unexpected": True})])
    with pytest.raises(DexApiError) as exc:
        ApiClient(base_url="https://x", session=s, token="t")._get("/route")
    assert exc.value.code is None


def test_get_protocol_state_is_unwrapped_and_takes_minimum_revision():
    # /protocol/state returns the document directly (no {"data": ...} envelope).
    api, s = _lifecycle_client(_Resp(200, {
        "revision": 7, "observed_block": 4242, "changed_at": None,
        "freshness": {"ready_for_quote": True, "ready_for_entry": False,
                      "lag_blocks": 2, "confirmed_head": 4244, "indexed_block": 4242,
                      "updated_at": None},
        "capabilities": {}, "controls": {}, "deployment": {}, "fee_configuration": {},
        "live_compatibility": {}}))
    st = api.get_protocol_state(minimum_revision=5)
    assert st.revision == 7 and st.freshness.ready_for_quote is True
    assert st.freshness.ready_for_entry is False
    assert s.calls[0][1] == "https://x/protocol/state"
    assert s.calls[0][2] == {"minimum_revision": 5}


def test_get_unclaimed_lists_pending_swaps_and_owed_positions():
    api, s = _lifecycle_client(_Resp(200, {"data": {
        "pending_swaps": [{"swap_id": "77field", "swap_tx_hash": "at1s", "is_multi_hop": False,
                           "is_private": True, "output": {"amount_out": "9"},
                           "token_in_info": None, "token_out_info": None}],
        "positions_with_owed": [{"token_id": "42field", "pool": "5field", "tick_lower": -60,
                                 "tick_upper": 60, "tokens_owed0": "1", "tokens_owed1": "2",
                                 "is_frozen": False, "is_private": True, "frozen_at": None,
                                 "last_transaction_hash": None,
                                 "token0_info": None, "token1_info": None}]}}))
    un = api.get_unclaimed()
    assert un.pending_swaps[0].swap_id == "77field"
    assert un.positions_with_owed[0].tokens_owed1 == "2"
    assert s.calls[0][1] == "https://x/unclaimed"


@pytest.mark.asyncio
async def test_async_client_mirrors_the_new_endpoints():
    from aleo_shield_swap.api import AsyncApiClient
    c = _AsyncClient([
        _AsyncResp(200, {"data": [-60, 60]}),
        _AsyncResp(200, {"data": {"edges": [], "max_hops": 3, "protocol_revision": 1}}),
        _AsyncResp(200, {"data": {"recorded": True}}),
        _AsyncResp(200, {"data": {"id": "u1", "revoked": True}}),
    ])
    api = AsyncApiClient(base_url="https://x", client=c, token="t")
    assert await api.get_initialized_ticks("5field") == [-60, 60]
    assert (await api.get_route_topology()).max_hops == 3
    assert (await api.report_referral_swap_claim(code="ABC", blinded_address="aleo1b")).recorded
    assert (await api.revoke_api_token("u1")).revoked is True
    assert c.calls[3][:2] == ("DELETE", "https://x/api-tokens/u1")
    # Every sync endpoint wrapper has an async twin.
    sync_api = {n for n in dir(ApiClient) if not n.startswith("_")}
    async_api = {n for n in dir(AsyncApiClient) if not n.startswith("_")}
    assert sync_api - async_api == set()


# ── 2026-09: session management, compliance, pool depth, referral issuance ──

SESSION = {"address": "aleo1me", "csrf_token": "csrf-2", "expires_at": 1_800_000_000,
           "session_id": "sid-1", "session_version": 3}


def _cookie_client(*resps):
    s = _Session(list(resps))
    api = ApiClient(base_url="https://x", session=s)
    api._csrf = "csrf-1"                           # a live cookie session
    return api, s


def test_session_reads_and_refresh_adopts_the_rotated_csrf():
    api, s = _cookie_client(
        _Resp(200, {"data": SESSION}),
        _Resp(200, {"data": SESSION}),
        _Resp(200, {"data": [{"session_id": "sid-1", "started_at": 1, "last_refreshed_at": 2,
                              "expires_at": 3, "user_agent": "py", "current": True},
                             {"session_id": "sid-0", "started_at": 0, "last_refreshed_at": 0,
                              "expires_at": 3, "user_agent": None, "current": False}]}),
        _Resp(200, {"data": {"token": "ws-jwt", "expires_at": 9}}),
    )
    assert api.get_session().session_id == "sid-1"
    assert s.calls[0][:2] == ("GET", "https://x/auth/session")
    assert s.calls[0][3]["x-csrf-token"] == "csrf-1"
    assert api.refresh_session().session_version == 3
    assert s.calls[1][:3] == ("POST", "https://x/auth/refresh", {})
    assert api._csrf == "csrf-2"                   # rotated token adopted
    sessions = api.list_sessions()
    assert [x.current for x in sessions] == [True, False]
    assert api.get_ws_ticket().token == "ws-jwt"
    assert s.calls[3][1] == "https://x/auth/ws-ticket"


def test_logout_binds_to_the_session_and_keeps_a_bearer():
    # The API refuses /auth/logout unless the request names the session it
    # ends, so logout() reads the session first and sends the binding headers.
    api, s = _cookie_client(
        _Resp(200, {"data": SESSION}),
        _Resp(200, {"data": {"ok": True, "ended": True, "session_id": "sid-1"}}),
    )
    api._token = "ss_reserve"
    out = api.logout()
    assert out.ended is True
    assert s.calls[0][:2] == ("GET", "https://x/auth/session")
    assert s.calls[1][:2] == ("POST", "https://x/auth/logout")
    assert s.calls[1][3]["x-shield-session-id"] == "sid-1"
    assert s.calls[1][3]["x-shield-wallet-address"] == "aleo1me"
    assert s.calls[1][3]["x-csrf-token"] == "csrf-1"   # binding adds to, never replaces
    assert api._csrf is None and api._token == "ss_reserve"
    assert api.is_authenticated                    # the bearer is still there
    # A caller holding the session already skips the extra read.
    api2, s2 = _cookie_client(_Resp(200, {"data": {"ok": True, "ended": True, "session_id": "sid-1"}}))
    from aleo_shield_swap import _api_models as models
    api2.logout(models.SessionPayload(**SESSION))
    assert [c[0] for c in s2.calls] == ["POST"]


def test_revoke_session_drops_local_state_only_for_the_current_one():
    api, s = _cookie_client(
        _Resp(200, {"data": {"ok": True, "revoked": True, "current": False}}),
        _Resp(200, {"data": {"ok": True, "revoked": True, "current": True}}),
    )
    assert api.revoke_session("sid-0").revoked is True
    assert s.calls[0][1] == "https://x/auth/sessions/sid-0/revoke"
    assert api._csrf == "csrf-1"                   # someone else's session
    api.revoke_session("sid-1")
    assert api._csrf is None


def test_logout_all_reports_the_bumped_session_version():
    api, s = _cookie_client(_Resp(200, {"data": {"ok": True, "ended": True,
                                                 "address": "aleo1me", "session_version": 4}}))
    assert api.logout_all().session_version == 4
    assert s.calls[0][:2] == ("POST", "https://x/auth/logout-all") and api._csrf is None


def test_compliance_reads_are_public():
    s = _Session([
        _Resp(200, {"data": {"global_paused": False, "pool_creation_is_open": True}}),
        _Resp(200, {"data": {"token_id": "1field", "allowed": True, "paused": False}}),
        _Resp(200, {"data": {"token0": "1field", "token1": "2field", "paused": True}}),
    ])
    api = ApiClient(base_url="https://x", session=s)
    assert api.get_compliance().pool_creation_is_open is True
    assert api.get_token_compliance("1field").allowed is True
    assert api.get_pair_compliance("1field", "2field").paused is True
    assert [c[1] for c in s.calls] == ["https://x/compliance", "https://x/compliance/tokens/1field",
                                       "https://x/compliance/pairs/1field/2field"]
    assert all("authorization" not in c[3] and "x-csrf-token" not in c[3] for c in s.calls)


STATS = {"price": "1.5", "liquidity": "10", "reserve0": "1", "reserve1": "2"}


def test_pool_stats_batch_and_liquidity_distribution():
    s = _Session([
        _Resp(200, {"data": {"5field": STATS, "6field": STATS}}),
        _Resp(200, {"data": [{"tick": -60, "liquidity_net": "100"},
                             {"tick": 60, "liquidity_net": "-100"}]}),
    ])
    api = ApiClient(base_url="https://x", session=s)
    stats = api.get_pool_stats_batch(["5field", "6field"])
    assert set(stats) == {"5field", "6field"} and stats["5field"].price == "1.5"
    assert s.calls[0][1:3] == ("https://x/pools/stats", {"keys": "5field,6field"})
    dist = api.get_liquidity_distribution("5field")
    assert [(d.tick, d.liquidity_net) for d in dist] == [(-60, "100"), (60, "-100")]
    assert s.calls[1][1] == "https://x/pools/5field/liquidity-distribution"
    assert api.get_pool_stats_batch([]) == {} and len(s.calls) == 2   # no request


def test_referral_issuance_surface():
    api, s = _lifecycle_client(
        _Resp(200, {"data": {"codes_per_user": 3, "codes_per_user_limit": 5, "max_users": None}}),
        _Resp(200, {"data": {"codes_per_user": 4, "codes_per_user_limit": 5, "max_users": 10}}),
        _Resp(200, {"data": {"total": 2, "redeemed": 1, "available": 1, "redemptions": 1,
                             "codes": [{"code": "AAA", "created_at": "t", "issued_to": None,
                                        "redeemed_at": "t2", "redeemed_by": "aleo1b",
                                        "redemption_count": 1}]}}),
        _Resp(200, {"data": {"codes": ["NEW1", "NEW2"]}}),
    )
    assert api.referral_settings().codes_per_user == 3
    updated = api.update_referral_settings(codes_per_user=4, max_users=10)
    assert updated.max_users == 10
    assert s.calls[1][:3] == ("PUT", "https://x/referral/settings",
                              {"codes_per_user": 4, "max_users": 10})
    page = api.list_referral_codes(limit=1)
    assert page.total == 2 and page.codes[0].redeemed_by == "aleo1b"
    assert s.calls[2][1:3] == ("https://x/referral/codes", {"limit": 1})
    assert api.generate_referral_codes(2) == ["NEW1", "NEW2"]
    assert s.calls[3][:3] == ("POST", "https://x/referral/generate", {"count": 2})


NEW_2026_09_METHODS = (
    "get_session", "refresh_session", "list_sessions", "revoke_session", "logout",
    "logout_all", "get_ws_ticket", "get_compliance", "get_token_compliance",
    "get_pair_compliance", "get_pool_stats_batch", "get_liquidity_distribution",
    "referral_settings", "update_referral_settings", "list_referral_codes",
    "generate_referral_codes",
)


@pytest.mark.parametrize("name", NEW_2026_09_METHODS)
def test_new_wrappers_exist_on_both_clients(name):
    assert callable(getattr(ApiClient, name)) and callable(getattr(AsyncApiClient, name))


@pytest.mark.asyncio
async def test_async_session_compliance_and_referral_mirrors():
    c = _AsyncClient([
        _AsyncResp(200, {"data": SESSION}),
        _AsyncResp(200, {"data": SESSION}),
        _AsyncResp(200, {"data": {"ok": True, "ended": True, "session_id": "sid-1"}}),
        _AsyncResp(200, {"data": {"global_paused": True, "pool_creation_is_open": False}}),
        _AsyncResp(200, {"data": {"5field": STATS}}),
        _AsyncResp(200, {"data": {"codes_per_user": 1, "codes_per_user_limit": 1, "max_users": None}}),
        _AsyncResp(200, {"data": {"codes": ["X"]}}),
    ])
    api = AsyncApiClient(base_url="https://x", client=c)
    api._csrf = "csrf-1"
    assert (await api.refresh_session()).address == "aleo1me" and api._csrf == "csrf-2"
    assert (await api.logout()).ok is True and api._csrf is None
    assert c.calls[2][1] == "https://x/auth/logout"
    assert c.calls[2][3]["x-shield-session-id"] == "sid-1"
    assert (await api.get_compliance()).global_paused is True
    assert (await api.get_pool_stats_batch(["5field"]))["5field"].liquidity == "10"
    assert (await api.update_referral_settings(codes_per_user=1)).codes_per_user == 1
    assert c.calls[5][:3] == ("PUT", "https://x/referral/settings", {"codes_per_user": 1})
    assert await api.generate_referral_codes() == ["X"]
    assert c.calls[6][2] == {"count": 1}


def test_build_coerces_dict_values_and_enums():
    # LiveCompatibility.observed_programs is dict[str, LiveProgramObservation]
    # and .status is an Enum — both must come back typed, recursively.
    from aleo_shield_swap import _api_models as models
    payload = {"revision": 7, "live_compatibility": {
        "status": "compatible", "checked_at": "t", "artifacts_checked": 1, "failures": [],
        "observed_programs": {"shield_swap.aleo": {"source_sha256": "ab", "edition": 3,
                                                   "version": "1.0"}}}}
    s = _Session([_Resp(200, payload)])
    state = ApiClient(base_url="https://x", session=s).get_protocol_state()
    lc = state.live_compatibility
    assert lc.status is models.LiveCompatibilityStatus.compatible
    obs = lc.observed_programs["shield_swap.aleo"]
    assert isinstance(obs, models.LiveProgramObservation) and obs.edition == 3
    assert state.capabilities is None                # a dropped field still reads as None
    # An enum value the spec does not know yet stays a plain string, not an error.
    payload["live_compatibility"]["status"] = "degraded"
    state = ApiClient(base_url="https://x", session=_Session([_Resp(200, payload)])).get_protocol_state()
    assert state.live_compatibility.status == "degraded"


def test_logout_on_an_expired_session_forgets_it_and_reraises():
    from aleo_shield_swap.errors import NotAuthenticatedError
    api, s = _cookie_client(_Resp(401, {"error": "session expired"}))
    with pytest.raises(NotAuthenticatedError):
        api.logout()
    assert api._csrf is None and not api.is_authenticated
