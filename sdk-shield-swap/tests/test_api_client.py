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

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self.responses.pop(0)


def test_get_pools_parses_models_and_token_info():
    s = _Session([_Resp(200, POOLS)])
    pools = ApiClient(base_url="https://x", session=s).get_pools()
    assert s.calls[0][0] == "https://x/pools"
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
    url, params = s.calls[0]
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
    assert s.calls[0][1] == {"user": "aleo1me"}


def test_non_2xx_raises_dex_api_error():
    s = _Session([_Resp(503, {"error": "down"})])
    with pytest.raises(DexApiError) as ei:
        ApiClient(base_url="https://x", session=s).get_pools()
    assert ei.value.status == 503
