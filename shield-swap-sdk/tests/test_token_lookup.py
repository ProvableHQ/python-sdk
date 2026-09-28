"""Symbol lookup returns token metadata without choosing ambiguous matches."""
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from aleo_shield_swap.api import ApiClient, AsyncApiClient
from aleo_shield_swap._api_models import TokenDoc
from aleo_shield_swap.errors import DexApiError


@pytest.fixture(params=[False, True], ids=["sync", "async"])
def lookup(request):
    asynchronous = request.param
    client = object.__new__(AsyncApiClient if asynchronous else ApiClient)
    client.get_tokens = AsyncMock() if asynchronous else Mock()

    def get(symbol):
        result = client.get_token(symbol)
        return asyncio.run(result) if asynchronous else result

    return client, get


def test_lookup_returns_matching_metadata(lookup):
    client, get = lookup
    usdc = TokenDoc("usdc.aleo", 6, "1field", "USD Coin", "USDCx")
    eth = TokenDoc("eth.aleo", 8, "2field", "Ether", "ETH")
    client.get_tokens.return_value = [usdc, eth]
    assert get("ETH") is eth
    assert get("USDCx") is usdc


def test_unknown_symbol_raises_clear_error(lookup):
    client, get = lookup
    client.get_tokens.return_value = []
    with pytest.raises(ValueError, match="Unknown token symbol: ETH"):
        get("ETH")


def test_ambiguous_symbol_does_not_choose_first(lookup):
    client, get = lookup
    client.get_tokens.return_value = [
        TokenDoc("a.aleo", 8, "1field", "Ether", "ETH"),
        TokenDoc("b.aleo", 8, "2field", "Other Ether", "ETH"),
    ]
    with pytest.raises(ValueError, match="Ambiguous token symbol: ETH"):
        get("ETH")


def test_symbol_matching_is_exact(lookup):
    client, get = lookup
    client.get_tokens.return_value = [TokenDoc("a.aleo", 8, "1field", "Ether", "ETH")]
    with pytest.raises(ValueError, match="Unknown token symbol: eth"):
        get("eth")


def test_api_failure_propagates(lookup):
    client, get = lookup
    error = DexApiError(503, "unavailable")
    client.get_tokens.side_effect = error
    with pytest.raises(DexApiError) as caught:
        get("ETH")
    assert caught.value is error
