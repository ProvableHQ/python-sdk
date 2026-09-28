"""Balance preflight matches the swap's single-record requirement."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock
import pytest
from aleo_shield_swap import ShieldSwap, AsyncShieldSwap
from .conftest import RECORD_TEXT


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("case", ["enough", "empty", "fragmented", "error"])
def test_balance_check(stub_aleo, async_mode, case):
    client = (AsyncShieldSwap if async_mode else ShieldSwap)(stub_aleo)
    make_mock = AsyncMock if async_mode else Mock
    client.api.get_tokens = make_mock(return_value=[SimpleNamespace(id="uuid", address="1field", decimals=6)])
    client._token_program = make_mock(return_value="underlying.aleo")
    record = RECORD_TEXT.replace("2000000000", "1500000" if case == "enough" else "1000000")
    records = [] if case == "empty" else [{"record_plaintext": record}] * (2 if case == "fragmented" else 1)
    stub_aleo.record_provider.find = make_mock(return_value=records)
    if case == "error":
        stub_aleo.record_provider.find.side_effect = RuntimeError("scanner unavailable")
    def check():
        result = client.has_swap_balance("1field", "1.5")
        return asyncio.run(result) if async_mode else result
    if case == "error":
        with pytest.raises(RuntimeError, match="scanner unavailable"):
            check()
    else:
        assert check() is (case == "enough")
    stub_aleo.record_provider.find.assert_called_once_with(stub_aleo.default_account, program="underlying.aleo", unspent=True)
    assert stub_aleo.submitted == []
