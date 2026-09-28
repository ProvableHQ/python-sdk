"""Claim preparation polls missing output without retrying transactions."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aleo_shield_swap import ShieldSwap, AsyncShieldSwap
from aleo_shield_swap.errors import SwapOutputNotFinalizedError
from .test_claim import _handle


@pytest.mark.parametrize('async_mode', [False, True])
@pytest.mark.parametrize('case', ['delayed', 'timeout', 'rpc_error', 'immediate', 'invalid'])
def test_claim_readiness(stub_aleo, monkeypatch, async_mode, case):
    client = (AsyncShieldSwap if async_mode else ShieldSwap)(stub_aleo)
    module = 'aleo_shield_swap.' + ('async_client' if async_mode else 'client')
    mock = AsyncMock if async_mode else Mock
    missing = SwapOutputNotFinalizedError('77field')
    output = SimpleNamespace(token_out='2field')
    effects = [missing, output] if case == 'delayed' else [missing]
    if case == 'rpc_error':
        effects = [RuntimeError('RPC unavailable')]
    client.get_swap_output = mock(side_effect=effects)
    client._is_wrapped = mock(side_effect=RuntimeError('output ready'))
    sleep = mock()
    if async_mode:
        monkeypatch.setattr(module + '.asyncio.sleep', sleep)
    # A fake clock keeps the timeout deterministic without sleeping.
    monkeypatch.setattr(module + '.time', Mock(monotonic=Mock(side_effect=[0, 10]), sleep=sleep))
    if case == 'delayed':
        monkeypatch.setattr(module + '.time', Mock(monotonic=Mock(side_effect=[0, 0]), sleep=sleep))
    seconds = 0 if case == 'immediate' else -1 if case == 'invalid' else 10
    def prepare():
        result = client.claim_swap_output(_handle(), timeout=seconds)
        return asyncio.run(result) if async_mode else result
    expected = ValueError if case == 'invalid' else RuntimeError if case in ('delayed', 'rpc_error') else SwapOutputNotFinalizedError
    with pytest.raises(expected) as caught:
        prepare()
    if case == 'delayed':
        assert str(caught.value) == 'output ready'
        assert client.get_swap_output.call_count == 2
        assert sleep.call_count == 1
    else:
        assert client.get_swap_output.call_count == (0 if case == 'invalid' else 1)
        sleep.assert_not_called()
        client._is_wrapped.assert_not_called()
    assert stub_aleo.submitted == []
