"""A symbol quote carries a direct or multi-hop route into swap preparation."""
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock
import pytest
from aleo_shield_swap import ShieldSwap
from .conftest import SLOT_TEXT, StubAleo


def token(symbol, address, decimals):
    return NS(symbol=symbol, address=address, id=symbol+'-uuid', decimals=decimals)


def setup_quote(hops=1, wrapped=False):
    pools = {f'{i+5}field': f'{{ token0: {i+1}field, token1: {i+2}field, fee: 3000u16, enabled: true }}' for i in range(hops)}
    stub = StubAleo(mappings={'pools': pools, 'slots': {k:SLOT_TEXT for k in pools}})
    dex = ShieldSwap(stub)
    tokens = [token('USDCx','1field',6), token('ETH',f'{hops+1}field',18)]
    dex.api.get_tokens = Mock(return_value=tokens)
    route = NS(token_in='1field', token_out=f'{hops+1}field', estimated_amount_out='0.0008',
               protocol_revision=1, protocol_config_observed_block=100,
               hops=[NS(pool_key=k, token_in=f'{i+1}field', token_out=f'{i+2}field', zero_for_one=True) for i,k in enumerate(pools)])
    dex.api.get_route = Mock(return_value=route)
    dex._is_wrapped = Mock(return_value=wrapped)
    dex._token_program = Mock(return_value='tok.aleo')
    dex._amm_token_program = Mock(return_value='tok.aleo')
    dex._ensure = Mock()
    return dex, stub


@pytest.mark.parametrize('hops', [1,2,3])
@pytest.mark.parametrize('wrapped', [False,True])
def test_quote_executes_route(hops, wrapped):
    dex, stub = setup_quote(hops,wrapped)
    q = dex.quote(token_in='USDCx', token_out='ETH', amount_in='1.5', slippage_bps=50)
    assert q.amount_in == '1.5'
    assert q.estimated_amount_out == '0.0008'
    assert q.minimum_amount_out == '0.000796'
    assert len(q.hops) == hops
    dex.api.get_route.assert_called_once_with(token_in='1field', token_out=f'{hops+1}field', amount_in='1.5')
    assert stub.last_call is None and stub.submitted == []
    handle = dex.swap(q, nonce=123).transact()
    fn, inputs = stub.last_call
    assert fn == (('swap_mh_from_wrapped' if wrapped else 'swap_multi_hop') if hops>1 else ('swap_from_wrapped' if wrapped else 'swap'))
    assert handle.token_out_id == f'{hops+1}field'
    assert handle.amount_in == 1500000
    if hops>1:
        offset = 4 if wrapped else 3
        assert inputs[offset:offset+4] == ['1field', f'{hops+1}field', '1500000u128', '796000000000000u128']
        assert inputs[offset+7:] == [f'{hops}u8','123u64','11000u32']
        assert handle.pool_keys == tuple(f'{i+5}field' for i in range(hops))


def test_quote_rejects_conflicting_swap_parameters():
    dex, stub = setup_quote()
    q=dex.quote(token_in='USDCx',token_out='ETH',amount_in='1.5')
    with pytest.raises(ValueError):
        dex.swap(q, amount_in='2')
    assert stub.submitted == []


def test_quote_rejects_wrong_network():
    dex, stub=setup_quote()
    q=dex.quote(token_in='USDCx',token_out='ETH',amount_in='1.5')
    with pytest.raises(ValueError):
        dex.swap(replace(q, network='other'))
    assert stub.last_call is None


@pytest.mark.parametrize('change', [
    lambda q: replace(q, hops=()),
    lambda q: replace(q, hops=q.hops * 2),
    lambda q: replace(q, hops=(replace(q.hops[0], token_out='99field'), *q.hops[1:])),
    lambda q: replace(q, hops=(q.hops[0], replace(q.hops[1], pool_key=q.hops[0].pool_key))),
    lambda q: replace(q, minimum_amount_out='0'),
    lambda q: replace(q, minimum_amount_out='0.0007'),
    lambda q: replace(q, program='other.aleo'),
])
def test_invalid_quote_never_prepares_a_transaction(change):
    dex, stub = setup_quote(2)
    quote = dex.quote(token_in='USDCx', token_out='ETH', amount_in='1.5')
    with pytest.raises(ValueError):
        dex.swap(change(quote))
    assert stub.last_call is None
    dex._token_program.assert_not_called()


@pytest.mark.parametrize('amount', ['0', '-1', '0.0000001', 'NaN', 'Infinity', 1.5, 1500000])
def test_quote_rejects_invalid_amount_before_route_lookup(amount):
    dex, _ = setup_quote()
    with pytest.raises((ValueError, TypeError)):
        dex.quote(token_in='USDCx', token_out='ETH', amount_in=amount)
    dex.api.get_route.assert_not_called()


def test_live_pool_direction_is_checked():
    dex, stub = setup_quote(2)
    quote = dex.quote(token_in='USDCx', token_out='ETH', amount_in='1.5')
    quote = replace(quote, hops=(quote.hops[0], replace(quote.hops[1], zero_for_one=False)))
    with pytest.raises(ValueError, match='direction'):
        dex.swap(quote)
    assert stub.last_call is None
    dex._token_program.assert_not_called()


def test_multihop_handle_survives_journal_and_json(tmp_path):
    from aleo_shield_swap import Journal, SwapHandle
    dex, _ = setup_quote(3)
    dex.journal = Journal(tmp_path / 'swaps.jsonl')
    quote = dex.quote(token_in='USDCx', token_out='ETH', amount_in='1.5')
    handle = dex.swap(quote).transact()
    assert dex.journal.pending_claims() == [handle]
    assert SwapHandle.from_json(handle.to_json()) == handle


@pytest.mark.asyncio
@pytest.mark.parametrize('hops', [1, 2, 3])
@pytest.mark.parametrize('wrapped', [False, True])
async def test_async_quote_encodes_same_transaction(hops, wrapped):
    from unittest.mock import AsyncMock
    from aleo_shield_swap import AsyncShieldSwap
    from .test_async_client import AsyncStubAleo
    sync, stub = setup_quote(hops, wrapped)
    quote = sync.quote(token_in='USDCx', token_out='ETH', amount_in='1.5')
    sync.swap(quote, nonce=123).transact()
    async_stub = AsyncStubAleo(mappings=stub._mappings)
    dex = AsyncShieldSwap(async_stub)
    dex.api.get_tokens = AsyncMock(return_value=sync.api.get_tokens.return_value)
    dex.api.get_route = AsyncMock(return_value=sync.api.get_route.return_value)
    for name in ('_is_wrapped', '_token_program', '_amm_token_program', '_ensure'):
        setattr(dex, name, AsyncMock(return_value=getattr(sync, name).return_value))
    async_quote = await dex.quote(token_in='USDCx', token_out='ETH', amount_in='1.5')
    assert async_quote == quote
    handle = await (await dex.swap(async_quote, nonce=123)).transact()
    assert async_stub.last_call == stub.last_call
    assert handle.token_out_id == quote.token_out_id
