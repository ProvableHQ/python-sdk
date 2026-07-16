import pytest

from aleo_shield_swap._core import (
    generate_field_nonce,
    generate_swap_nonce,
    parse_token_record_info,
    resolve_imports,
    resolve_swap_params,
    select_token_record,
)
from aleo_shield_swap.errors import InsufficientRecordsError
from aleo_shield_swap.tick_math import MAX_SQRT_PRICE, MIN_SQRT_PRICE, Q64


class _Pool:  # duck-typed: only the fields resolve_swap_params reads
    token0 = "1field"
    token1 = "2field"
    scale0 = 10**9
    scale1 = 1


class _Slot:
    sqrt_price = Q64   # price 1.0 in normalized units


def test_direction_and_spot_estimate():
    r = resolve_swap_params(pool=_Pool(), slot=_Slot(), token_in_id="1field",
                            amount_in=10**9, slippage_bps=50)
    assert r.zero_for_one is True and r.token_out_id == "2field"
    # spot: norm_in=1, price 1.0, scale_out=1 → expected 1; 1*9950//10000 == 0
    assert r.amount_out_min == 0
    assert r.sqrt_price_limit == MIN_SQRT_PRICE


def test_explicit_quote_and_reverse_direction():
    r = resolve_swap_params(pool=_Pool(), slot=_Slot(), token_in_id="2field",
                            amount_in=5, slippage_bps=100, expected_out=10**9)
    assert r.zero_for_one is False and r.token_out_id == "1field"
    assert r.amount_out_min == 10**9 * 9900 // 10000
    assert r.sqrt_price_limit == MAX_SQRT_PRICE


def test_rejections():
    with pytest.raises(ValueError, match="dust"):
        resolve_swap_params(pool=_Pool(), slot=_Slot(), token_in_id="1field",
                            amount_in=10**9 + 1, slippage_bps=50)
    with pytest.raises(ValueError, match="not in this pool"):
        resolve_swap_params(pool=_Pool(), slot=_Slot(), token_in_id="9field",
                            amount_in=10**9, slippage_bps=50)
    with pytest.raises(ValueError, match="slippage_bps"):
        resolve_swap_params(pool=_Pool(), slot=_Slot(), token_in_id="1field",
                            amount_in=10**9, slippage_bps=10_001)
    with pytest.raises(ValueError, match="sqrt_price_limit"):
        resolve_swap_params(pool=_Pool(), slot=_Slot(), token_in_id="1field",
                            amount_in=10**9, slippage_bps=0,
                            sqrt_price_limit=MIN_SQRT_PRICE - 1)


def test_nonces():
    assert 0 <= generate_swap_nonce() < 2**64
    n = generate_field_nonce()
    assert n.endswith("field") and int(n.removesuffix("field")) < 2**248


def test_parse_token_record_info():
    assert parse_token_record_info(
        "{ owner: aleo1me.private, amount: 5000u128.private, _nonce: 1group.public }"
    ) == {"amount": 5000}
    info = parse_token_record_info(
        "{ owner: aleo1me.private, amount: 7u128.private, token_id: 9field.private, "
        "_nonce: 1group.public }")
    assert info == {"amount": 7, "token_id": "9field"}
    assert parse_token_record_info("{ owner: aleo1me.private, _nonce: 1group.public }") is None
    assert parse_token_record_info("garbage {") is None


class _Programs:
    def __init__(self):
        self.fetches = 0

    def get(self, pid):
        self.fetches += 1
        prog = type("P", (), {"source": f"program {pid};"})()
        return prog


class _Aleo:
    network_name = "testnet"

    def __init__(self, records=()):
        self.programs = _Programs()
        recs = list(records)

        class _Provider:
            def find(self, account=None, *, program=None, unspent=True, **_):
                return recs

        self.record_provider = _Provider()


def test_resolve_imports_memoizes_and_overrides():
    aleo = _Aleo()
    out = resolve_imports(aleo, ["a.aleo", "b.aleo", "a.aleo"])
    assert set(out) == {"a.aleo", "b.aleo"} and aleo.programs.fetches == 2
    out2 = resolve_imports(aleo, ["a.aleo"], overrides={"a.aleo": "override"})
    assert out2["a.aleo"] == "override"
    resolve_imports(aleo, ["b.aleo"])          # cached — no new fetch
    assert aleo.programs.fetches == 2


def _rec(text):
    return {"record_plaintext": text}


def test_select_token_record_picks_smallest_covering():
    small = _rec("{ owner: aleo1me.private, amount: 500u128.private, _nonce: 1group.public }")
    big = _rec("{ owner: aleo1me.private, amount: 9000u128.private, _nonce: 2group.public }")
    chosen = select_token_record(_Aleo([big, small]), program="tok.aleo", min_amount=400)
    assert "500u128" in chosen                 # smallest covering record wins


def test_select_token_record_filters_token_id_and_raises():
    other = _rec("{ owner: aleo1me.private, amount: 9000u128.private, "
                 "token_id: 8field.private, _nonce: 2group.public }")
    with pytest.raises(InsufficientRecordsError):
        select_token_record(_Aleo([other]), program="tok.aleo",
                            min_amount=400, token_id="9field")
    with pytest.raises(InsufficientRecordsError):
        select_token_record(_Aleo([]), program="tok.aleo", min_amount=1)


def test_extract_tx_id_handles_all_dps_shapes():
    import pytest
    from aleo_shield_swap._calls import extract_tx_id

    assert extract_tx_id("at1abc") == "at1abc"
    assert extract_tx_id({"transaction_id": "at1a"}) == "at1a"
    assert extract_tx_id({"id": "at1b"}) == "at1b"
    # current DPS shape: full transaction nested under "transaction"
    assert extract_tx_id({"transaction": {"type": "execute", "id": "at1c",
                                          "execution": {}}}) == "at1c"
    with pytest.raises(ValueError, match="Cannot find"):
        extract_tx_id({"transaction": {"type": "execute"}})
