from types import SimpleNamespace as NS

import pytest

from aleo_shield_swap._core import (
    EMPTY_MERKLE_PROOF,
    default_merkle_proofs,
    generate_field_nonce,
    generate_swap_nonce,
    parse_token_record_info,
    pick_covering_record,
    resolve_imports,
    resolve_swap_params,
    select_token_record,
)
from aleo_shield_swap.errors import InsufficientRecordsError
from aleo_shield_swap.tick_math import MAX_SQRT_RATIO_X128, MIN_SQRT_RATIO_X128


class _Pool:  # duck-typed: only the fields resolve_swap_params reads
    token0 = "1field"
    token1 = "2field"


class _Slot:
    sqrt_price = NS(hi=1, lo=0)   # 1.0 in Q128.128 — price 1.0, raw units


def test_direction_and_spot_estimate():
    r = resolve_swap_params(pool=_Pool(), slot=_Slot(), token_in_id="1field",
                            amount_in=1000, slippage_bps=50)
    assert r.zero_for_one is True and r.token_out_id == "2field"
    # spot at price 1.0: expected 1000 raw; 1000*9950//10000 == 995
    assert r.amount_out_min == 995
    assert r.sqrt_price_limit == MIN_SQRT_RATIO_X128


def test_spot_estimate_uses_x128_price():
    pool = NS(token0="1field", token1="2field")
    # sqrt_price = 2.0 in Q128.128 → price 4.0 token1/token0, raw units.
    slot = NS(sqrt_price=NS(hi=2, lo=0))
    r = resolve_swap_params(pool=pool, slot=slot, token_in_id="1field",
                            amount_in=100, slippage_bps=0)
    assert r.zero_for_one is True and r.amount_out_min == 400
    # reverse direction: price 1/4
    r2 = resolve_swap_params(pool=pool, slot=slot, token_in_id="2field",
                             amount_in=100, slippage_bps=0)
    assert r2.zero_for_one is False and r2.amount_out_min == 25


def test_explicit_quote_and_reverse_direction():
    r = resolve_swap_params(pool=_Pool(), slot=_Slot(), token_in_id="2field",
                            amount_in=5, slippage_bps=100, expected_out=10**9)
    assert r.zero_for_one is False and r.token_out_id == "1field"
    assert r.amount_out_min == 10**9 * 9900 // 10000
    assert r.sqrt_price_limit == MAX_SQRT_RATIO_X128


def test_rejections():
    with pytest.raises(ValueError, match="not in this pool"):
        resolve_swap_params(pool=_Pool(), slot=_Slot(), token_in_id="9field",
                            amount_in=10**9, slippage_bps=50)
    with pytest.raises(ValueError, match="slippage_bps"):
        resolve_swap_params(pool=_Pool(), slot=_Slot(), token_in_id="1field",
                            amount_in=10**9, slippage_bps=10_001)
    with pytest.raises(ValueError, match="sqrt_price_limit"):
        resolve_swap_params(pool=_Pool(), slot=_Slot(), token_in_id="1field",
                            amount_in=10**9, slippage_bps=0,
                            sqrt_price_limit=MIN_SQRT_RATIO_X128 - 1)


def test_default_merkle_proofs_shape():
    assert EMPTY_MERKLE_PROOF == (
        "{ siblings: [" + ", ".join(["0field"] * 16) + "], leaf_index: 1u32 }"
    )
    assert default_merkle_proofs() == f"[{EMPTY_MERKLE_PROOF}, {EMPTY_MERKLE_PROOF}]"


def test_credits_record_amount_parses():
    info = parse_token_record_info(
        "{ owner: aleo1me.private, microcredits: 5000000u64.private, _nonce: 7group.public }")
    assert info == {"amount": 5000000, "recipient_bound": False}


def test_bound_wrapper_records_are_never_selected():
    bound = ("{ owner: aleo1me.private, amount: 900u128.private, "
             "recipient_bound: true.private, bound_recipient: aleo1other.private, "
             "_nonce: 7group.public }")
    free = ("{ owner: aleo1me.private, amount: 900u128.private, "
            "recipient_bound: false.private, bound_recipient: aleo1me.private, "
            "_nonce: 8group.public }")
    recs = [{"record_plaintext": bound}, {"record_plaintext": free}]
    assert pick_covering_record(recs, min_amount=100, token_id=None) == free


def test_nonces():
    assert 0 <= generate_swap_nonce() < 2**64
    n = generate_field_nonce()
    assert n.endswith("field") and int(n.removesuffix("field")) < 2**248


def test_parse_token_record_info():
    assert parse_token_record_info(
        "{ owner: aleo1me.private, amount: 5000u128.private, _nonce: 1group.public }"
    ) == {"amount": 5000, "recipient_bound": False}
    info = parse_token_record_info(
        "{ owner: aleo1me.private, amount: 7u128.private, token_id: 9field.private, "
        "_nonce: 1group.public }")
    assert info == {"amount": 7, "token_id": "9field", "recipient_bound": False}
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


def test_find_position_plaintext_rejects_a_lookalike_record():
    """A non-PositionNFT record carrying a matching `pool` must not be returned
    as a position — it would be spent as one."""
    from aleo_shield_swap._core import find_position_plaintext
    lookalike = "{ owner: aleo1x.private, pool: 5field.private, amount: 9u128.private }"
    position = ("{ owner: aleo1x.private, withdrawal: aleo1y.private, "
                "token_id: 1field.private, token0_id: 2field.private, "
                "token1_id: 3field.private, pool: 5field.private, "
                "tick_lower: -60i32.private, tick_upper: 60i32.private }")
    recs = [{"record_plaintext": lookalike}, {"record_plaintext": position}]
    assert find_position_plaintext(recs, "5field") == position


def test_find_position_plaintext_selects_by_token_id_within_a_pool():
    """Two positions in one pool: a write verb given a token id must get THAT
    NFT, not the first one for the pool — pairing a rebalance plan with the
    wrong record is a guaranteed revert after the proof is paid for."""
    from aleo_shield_swap._core import find_position_plaintext

    def pos(token_id):
        return ("{ owner: aleo1x.private, withdrawal: aleo1y.private, "
                f"token_id: {token_id}.private, token0_id: 2field.private, "
                "token1_id: 3field.private, pool: 5field.private, "
                "tick_lower: -60i32.private, tick_upper: 60i32.private }")

    first, second = pos("1field"), pos("9field")
    recs = [{"record_plaintext": first}, {"record_plaintext": second}]
    assert find_position_plaintext(recs, "5field") == first                      # legacy: first in pool
    assert find_position_plaintext(recs, "5field", "9field") == second
    assert find_position_plaintext(recs, "5field", "7field") is None             # not held → None, never a substitute
    assert find_position_plaintext(recs, "6field", "9field") is None             # pool must match too
