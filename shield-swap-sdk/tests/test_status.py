import pytest

from aleo_shield_swap import ShieldSwap
from aleo_shield_swap.journal import Journal
from aleo_shield_swap.profile import Profile

POS_A = "{ owner: aleo1x.private, pool: 1field.private, tick_lower: -60i32.private, tick_upper: 60i32.private, liquidity: 5u128.private, position_token_id: 11field.private }"
POS_B = "{ owner: aleo1x.private, pool: 2field.private, tick_lower: -10i32.private, tick_upper: 10i32.private, liquidity: 9u128.private, position_token_id: 22field.private }"


class _Provider:
    def find(self, account, program=None, unspent=True):
        return [{"record_plaintext": POS_A}, {"record_plaintext": POS_B}]


class _Account:
    address = "aleo1x"


class _Facade:
    record_provider = _Provider()
    default_account = _Account()
    network_name = "testnet"


@pytest.fixture
def dex(tmp_path):
    d = ShieldSwap(_Facade())
    d.profile = Profile.load_or_create(tmp_path / "home")
    d.journal = Journal(d.profile.journal_path)
    return d


def test_get_positions_merges_journal_and_scan(dex):
    dex.journal.record_position("11field", "1field", "tx1")
    views = dex.get_positions()
    by_id = {v.position_token_id: v.source for v in views}
    assert by_id == {"11field": "journal", "22field": "scanned"}
    assert {v.pool_key for v in views} == {"1field", "2field"}


def test_status_reorients_from_disk(dex, monkeypatch):
    dex.api.set_token("jwt")
    monkeypatch.setattr(dex.api, "access_status",
                        lambda: type("S", (), {"has_access": True})())
    monkeypatch.setattr(dex, "get_balances", lambda: {"tok": {"total": 5}})
    dex.journal.append(
        "swap", counter=0, swap_id="s1", blinding_factor="bf",
        blinded_address="ba", token_in_id="t0", token_out_id="t1",
        pool_key="1field", amount_in=5, transaction_id="tx", program="p")
    st = dex.status()
    assert st.authenticated is True and st.has_access is True
    assert st.pending_claim_ids == ["s1"]
    assert st.counter_cursor == 1
    assert st.balances == {"tok": {"total": 5}}
    assert st.address == dex.profile.address


def test_status_survives_unauthenticated_api(dex, monkeypatch):
    def boom():
        raise AssertionError("must not be called without a token")

    monkeypatch.setattr(dex.api, "access_status", boom)
    monkeypatch.setattr(dex, "get_balances", lambda: {})
    st = dex.status()                      # no token set
    assert st.authenticated is False and st.has_access is None
