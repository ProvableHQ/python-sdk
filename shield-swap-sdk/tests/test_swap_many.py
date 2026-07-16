import pytest

from aleo_shield_swap import ShieldSwap
from aleo_shield_swap.journal import Journal
from aleo_shield_swap.profile import Profile
from aleo_shield_swap.types import SwapHandle


RECORDS = [
    {"record_plaintext": f"{{ owner: aleo1x.private, amount: {n}u128.private }}"}
    for n in (100, 101, 102)
]


class _Provider:
    def find(self, account, program=None, unspent=True):
        return list(RECORDS)


class _Facade:
    network_name = "testnet"
    record_provider = _Provider()


@pytest.fixture
def dex(tmp_path, monkeypatch):
    d = ShieldSwap(_Facade())
    d.profile = Profile.load_or_create(tmp_path / "home")
    d.journal = Journal(d.profile.journal_path)
    monkeypatch.setattr(ShieldSwap, "_account", lambda self, a=None: object())
    monkeypatch.setattr(
        "aleo_shield_swap.client.blinded_identity_at",
        lambda aleo, acct, prog, c: type(
            "I", (), {"counter": c, "blinding_factor": f"bf{c}",
                      "blinded_address": f"ba{c}"})())
    monkeypatch.setattr(ShieldSwap, "get_pool",
                        lambda self, key: type("P", (), {"token0": "t0",
                                                         "token1": "t1"})())
    monkeypatch.setattr(ShieldSwap, "_quote_expected_out",
                        lambda self, **kw: None)
    monkeypatch.setattr(ShieldSwap, "_token_program",
                        lambda self, token_id: "tok.aleo")
    return d


def _fake_swap_factory(fail_counters=()):
    calls = []

    def fake_swap(self, *, pool_key, token_in_id, amount_in, identity=None, **kw):
        calls.append(identity.counter)

        class _Call:
            def delegate(inner, account=None, **kw):
                if identity.counter in fail_counters:
                    raise RuntimeError(f"boom at {identity.counter}")
                return SwapHandle(swap_id=f"s{identity.counter}",
                                  blinding_factor=identity.blinding_factor,
                                  blinded_address=identity.blinded_address,
                                  token_in_id=token_in_id, token_out_id="t1",
                                  pool_key=pool_key, amount_in=amount_in,
                                  transaction_id=f"tx{identity.counter}",
                                  program="shield_swap_v3.aleo")
        return _Call()

    return fake_swap, calls


def test_swap_many_reserves_distinct_counters_and_journals(dex, monkeypatch):
    fake, calls = _fake_swap_factory()
    monkeypatch.setattr(ShieldSwap, "swap", fake)
    report = dex.swap_many(pool_key="1field", token_in_id="t0",
                           amount_in=5, count=3)
    assert calls == [0, 1, 2]
    assert [h.swap_id for h in report.handles] == ["s0", "s1", "s2"]
    assert report.failures == []
    assert [h.swap_id for h in dex.journal.pending_claims()] == ["s0", "s1", "s2"]


def test_swap_many_burns_failed_counter_and_continues(dex, monkeypatch):
    fake, calls = _fake_swap_factory(fail_counters={1})
    monkeypatch.setattr(ShieldSwap, "swap", fake)
    report = dex.swap_many(pool_key="1field", token_in_id="t0",
                           amount_in=5, count=3)
    assert [h.swap_id for h in report.handles] == ["s0", "s2"]
    assert report.failures == [{"counter": 1, "error": "boom at 1"}]
    assert dex.journal.counter_cursor() == 3           # 1 burned, not reused
    assert [h.swap_id for h in dex.journal.pending_claims()] == ["s0", "s2"]


def test_swap_many_requires_journal(dex):
    dex.journal = None
    with pytest.raises(ValueError, match="from_profile"):
        dex.swap_many(pool_key="1field", token_in_id="t0", amount_in=5, count=2)


def test_swap_many_quotes_route_for_expected_out(dex, monkeypatch):
    monkeypatch.setattr(ShieldSwap, "_quote_expected_out",
                        lambda self, **kw: 990)
    seen = []

    def fake_swap(self, *, expected_out=None, identity=None, **kw):
        seen.append(expected_out)

        class _Call:
            def delegate(inner, account=None, **kw):
                return SwapHandle(swap_id=f"s{identity.counter}",
                                  blinding_factor="bf", blinded_address="ba",
                                  token_in_id="t0", token_out_id="t1",
                                  pool_key="1field", amount_in=5,
                                  transaction_id="tx", program="p")
        return _Call()

    monkeypatch.setattr(ShieldSwap, "swap", fake_swap)
    dex.swap_many(pool_key="1field", token_in_id="t0", amount_in=5, count=2)
    assert seen == [990, 990]              # quoted once, applied to every swap


def test_quote_expected_out_converts_units_both_ways(tmp_path, monkeypatch):
    # fresh client: the shared fixture stubs _quote_expected_out itself
    dex = ShieldSwap(_Facade())
    monkeypatch.setattr(dex.api, "get_tokens", lambda: [
        type("T", (), {"address": "tin", "decimals": 18})(),
        type("T", (), {"address": "tout", "decimals": 6})()])
    seen = {}

    def fake_route(*, token_in, token_out, amount_in):
        seen["amount_in"] = str(amount_in)
        return type("R", (), {"estimated_amount_out": "1089.461274"})()

    monkeypatch.setattr(dex.api, "get_route", fake_route)
    out = dex._quote_expected_out(token_in_id="tin", token_out_id="tout",
                                  amount_in=10**19)
    assert seen["amount_in"] == "10"           # base -> canonical decimal
    assert out == 1089461274                   # canonical -> base units
    assert dex._quote_expected_out(token_in_id="unknown", token_out_id="tout",
                                   amount_in=1) is None


def test_swap_many_partitions_distinct_records(dex, monkeypatch):
    fake, _ = _fake_swap_factory()
    seen_records = []
    orig = fake

    def spy(self, *, token_record=None, **kw):
        seen_records.append(token_record)
        return orig(self, **kw)

    monkeypatch.setattr(ShieldSwap, "swap", spy)
    report = dex.swap_many(pool_key="1field", token_in_id="t0",
                           amount_in=5, count=3)
    assert len(report.handles) == 3
    assert len(set(seen_records)) == 3     # every swap spends its own record


def test_swap_many_fails_cleanly_when_records_run_out(dex, monkeypatch):
    fake, _ = _fake_swap_factory()
    monkeypatch.setattr(ShieldSwap, "swap", fake)

    class _Net:
        def wait_for_transaction(self, tx_id, timeout=180.0):
            return None

    dex._aleo.network = _Net()
    report = dex.swap_many(pool_key="1field", token_in_id="t0",
                           amount_in=5, count=5, record_wait_seconds=0)
    assert len(report.handles) == 3        # one per distinct record
    assert len(report.failures) == 2
    assert all("distinct unspent record" in f["error"] for f in report.failures)
