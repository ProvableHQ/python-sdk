import pytest

from aleo_shield_swap import ShieldSwap
from aleo_shield_swap.journal import Journal
from aleo_shield_swap.profile import Profile
from aleo_shield_swap.types import SwapHandle


class _Facade:
    network_name = "testnet"


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
    return d


def _fake_swap_factory(fail_counters=()):
    calls = []

    def fake_swap(self, *, pool_key, token_in_id, amount_in, identity=None, **kw):
        calls.append(identity.counter)

        class _Call:
            def delegate(inner, account=None):
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
