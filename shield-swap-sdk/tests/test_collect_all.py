import pytest

from aleo_shield_swap import ShieldSwap
from aleo_shield_swap.errors import SwapOutputNotFinalizedError
from aleo_shield_swap.journal import Journal
from aleo_shield_swap.profile import Profile
from aleo_shield_swap.types import ClaimResult, PositionView, SwapHandle, TxResult


def _handle(sid):
    return SwapHandle(swap_id=sid, blinding_factor="bf", blinded_address="ba",
                      token_in_id="t0", token_out_id="t1", pool_key="1field",
                      amount_in=5, transaction_id="tx", program="p")


class _Facade:
    network_name = "testnet"


@pytest.fixture
def dex(tmp_path, monkeypatch):
    d = ShieldSwap(_Facade())
    d.profile = Profile.load_or_create(tmp_path / "home")
    d.journal = Journal(d.profile.journal_path)
    monkeypatch.setattr(ShieldSwap, "_account", lambda self, a=None: object())
    return d


def test_collect_all_claims_finalized_skips_pending(dex, monkeypatch):
    dex.journal.record_swap(_handle("ok"), 0)
    dex.journal.record_swap(_handle("early"), 1)

    def fake_claim(self, handle, **kw):
        class _Call:
            def delegate(inner, account=None):
                if handle.swap_id == "early":
                    raise SwapOutputNotFinalizedError("early")
                return ClaimResult("txc", 42, 0)
        return _Call()

    monkeypatch.setattr(ShieldSwap, "claim_swap_output", fake_claim)
    monkeypatch.setattr(ShieldSwap, "get_positions",
                        lambda self, account=None: [])

    report = dex.collect_all()
    assert report.claimed == [{"swap_id": "ok", "transaction_id": "txc",
                               "amount_out": 42}]
    assert report.still_pending == ["early"]
    assert [h.swap_id for h in dex.journal.pending_claims()] == ["early"]


def test_collect_all_rerun_is_idempotent(dex, monkeypatch):
    dex.journal.record_swap(_handle("ok"), 0)
    calls = []

    def fake_claim(self, handle, **kw):
        calls.append(handle.swap_id)

        class _Call:
            def delegate(inner, account=None):
                return ClaimResult("txc", 42, 0)
        return _Call()

    monkeypatch.setattr(ShieldSwap, "claim_swap_output", fake_claim)
    monkeypatch.setattr(ShieldSwap, "get_positions",
                        lambda self, account=None: [])
    dex.collect_all()
    report2 = dex.collect_all()
    assert calls == ["ok"]                            # not re-claimed
    assert report2.claimed == [] and report2.still_pending == []


def test_collect_all_requests_exactly_owed_fees(dex, monkeypatch):
    class _Pos:
        tokens_owed0 = 3
        tokens_owed1 = 0

    class _Pool:
        scale0 = 1000
        scale1 = 10

    monkeypatch.setattr(ShieldSwap, "get_positions", lambda self, account=None: [
        PositionView("11field", "1field", "journal"),
        PositionView("22field", "2field", "scanned"),   # skipped: no journal context
    ])
    monkeypatch.setattr(ShieldSwap, "_position_state",
                        lambda self, pid: _Pos() if pid == "11field" else None)
    monkeypatch.setattr(ShieldSwap, "get_pool", lambda self, key: _Pool())
    collected = []

    def fake_collect(self, *, pool_key, amount0_requested, amount1_requested,
                     account=None, **kw):
        collected.append((pool_key, amount0_requested, amount1_requested))

        class _Call:
            def delegate(inner, account=None):
                return TxResult("11field", "txf")
        return _Call()

    monkeypatch.setattr(ShieldSwap, "collect", fake_collect)
    report = dex.collect_all()
    assert collected == [("1field", 3000, 0)]          # owed * scale, exactly
    assert report.fees == [{"position_token_id": "11field",
                            "pool_key": "1field", "transaction_id": "txf"}]


def test_collect_all_skips_zero_owed_positions(dex, monkeypatch):
    class _Pos:
        tokens_owed0 = 0
        tokens_owed1 = 0

    monkeypatch.setattr(ShieldSwap, "get_positions", lambda self, account=None: [
        PositionView("11field", "1field", "journal")])
    monkeypatch.setattr(ShieldSwap, "_position_state",
                        lambda self, pid: _Pos())
    monkeypatch.setattr(ShieldSwap, "collect",
                        lambda self, **kw: pytest.fail("must not collect zero"))
    report = dex.collect_all()
    assert report.fees == []
