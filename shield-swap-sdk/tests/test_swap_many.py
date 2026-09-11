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
    # A real batch always has a quote; swap_many refuses without one, because
    # the spot fallback sets amount_out_min above what the pool can pay.
    monkeypatch.setattr(ShieldSwap, "_quote_expected_out",
                        lambda self, **kw: 990_000)
    monkeypatch.setattr(ShieldSwap, "_token_program",
                        lambda self, token_id: "tok.aleo")
    # Nothing used on chain unless a test says otherwise.
    monkeypatch.setattr(ShieldSwap, "_blinded_address_used", lambda self, ba: False)
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
                                  program="shield_swap.aleo")
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

    def fake_route(*, token_in, token_out, amount_in, pool_key=None):
        seen["amount_in"] = str(amount_in)
        seen["pool_key"] = pool_key
        return type("R", (), {"estimated_amount_out": "1089.461274"})()

    monkeypatch.setattr(dex.api, "get_route", fake_route)
    out = dex._quote_expected_out(token_in_id="tin", token_out_id="tout",
                                  amount_in=10**19, pool_key="5field")
    assert seen["amount_in"] == "10"           # base -> canonical decimal
    # The quote is PINNED to the pool being traded: /route otherwise answers
    # with the router's best path (possibly multi-hop through a deeper pool),
    # and a floor derived from that is one the traded pool cannot pay — the
    # swap proves, broadcasts, and is rejected at finalize (seen live on
    # testnet, 2026-09-03: a 2-hop quote 165x the direct pool's output).
    assert seen["pool_key"] == "5field"
    assert out == 1089461274                   # canonical -> base units
    assert dex._quote_expected_out(token_in_id="unknown", token_out_id="tout",
                                   amount_in=1, pool_key="5field") is None


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


# ── Quote failures must not become bad trade parameters ──────────────────────

def test_swap_many_refuses_without_a_quote(dex, monkeypatch):
    """A missing quote would set amount_out_min above what the pool can pay, so
    every swap would be proved, broadcast and rejected. Refuse first."""
    from aleo_shield_swap.errors import ShieldSwapError

    monkeypatch.setattr(ShieldSwap, "_quote_expected_out", lambda self, **kw: None)
    with pytest.raises(ShieldSwapError, match="no route quote"):
        dex.swap_many(pool_key="5field", token_in_id="t0",
                      amount_in=10**6, count=3)
    assert dex.journal.counter_cursor() == 0      # nothing reserved or spent


def test_swap_many_allows_no_quote_at_full_slippage(dex, monkeypatch):
    """slippage_bps=10000 means "accept any output", so no quote is needed."""
    monkeypatch.setattr(ShieldSwap, "_quote_expected_out", lambda self, **kw: None)
    dex.swap_many(pool_key="5field", token_in_id="t0", amount_in=10**6,
                  count=1, slippage_bps=10_000)
    assert dex.journal.counter_cursor() == 1      # it proceeded


def test_quote_propagates_auth_failure():
    """'could not ask' is not 'no route' — surfacing beats a bad min-out.

    Built outside the ``dex`` fixture on purpose: that fixture patches
    ``_quote_expected_out`` on the class, which would mask the real method.
    """
    from aleo_shield_swap.errors import NotAuthenticatedError

    class _Api:
        def get_route(self, **_):
            raise NotAuthenticatedError("no session")

    fresh = ShieldSwap(_Facade())
    fresh.api = _Api()
    fresh._token_decimals = lambda _tid: 6   # registry known, route not askable
    with pytest.raises(NotAuthenticatedError):
        fresh._quote_expected_out(token_in_id="t0", token_out_id="t1",
                                  amount_in=10**6, pool_key="5field")


def test_swap_many_accepts_a_caller_supplied_quote(dex, monkeypatch):
    """expected_out skips the route quote — the escape hatch the refusal
    message points callers at, so it must actually exist."""
    called = {"n": 0}

    def _never(self, **kw):
        called["n"] += 1
        return None

    monkeypatch.setattr(ShieldSwap, "_quote_expected_out", _never)
    dex.swap_many(pool_key="5field", token_in_id="t0", amount_in=10**6,
                  count=1, expected_out=990_000)
    assert called["n"] == 0, "should not have quoted"
    assert dex.journal.counter_cursor() == 1


def test_swap_many_skips_counters_already_used_on_chain(dex, monkeypatch):
    """A fresh journal for an account with history reserved counters 0 and 1
    — both long used on chain — and both swaps were rejected at finalize
    (live, 2026-09-03).  Reservation must seed the cursor past the used run
    and verify each identity against the chain before spending a proof."""
    used = {f"ba{c}" for c in range(5)} | {"ba7"}     # 0..4 used, gap at 5,6, 7 used
    monkeypatch.setattr(ShieldSwap, "_blinded_address_used",
                        lambda self, ba: ba in used)
    fake, calls = _fake_swap_factory()
    monkeypatch.setattr(ShieldSwap, "swap", fake)
    report = dex.swap_many(pool_key="1field", token_in_id="t0", amount_in=5, count=3)
    assert calls == [5, 6, 8]                         # skipped 0..4 and 7
    assert [h.swap_id for h in report.handles] == ["s5", "s6", "s8"]
    assert report.failures == []
    assert dex.journal.counter_cursor() == 9
    # The skipped-used counter 7 is burned in the journal, never reissued.
    assert 7 in {e.get("counter") for e in dex.journal.events() if e["type"] == "swap_failed"}


def test_reserve_identities_is_shared_by_single_swaps(dex, monkeypatch):
    used = {"ba0", "ba1"}
    monkeypatch.setattr(ShieldSwap, "_blinded_address_used",
                        lambda self, ba: ba in used)
    idents = dex._reserve_identities(object(), 2)
    assert [i.counter for i in idents] == [2, 3]
    assert dex.journal.counter_cursor() == 4


def test_swap_many_tolerates_a_transient_scanner_error(dex, monkeypatch):
    """One reset connection to the record scanner mid-batch must not abort a
    batch whose earlier swaps are already broadcast (live, 2026-09-03:
    'Connection aborted' from the scanner between swap 1 and swap 2)."""
    calls = {"n": 0}
    real_find = dex._aleo.record_provider.find

    def flaky_find(account, program=None, unspent=True):
        calls["n"] += 1
        if calls["n"] == 2:
            raise ConnectionError("Connection aborted.")
        return real_find(account, program=program, unspent=unspent)

    monkeypatch.setattr(dex._aleo.record_provider, "find", flaky_find)
    monkeypatch.setattr("aleo_shield_swap.client.time.sleep", lambda s: None)
    fake, calls_made = _fake_swap_factory()
    monkeypatch.setattr(ShieldSwap, "swap", fake)
    report = dex.swap_many(pool_key="1field", token_in_id="t0", amount_in=5, count=3)
    assert calls_made == [0, 1, 2] and report.failures == []
