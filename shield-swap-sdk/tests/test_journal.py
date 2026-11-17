import json
import stat
import threading

from aleo_shield_swap.journal import Journal
from aleo_shield_swap.types import SwapHandle


def _handle(swap_id="s1", **kw):
    base = dict(swap_id=swap_id, blinding_factor="bf", blinded_address="ba",
                token_in_id="t0", token_out_id="t1", pool_key="pk",
                amount_in=5, transaction_id="tx1", program="shield_swap.aleo")
    base.update(kw)
    return SwapHandle(**base)


def test_append_is_jsonl_and_replayable(tmp_path):
    j = Journal(tmp_path / "journal.jsonl")
    j.append("stage", name="authenticate", action="ran")
    lines = (tmp_path / "journal.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["type"] == "stage"
    assert Journal(tmp_path / "journal.jsonl").events()[0]["name"] == "authenticate"


def test_reserve_counters_monotonic_and_persistent(tmp_path):
    j = Journal(tmp_path / "journal.jsonl")
    assert j.reserve_counters(3) == [0, 1, 2]
    assert j.reserve_counters(2) == [3, 4]
    j2 = Journal(tmp_path / "journal.jsonl")          # fresh process
    assert j2.reserve_counters(1) == [5]
    assert j2.counter_cursor() == 6


def test_reserve_counters_thread_safe(tmp_path):
    got: list[int] = []
    lock = threading.Lock()

    def grab():
        counters = Journal(tmp_path / "journal.jsonl").reserve_counters(5)
        with lock:
            got.extend(counters)

    ts = [threading.Thread(target=grab) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(got) == list(range(20))              # no counter issued twice


def test_pending_claims_shrink_after_claim(tmp_path):
    j = Journal(tmp_path / "journal.jsonl")
    j.record_swap(_handle("s1"), counter=0)
    j.record_swap(_handle("s2", transaction_id="tx2"), counter=1)
    assert {h.swap_id for h in j.pending_claims()} == {"s1", "s2"}
    j.record_claim("s1", "txc", amount_out=42)
    pending = j.pending_claims()
    assert [h.swap_id for h in pending] == ["s2"]
    assert isinstance(pending[0], SwapHandle)
    assert pending[0].transaction_id == "tx2"


def test_open_positions_close_on_burn(tmp_path):
    j = Journal(tmp_path / "journal.jsonl")
    j.record_position("p1field", "poolA", "tx1")
    j.record_position("p2field", "poolB", "tx2")
    j.record_position_burned("p1field", "tx3")
    assert [p["position_token_id"] for p in j.open_positions()] == ["p2field"]


def test_failed_swap_burns_counter_and_is_not_pending(tmp_path):
    j = Journal(tmp_path / "journal.jsonl")
    cs = j.reserve_counters(1)
    j.record_swap_failed(cs[0], "boom")
    assert j.pending_claims() == []
    assert j.reserve_counters(1) == [1]                # 0 never reused


def test_skip_counters_through_advances_the_cursor(tmp_path):
    """A fresh journal for an account with swap history must start past the
    counters already consumed on chain; skipping is durable and never
    reissues the skipped range."""
    j = Journal(tmp_path / "journal.jsonl")
    j.skip_counters_through(9)
    assert j.counter_cursor() == 10
    assert j.reserve_counters(2) == [10, 11]
    assert Journal(tmp_path / "journal.jsonl").counter_cursor() == 12
    j.skip_counters_through(5)                       # never moves backwards
    assert j.counter_cursor() == 12


# Journals of one account sharing a counter ledger

def _on_ledger(tmp_path, home):
    """Open the journal of one profile home on the ledger every home shares."""
    return Journal(tmp_path / home / "journal.jsonl", counters=tmp_path / "counters" / "account.jsonl")


def test_journals_on_one_ledger_never_issue_the_same_counter(tmp_path):
    one, two = _on_ledger(tmp_path, "one"), _on_ledger(tmp_path, "two")
    assert one.reserve_counters(2) == [0, 1]
    assert two.reserve_counters(2) == [2, 3]
    assert one.reserve_counters(1) == [4]
    assert one.counter_cursor() == two.counter_cursor() == 5
    assert _on_ledger(tmp_path, "three").reserve_counters(1) == [5]     # a home opened later


def test_ledger_directory_is_readable_by_its_owner_only(tmp_path):
    _on_ledger(tmp_path, "one").reserve_counters(1)
    assert stat.S_IMODE((tmp_path / "counters").stat().st_mode) == 0o700


def test_journals_on_one_ledger_are_safe_across_threads(tmp_path):
    got: list[int] = []
    lock = threading.Lock()

    def grab(home):
        counters = _on_ledger(tmp_path, home).reserve_counters(5)
        with lock:
            got.extend(counters)

    ts = [threading.Thread(target=grab, args=(f"home-{n}",)) for n in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(got) == list(range(20))              # no counter issued twice


def test_skipped_counters_reach_every_journal_on_the_ledger(tmp_path):
    one, two = _on_ledger(tmp_path, "one"), _on_ledger(tmp_path, "two")
    one.skip_counters_through(9)
    assert two.counter_cursor() == 10
    assert two.reserve_counters(1) == [10]
    two.skip_counters_through(5)                       # never moves backwards
    assert one.counter_cursor() == 11


def test_journal_ahead_of_the_ledger_starts_past_its_own_history(tmp_path):
    # A journal that issued counters before it joined a ledger keeps them spent.
    Journal(tmp_path / "one" / "journal.jsonl").reserve_counters(8)
    one, two = _on_ledger(tmp_path, "one"), _on_ledger(tmp_path, "two")
    assert one.reserve_counters(1) == [8]
    assert two.reserve_counters(1) == [9]


def test_opening_a_journal_on_a_ledger_publishes_the_counters_it_already_issued(tmp_path):
    # Counters reserved before the ledger existed may belong to swaps that have not finalized.
    assert Journal(tmp_path / "one" / "journal.jsonl").reserve_counters(2) == [0, 1]
    _on_ledger(tmp_path, "one")                         # opened, never asked for a counter
    assert _on_ledger(tmp_path, "two").reserve_counters(1) == [2]


def test_opening_a_new_journal_on_a_ledger_writes_nothing(tmp_path):
    _on_ledger(tmp_path, "one")
    assert list(tmp_path.iterdir()) == []


def test_reopening_a_journal_on_a_ledger_publishes_its_counters_once(tmp_path):
    Journal(tmp_path / "one" / "journal.jsonl").reserve_counters(2)
    for _ in range(3):
        _on_ledger(tmp_path, "one")
    assert len((tmp_path / "counters" / "account.jsonl").read_text().splitlines()) == 1


def test_swap_recorded_on_a_caller_chosen_counter_reaches_the_ledger(tmp_path):
    one, two = _on_ledger(tmp_path, "one"), _on_ledger(tmp_path, "two")
    one.record_swap(_handle("s1"), counter=7)           # an identity the ledger did not issue
    assert two.reserve_counters(1) == [8]
    two.record_swap_failed(11, "boom")
    assert one.reserve_counters(1) == [12]


def test_each_journal_records_the_counters_it_was_issued(tmp_path):
    # Without the ledger file, a journal still never reissues its own counters.
    one, two = _on_ledger(tmp_path, "one"), _on_ledger(tmp_path, "two")
    one.reserve_counters(2)
    two.reserve_counters(2)
    assert Journal(tmp_path / "two" / "journal.jsonl").counter_cursor() == 4
    assert Journal(tmp_path / "one" / "journal.jsonl").counter_cursor() == 2


def test_ledger_holds_counters_and_nothing_about_the_swaps(tmp_path):
    one = _on_ledger(tmp_path, "one")
    (counter,) = one.reserve_counters(1)
    one.record_swap(_handle("s1"), counter=counter)
    one.record_swap_failed(one.reserve_counters(1)[0], "boom")
    one.skip_counters_through(6)
    events = [json.loads(line) for line in (tmp_path / "counters" / "account.jsonl").read_text().splitlines()]
    assert [sorted(event) for event in events] == [
        ["counters", "ts", "type"], ["counters", "ts", "type"], ["through", "ts", "type"]]
    assert [h.swap_id for h in one.pending_claims()] == ["s1"]
