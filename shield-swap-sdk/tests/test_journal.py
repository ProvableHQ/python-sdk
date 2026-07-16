import json
import threading

from aleo_shield_swap.journal import Journal
from aleo_shield_swap.types import SwapHandle


def _handle(swap_id="s1", **kw):
    base = dict(swap_id=swap_id, blinding_factor="bf", blinded_address="ba",
                token_in_id="t0", token_out_id="t1", pool_key="pk",
                amount_in=5, transaction_id="tx1", program="shield_swap_v3.aleo")
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
