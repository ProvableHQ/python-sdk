"""Hermetic tests for the live-funds harness itself (port of veil `test/integration/live/helpers.test.ts`).

Nothing here touches a network, a key, or a chain: the gates are exercised with a monkeypatched
environment, the state files in ``tmp_path``, the Hyperlane explorer through an injected HTTP
callable, and the rehearsal CLI against ``FakeBridge``.  The harness is tested BEFORE the funded
cases can run it, which is the whole point of the file: a bug in the gate is a bug that spends
real money.
"""
from __future__ import annotations

import json
import os
import stat

import pytest

from tests.live import config as live_config
from tests.live import helpers as live_helpers

FUNDS = "BRIDGE_LIVE_FUNDS"
STATE_DIR = "BRIDGE_LIVE_STATE_DIR"
ACK = "BRIDGE_LIVE_MAINNET_ACK"
CASES = "BRIDGE_LIVE_MAINNET_CASES"
EXECUTE = "BRIDGE_LIVE_MAINNET_EXECUTE"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every gate variable starts unset: the truth tables below set exactly what they test."""
    for name in (FUNDS, STATE_DIR, ACK, CASES, EXECUTE, "TEST_EVM_KEY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


# ── config.py: reading required values without ever logging them ──────────────

def test_required_names_the_variable_and_never_the_value(monkeypatch):
    monkeypatch.setenv("TEST_EVM_KEY", "  value-with-space  ")
    assert live_config.required("TEST_EVM_KEY") == "value-with-space"

    with pytest.raises(live_config.LiveConfigError) as excinfo:
        live_config.required("BRIDGE_LIVE_ABSENT")
    assert "BRIDGE_LIVE_ABSENT" in str(excinfo.value)


def test_required_treats_whitespace_only_as_missing(monkeypatch):
    monkeypatch.setenv("TEST_EVM_KEY", "   ")
    with pytest.raises(live_config.LiveConfigError):
        live_config.required("TEST_EVM_KEY")


def test_normalizes_prefixed_and_unprefixed_evm_private_keys(monkeypatch):
    """veil helpers.test.ts:14-20 — the 0x prefix is optional, the value is never echoed."""
    key = "ab" * 32
    monkeypatch.setenv("TEST_EVM_KEY", key)
    assert live_config.required_evm_private_key("TEST_EVM_KEY") == f"0x{key}"
    monkeypatch.setenv("TEST_EVM_KEY", f"0X{key}")
    assert live_config.required_evm_private_key("TEST_EVM_KEY") == f"0x{key}"


def test_rejects_a_malformed_evm_private_key_without_printing_it(monkeypatch):
    monkeypatch.setenv("TEST_EVM_KEY", "deadbeef")
    with pytest.raises(live_config.LiveConfigError) as excinfo:
        live_config.required_evm_private_key("TEST_EVM_KEY")
    message = str(excinfo.value)
    assert "TEST_EVM_KEY" in message and "32" in message and "deadbeef" not in message


# ── config.py: the gate truth table ───────────────────────────────────────────

def test_live_funds_needs_both_the_flag_and_a_state_dir(monkeypatch):
    assert live_config.live_funds_enabled() is False
    monkeypatch.setenv(FUNDS, "1")
    assert live_config.live_funds_enabled() is False          # no state dir
    monkeypatch.setenv(STATE_DIR, "/tmp/bridge-state")
    assert live_config.live_funds_enabled() is True
    monkeypatch.setenv(FUNDS, "true")
    assert live_config.live_funds_enabled() is False          # exactly "1", like veil


def test_mainnet_case_requires_funding_state_acknowledgement_and_the_named_case(monkeypatch):
    """veil helpers.test.ts:56-64."""
    monkeypatch.setenv(FUNDS, "1")
    monkeypatch.setenv(STATE_DIR, "/tmp/bridge-state")
    monkeypatch.setenv(ACK, "I_ACKNOWLEDGE_BRIDGE_MAINNET_FUNDS")
    monkeypatch.setenv(CASES, "evm-xreserve, aleo-hyperlane")

    assert live_config.mainnet_case_enabled("evm-xreserve") is True
    assert live_config.mainnet_case_enabled("aleo-hyperlane") is True
    assert live_config.mainnet_case_enabled("solana-hyperlane") is False

    monkeypatch.setenv(ACK, "yes")
    assert live_config.mainnet_case_enabled("evm-xreserve") is False
    monkeypatch.setenv(ACK, "I_ACKNOWLEDGE_BRIDGE_MAINNET_FUNDS")
    monkeypatch.delenv(FUNDS)
    assert live_config.mainnet_case_enabled("evm-xreserve") is False


def test_execution_requires_a_separate_exact_acknowledgement(monkeypatch):
    """veil helpers.test.ts:66-71 — nothing in this repo ever SETS this variable."""
    assert live_config.mainnet_execution_enabled() is False
    monkeypatch.setenv(EXECUTE, "yes")
    assert live_config.mainnet_execution_enabled() is False
    monkeypatch.setenv(EXECUTE, "I_ACKNOWLEDGE_THIS_SUBMITS_MAINNET_TRANSACTIONS")
    assert live_config.mainnet_execution_enabled() is True


def test_gates_only_read_the_environment(monkeypatch):
    """No gate may write, default or repair a variable — a gate that sets its own key is not a gate."""
    before = dict(os.environ)
    live_config.live_funds_enabled()
    live_config.mainnet_case_enabled("evm-hyperlane")
    live_config.mainnet_execution_enabled()
    assert dict(os.environ) == before


def test_case_names_are_veils_five_mainnet_cases():
    assert live_config.CASE_NAMES == ("evm-hyperlane", "evm-xreserve", "aleo-hyperlane",
                                      "aleo-xreserve", "solana-hyperlane")


def test_one_atomic_unit_per_asset_precision():
    """veil helpers.test.ts:73-77."""
    assert live_config.one_atomic_unit(0) == "1"
    assert live_config.one_atomic_unit(6) == "0.000001"
    assert live_config.one_atomic_unit(9) == "0.000000001"
    assert live_config.one_atomic_unit(18) == "0.000000000000000001"
    for bad in (-1, 1.5, True):
        with pytest.raises(live_config.LiveConfigError):
            live_config.one_atomic_unit(bad)


def test_live_state_path_is_namespaced_by_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(STATE_DIR, str(tmp_path))
    assert live_config.live_state_path("mainnet", "evm-xreserve") == tmp_path / "mainnet" / "evm-xreserve.json"
    with pytest.raises(live_config.LiveConfigError):
        live_config.live_state_path("devnet", "evm-xreserve")
    monkeypatch.delenv(STATE_DIR)
    with pytest.raises(live_config.LiveConfigError):
        live_config.live_state_path("mainnet", "evm-xreserve")


def test_route_and_recipient_overrides_are_optional(monkeypatch):
    assert live_config.case_route_override("evm-hyperlane") is None
    monkeypatch.setenv("BRIDGE_LIVE_EVM_HYPERLANE_ROUTE_ID", "hyperlane:ethereum/wbtc->aleo/wbtc")
    assert live_config.case_route_override("evm-hyperlane") == "hyperlane:ethereum/wbtc->aleo/wbtc"

    assert live_config.recipient_override("aleo") is None
    monkeypatch.setenv("BRIDGE_LIVE_ALEO_MAINNET_RECIPIENT", "aleo1recipient")
    assert live_config.recipient_override("aleo") == "aleo1recipient"
    with pytest.raises(live_config.LiveConfigError):
        live_config.recipient_override("bitcoin")


# ── helpers.py: state files ───────────────────────────────────────────────────

def test_state_round_trips_and_starts_empty_for_an_absent_file(tmp_path):
    """veil helpers.test.ts:34-41."""
    path = tmp_path / "mainnet" / "state.json"
    assert live_helpers.load_live_state(path, "route:a") == live_helpers.LiveState(route_id="route:a")

    state = live_helpers.LiveState(route_id="route:a", source_tx_id="source-1")
    live_helpers.save_live_state(path, state)
    assert live_helpers.load_live_state(path, "route:a") == state


def test_state_keeps_every_recorded_field_across_a_reload(tmp_path):
    path = tmp_path / "state.json"
    state = live_helpers.LiveState(
        route_id="route:a", source_tx_id="0xsource", message_id="0xmessage",
        destination_tx_id="at1destination", destination_balance_before="1000", completed=True,
        checkpoint={"version": 1, "receiptId": "r"}, secret_nonce_present=True)
    live_helpers.save_live_state(path, state)
    assert live_helpers.load_live_state(path, "route:a") == state
    assert json.loads(path.read_text())["routeId"] == "route:a"          # veil's on-disk key names


def test_state_files_are_owner_only_and_written_atomically(tmp_path):
    path = tmp_path / "nested" / "state.json"
    live_helpers.save_live_state(path, live_helpers.LiveState(route_id="route:a"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert not list(path.parent.glob("*.tmp"))


def test_state_fails_closed_for_corrupt_malformed_or_wrong_route_files(tmp_path):
    """veil helpers.test.ts:43-52 — a state file we cannot trust never becomes a fresh run."""
    path = tmp_path / "state.json"

    path.write_text("{")
    with pytest.raises(live_helpers.LiveStateError):
        live_helpers.load_live_state(path, "route:a")

    path.write_text(json.dumps([1, 2]))
    with pytest.raises(live_helpers.LiveStateError):
        live_helpers.load_live_state(path, "route:a")

    path.write_text(json.dumps({"routeId": "route:b", "sourceTxId": "x"}))
    with pytest.raises(live_helpers.LiveStateError, match="does not match"):
        live_helpers.load_live_state(path, "route:a")

    path.write_text(json.dumps({"routeId": "route:a", "sourceTxId": 1}))
    with pytest.raises(live_helpers.LiveStateError, match="sourceTxId"):
        live_helpers.load_live_state(path, "route:a")

    path.write_text(json.dumps({"routeId": "route:a", "completed": "yes"}))
    with pytest.raises(live_helpers.LiveStateError, match="completed"):
        live_helpers.load_live_state(path, "route:a")

    path.write_text(json.dumps({"routeId": "route:a", "checkpoint": "not-an-object"}))
    with pytest.raises(live_helpers.LiveStateError, match="checkpoint"):
        live_helpers.load_live_state(path, "route:a")


# ── helpers.py: the secret nonce lives beside the state, never inside it ──────

def test_secret_nonce_is_a_valid_scalar_and_freshly_random():
    from aleo_bridge.encoding import validate_scalar

    first = live_helpers.generate_secret_nonce()
    assert validate_scalar(first) == first
    assert first != live_helpers.generate_secret_nonce()


def test_secret_file_is_exclusive_owner_only_and_absent_from_the_state(tmp_path):
    path = tmp_path / "state.json"
    nonce = live_helpers.ensure_secret_nonce(path)
    secret = live_helpers.secret_path(path)

    assert secret.name.endswith(".secret")
    assert stat.S_IMODE(secret.stat().st_mode) == 0o600
    assert live_helpers.load_secret_nonce(path) == nonce
    assert live_helpers.ensure_secret_nonce(path) == nonce               # stable across invocations

    with pytest.raises(FileExistsError):
        live_helpers.save_secret_nonce(path, nonce)                      # O_CREAT|O_EXCL: never overwritten

    live_helpers.save_live_state(path, live_helpers.LiveState(route_id="route:a", secret_nonce_present=True))
    assert nonce not in path.read_text()
    assert json.loads(path.read_text())["secretNoncePresent"] is True


def test_missing_secret_file_reads_as_none(tmp_path):
    assert live_helpers.load_secret_nonce(tmp_path / "state.json") is None


# ── helpers.py: benchmark marks ───────────────────────────────────────────────

def test_benchmark_reports_per_step_and_total_elapsed_time():
    """veil helpers.test.ts:22-32 (millisecond deltas from an injected clock)."""
    times = iter([1.000, 1.250, 1.900])
    lines: list[str] = []
    benchmark = live_helpers.LiveBenchmark("route", now=lambda: next(times), log=lines.append)

    benchmark.mark("quote-returned")
    benchmark.mark("execute-returned")

    assert lines == ["[route] quote-returned: +250ms (total 250ms)",
                     "[route] execute-returned: +650ms (total 900ms)"]
    assert [m.step for m in benchmark.marks] == ["quote-returned", "execute-returned"]
    assert [m.elapsed_ms for m in benchmark.marks] == [250, 650]
    assert benchmark.summary() == "route: quote-returned +250ms, execute-returned +650ms (total 900ms)"


def test_benchmark_summary_of_a_run_that_marked_nothing():
    benchmark = live_helpers.LiveBenchmark("route", now=lambda: 0.0, log=lambda _: None)
    assert benchmark.summary() == "route: no marks"


# ── helpers.py: polling ───────────────────────────────────────────────────────

def test_wait_for_returns_the_first_non_none_read_and_sleeps_between_polls():
    reads = iter([None, None, "value"])
    slept: list[float] = []
    clock = iter([0.0, 15.0, 30.0, 45.0])

    value = live_helpers.wait_for(lambda: next(reads), timeout_seconds=600, poll_seconds=15,
                                  sleep=slept.append, now=lambda: next(clock))
    assert value == "value" and slept == [15, 15]


def test_wait_for_raises_at_the_deadline_and_names_the_state_file():
    clock = iter([0.0, 5.0, 10.0, 10.0])
    with pytest.raises(live_helpers.LiveTimeoutError, match="state file"):
        live_helpers.wait_for(lambda: None, timeout_seconds=10, poll_seconds=1,
                              sleep=lambda _: None, now=lambda: next(clock))


def test_wait_for_always_reads_at_least_once_even_with_a_zero_timeout():
    calls = []

    def read():
        calls.append(1)
        return "immediate"

    assert live_helpers.wait_for(read, timeout_seconds=0, sleep=lambda _: None) == "immediate"
    assert calls == [1]


# ── helpers.py: the Hyperlane explorer (read-only HTTP) ───────────────────────

class _Response:
    def __init__(self, payload, status_code=200):
        self._payload, self.status_code = payload, status_code

    def json(self):
        return self._payload


DELIVERED = {"data": {"message_view": [{"msg_id": "\\xmessage", "is_delivered": True,
                                        "destination_tx_hash": "\\xdestination"}]}}


def test_hyperlane_lookup_uses_bytea_hashes_and_normalizes_the_result():
    """veil helpers.test.ts:81-103 — the explorer speaks PostgreSQL bytea, the SDK speaks 0x."""
    seen = {}

    def post(url, payload, timeout):
        seen.update(url=url, payload=payload)
        return _Response(DELIVERED)

    delivery = live_helpers.hyperlane_delivery("0xsource", post=post)
    assert delivery == live_helpers.HyperlaneDelivery(message_id="0xmessage", destination_tx_id="0xdestination")
    assert seen["url"] == live_helpers.HYPERLANE_EXPLORER_URL
    assert "$hash: bytea!" in seen["payload"]["query"]
    assert seen["payload"]["variables"]["hash"] == "\\xsource"


def test_hyperlane_lookup_decodes_a_solana_base58_signature():
    """veil helpers.test.ts:105-126."""
    seen = {}

    def post(url, payload, timeout):
        seen.update(payload=payload)
        return _Response(DELIVERED)

    live_helpers.hyperlane_delivery(
        "QrRfJM8xSiKgvqgd8PeiYTgyA7EkLbzKSnEn5wV6amxA4P15cQY41Vh4H85km8RvTX5pDph6oKxhVzsewdGhdnM", post=post)
    assert seen["payload"]["variables"]["hash"] == (
        "\\x1491b6d2018d56b09ce9e368e701ccfc618485ff784f6419fe72d660a4a992d5"
        "f5d0a4392bf75b8172f57faeea28c3e660c0e9544e4320fb9f4df4d9cce9da06")


def test_hyperlane_lookup_is_none_while_the_message_is_undelivered():
    undelivered = {"data": {"message_view": [{"msg_id": "\\xmessage", "is_delivered": False}]}}
    assert live_helpers.hyperlane_delivery("0xsource", post=lambda *a, **k: _Response(undelivered)) is None
    assert live_helpers.hyperlane_delivery("0xsource", post=lambda *a, **k: _Response({"data": {"message_view": []}})) is None


@pytest.mark.parametrize("status", [429, 500, 502, 503])
def test_hyperlane_lookup_returns_none_when_the_explorer_rate_limits_or_fails(status):
    """A throttled explorer is an environment condition, never a verdict on the transfer."""
    assert live_helpers.hyperlane_delivery("0xsource", post=lambda *a, **k: _Response({}, status)) is None


def test_hyperlane_lookup_surfaces_graphql_errors():
    """veil helpers.test.ts:128-136 — a bad query must not poll silently until the timeout."""
    errors = {"errors": [{"message": "invalid bytea input"}]}
    with pytest.raises(live_helpers.ExplorerError, match="invalid bytea input"):
        live_helpers.hyperlane_delivery("0xsource", post=lambda *a, **k: _Response(errors))


def test_wait_for_hyperlane_delivery_gives_up_quietly_rather_than_failing_a_done_leg():
    """The leg is already `done`; the destination-tx lookup is a convenience, not an assertion."""
    assert live_helpers.wait_for_hyperlane_delivery(
        "0xsource", post=lambda *a, **k: _Response({}, 429),
        timeout_seconds=0, poll_seconds=0, sleep=lambda _: None) is None

    delivery = live_helpers.wait_for_hyperlane_delivery(
        "0xsource", post=lambda *a, **k: _Response(DELIVERED),
        timeout_seconds=60, poll_seconds=0, sleep=lambda _: None)
    assert delivery.destination_tx_id == "0xdestination"


# ── helpers.py: Aleo confirmation ─────────────────────────────────────────────

class _StatusBridge:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.seen: list[str] = []

    def status_of(self, tx_id):
        self.seen.append(tx_id)
        return next(self.statuses)


def test_wait_for_aleo_transaction_accepts_after_pending(monkeypatch):
    bridge = _StatusBridge([("pending", None), ("accepted", None)])
    monkeypatch.setattr(live_helpers, "aleo_transaction_status", lambda b, tx: b.status_of(tx))
    live_helpers.wait_for_aleo_transaction(bridge, "at1x", timeout_seconds=60, poll_seconds=0,
                                           sleep=lambda _: None)
    assert bridge.seen == ["at1x", "at1x"]


def test_wait_for_aleo_transaction_raises_on_a_rejected_transaction(monkeypatch):
    """veil helpers.test.ts:139-150 — a rejected transaction is a failure, not a slow confirmation."""
    bridge = _StatusBridge([("rejected", "Aleo transaction at1rejected was rejected by the network")])
    monkeypatch.setattr(live_helpers, "aleo_transaction_status", lambda b, tx: b.status_of(tx))
    with pytest.raises(live_helpers.LiveCaseError, match="rejected"):
        live_helpers.wait_for_aleo_transaction(bridge, "at1rejected", timeout_seconds=60, poll_seconds=0,
                                               sleep=lambda _: None)


def test_underfunded_carries_the_shortfall():
    error = live_helpers.Underfunded(asset_id="ethereum/usdc", needed=2_000_000, have=1_500_000)
    assert error.shortfall == 500_000
    assert "ethereum/usdc" in str(error) and "500000" in str(error)
