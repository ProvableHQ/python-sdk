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


# ══ cases.py ══════════════════════════════════════════════════════════════════

from pathlib import Path                                                 # noqa: E402

from aleo_bridge import lifecycle                                        # noqa: E402
from aleo_bridge.registry import DEFAULT_REGISTRY                        # noqa: E402
from aleo_bridge.types import Progress, Receipt, Status                  # noqa: E402
from tests.fakes.fake_bridge import ALEO_RECIPIENT, EVM_ADDRESS, FakeBridge   # noqa: E402
from tests.live import cases as live_cases                               # noqa: E402

ETH_ROUTE = "hyperlane:ethereum/eth->aleo/eth"
USDC_ROUTE = "xreserve:ethereum/usdc->aleo/usdcx"


class LiveFakeBridge(FakeBridge):
    """``FakeBridge`` plus the public lifecycle verbs — the surface ``cases.py`` is allowed to use.

    The real ``Bridge`` methods are thin wrappers over ``lifecycle``; wiring the same functions onto
    the fake keeps the harness honest (it may only call verbs that exist) without a second fake.
    """

    def quote(self, **kwargs):
        return lifecycle.quote(self, **kwargs)

    def execute(self, plan, **kwargs):
        return lifecycle.execute(self, plan, **kwargs)

    def wait(self, progress, **kwargs):
        return lifecycle.wait(self, progress, **kwargs)

    def recover(self, checkpoint):
        return lifecycle.recover(self, checkpoint)

    def resume(self, progress, **kwargs):
        return lifecycle.resume(self, progress, **kwargs)

    def complete(self, progress, **kwargs):
        return lifecycle.complete(self, progress, **kwargs)

    def pending(self):
        if self.checkpoints is None:
            return []
        return [lifecycle.progress_from_checkpoint(self.registry, cp) for cp in self.checkpoints.list()]


@pytest.fixture
def fake():
    return LiveFakeBridge()


def test_cases_are_veils_five_mainnet_cases_with_their_literals():
    assert live_cases.CASE_NAMES == live_config.CASE_NAMES
    assert live_cases.CASES["evm-xreserve"].amount == "2"                 # veil evm-xreserve:57
    assert live_cases.CASES["evm-xreserve"].mint_mode == "private"        # veil evm-xreserve:60
    assert live_cases.CASES["aleo-xreserve"].amount == "2.000001"         # veil aleo-xreserve:97
    assert live_cases.CASES["aleo-xreserve"].mode == "private"
    assert live_cases.CASES["aleo-hyperlane"].mode == "signer"            # veil aleo-hyperlane:135
    assert live_cases.CASES["evm-hyperlane"].amount is None               # one atomic unit
    assert all(spec.veil_source for spec in live_cases.CASES.values())


def test_every_mainnet_route_is_covered_by_exactly_one_case():
    """'All the routes back and forth': no mainnet route may be left without a case."""
    routes = DEFAULT_REGISTRY.routes(environment="mainnet")
    covered = {route.id: live_cases.case_for_route(DEFAULT_REGISTRY, route) for route in routes}
    assert all(case is not None for case in covered.values()), \
        [rid for rid, case in covered.items() if case is None]

    by_case = {case: {route.id for route in live_cases.routes_for_case(DEFAULT_REGISTRY, case)}
               for case in live_cases.CASE_NAMES}
    assert set().union(*by_case.values()) == set(covered)
    for left in live_cases.CASE_NAMES:
        for right in live_cases.CASE_NAMES:
            if left != right:
                assert not by_case[left] & by_case[right]

    active = {route.id for route in routes if route.active}
    assert ETH_ROUTE in by_case["evm-hyperlane"] and "hyperlane:ethereum/wbtc->aleo/wbtc" in by_case["evm-hyperlane"]
    assert "hyperlane:aleo/sol->solana/sol" in by_case["aleo-hyperlane"]
    assert by_case["solana-hyperlane"] & active == {"hyperlane:solana/sol->aleo/sol"}
    assert by_case["evm-xreserve"] & active == {USDC_ROUTE}
    assert by_case["aleo-xreserve"] & active == {"xreserve:aleo/usdcx->ethereum/usdc"}


def test_default_amount_is_one_atomic_unit_or_veils_literal():
    route = DEFAULT_REGISTRY.route(ETH_ROUTE)
    assert live_cases.default_amount(DEFAULT_REGISTRY, "evm-hyperlane", route) == "0.000000000000000001"
    usdc = DEFAULT_REGISTRY.route(USDC_ROUTE)
    assert live_cases.default_amount(DEFAULT_REGISTRY, "evm-xreserve", usdc) == "2"


def test_state_names_are_route_qualified():
    assert live_cases.state_name("evm-hyperlane", ETH_ROUTE) == "evm-hyperlane-hyperlane-ethereum-eth-aleo-eth"
    assert live_cases.state_name("evm-hyperlane", "hyperlane:ethereum/wbtc->aleo/wbtc") \
        != live_cases.state_name("evm-hyperlane", ETH_ROUTE)


def test_quote_only_prints_the_table_and_submits_nothing(fake, tmp_path):
    lines: list[str] = []
    state_path = tmp_path / "evm-hyperlane.json"
    state = live_cases.run_case(fake, "evm-hyperlane", ETH_ROUTE, state_path=state_path,
                                execute=False, log=lines.append)

    assert state.completed is False and state.checkpoint is None
    assert not state_path.exists()                      # a rehearsal leaves no state behind
    assert not any(event[0] in {"evm_send", "submit", "prove"} for event in fake.events)
    printed = "\n".join(lines)
    assert ETH_ROUTE in printed and "evm-hyperlane" in printed and "quote only" in printed


def test_the_case_records_veils_benchmark_marks(fake, tmp_path):
    benchmark = live_helpers.LiveBenchmark("evm-hyperlane", log=lambda _: None)
    live_cases.run_case(fake, "evm-hyperlane", ETH_ROUTE, state_path=tmp_path / "s.json",
                        execute=False, benchmark=benchmark, log=lambda _: None)
    assert [mark.step for mark in benchmark.marks] == ["plan-prepared", "quote-returned"]


def test_quote_only_never_creates_the_private_mint_secret(fake, tmp_path):
    state_path = tmp_path / "evm-xreserve.json"
    live_cases.run_case(fake, "evm-xreserve", USDC_ROUTE, state_path=state_path, execute=False,
                        log=lambda _: None)
    assert not live_helpers.secret_path(state_path).exists()


def test_underfunded_names_the_asset_and_the_shortfall(fake, tmp_path):
    fake.eth.balances = {"ethereum/eth": 5}
    with pytest.raises(live_helpers.Underfunded) as excinfo:
        live_cases.run_case(fake, "evm-hyperlane", ETH_ROUTE, state_path=tmp_path / "s.json",
                            execute=False, log=lambda _: None)
    assert excinfo.value.asset_id == "ethereum/eth" and excinfo.value.shortfall > 0


def test_a_completed_case_is_a_no_op_that_re_asserts_its_record(fake, tmp_path):
    state_path = tmp_path / "done.json"
    live_helpers.save_live_state(state_path, live_helpers.LiveState(
        route_id=ETH_ROUTE, source_tx_id="0xsource", message_id="0xmessage",
        destination_tx_id="at1destination", completed=True))

    state = live_cases.run_case(fake, "evm-hyperlane", ETH_ROUTE, state_path=state_path,
                                execute=True, log=lambda _: None)
    assert state.completed and state.source_tx_id == "0xsource"
    assert fake.calls == [] and fake.events == []       # nothing was quoted, nothing was submitted


def test_a_completed_state_without_a_source_transaction_fails_closed(fake, tmp_path):
    state_path = tmp_path / "bad.json"
    live_helpers.save_live_state(state_path, live_helpers.LiveState(route_id=ETH_ROUTE, completed=True))
    with pytest.raises(live_helpers.LiveCaseError):
        live_cases.run_case(fake, "evm-hyperlane", ETH_ROUTE, state_path=state_path, execute=True,
                            log=lambda _: None)


def test_an_inactive_route_is_refused_before_anything_is_read(fake, tmp_path):
    with pytest.raises(live_helpers.LiveCaseError, match="metadata-required"):
        live_cases.run_case(fake, "evm-hyperlane", "hyperlane:ethereum/usad->aleo/usad",
                            state_path=tmp_path / "s.json", execute=False, log=lambda _: None)


def test_recipient_and_sender_default_to_our_own_addresses(fake):
    route = DEFAULT_REGISTRY.route(ETH_ROUTE)
    assert live_cases.default_recipient(fake, route) == ALEO_RECIPIENT
    assert live_cases.sender_for(fake, route) == EVM_ADDRESS

    outbound = DEFAULT_REGISTRY.route("hyperlane:aleo/eth->ethereum/eth")
    assert live_cases.default_recipient(fake, outbound) == EVM_ADDRESS
    assert live_cases.sender_for(fake, outbound) == ALEO_RECIPIENT


def test_an_unconfigured_destination_chain_is_reported_not_an_attribute_error():
    """Ruling: probe ``bridge.solana``/``bridge.ethereum``; ``bridge.sol``/``bridge.eth`` RAISE."""
    aleo_only = LiveFakeBridge(ethereum=False, solana=False)
    assert aleo_only.ethereum is None and aleo_only.solana is None
    route = DEFAULT_REGISTRY.route("hyperlane:aleo/sol->solana/sol")
    with pytest.raises(live_helpers.LiveCaseError, match="solana"):
        live_cases.default_recipient(aleo_only, route)
    aleo_only.public_balances = {"aleo/sol": 7}
    assert live_cases.read_balances(aleo_only) == {"aleo/sol": 7}        # the Aleo row still reads


def _xreserve_quote(**extra):
    from aleo_bridge.types import EvmXReserveQuote

    plan = lifecycle.prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx",
                             amount="2", recipient=ALEO_RECIPIENT, mint_mode="private")
    return EvmXReserveQuote(kind="evm-xreserve", plan=plan, fees=(), amount_out="2", hook_data=b"",
                            remote_recipient_bytes32=b"\x00" * 32, balance_atomic=5_000_000,
                            allowance_atomic=0, approval_required=True, **extra)


def test_print_quote_renders_the_xreserve_max_fee_as_the_protocol_fee_line(fake):
    """``EvmXReserveQuote.fees`` is empty: its max fee is the protocol cost a human has to see."""
    lines: list[str] = []
    live_cases.print_quote(fake, _xreserve_quote(max_fee_atomic=100_000),
                           case="evm-xreserve", route_id=USDC_ROUTE, log=lines.append)
    printed = "\n".join(lines)
    assert "0.1 USDC [xReserve max fee]" in printed
    assert "max_fee_atomic" in printed and ALEO_RECIPIENT in printed
    assert "scalar" not in printed


# ── the drive loop: wait → resume → complete → done, and never execute twice ──

class _ScriptedBridge:
    """A bridge whose lifecycle verbs return a scripted sequence, to test ``_drive`` in isolation."""

    def __init__(self, steps):
        self.steps, self.calls = list(steps), []
        self.plan = lifecycle.prepare(DEFAULT_REGISTRY, source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx",
                                      amount="2", recipient=ALEO_RECIPIENT, mint_mode="private")

    def _next(self, verb):
        self.calls.append(verb)
        state = self.steps.pop(0)
        receipt = Receipt(id="r1", protocol="xreserve", status=Status.SOURCE_CONFIRMING,
                          source_tx_id="0xsource",
                          protocol_state={"routeId": USDC_ROUTE, "messageId": "0xmessage"})
        return Progress(state, self.plan, receipt, error="scripted failure" if state == "failed" else None)

    def wait(self, progress, **kwargs):
        return self._next("wait")

    def resume(self, progress, **kwargs):
        return self._next("resume")

    def complete(self, progress, **kwargs):
        return self._next("complete")

    def execute(self, *args, **kwargs):
        raise AssertionError("execute must never be called from the drive loop")


def _drive(bridge, first, **kwargs):
    benchmark = live_helpers.LiveBenchmark("t", log=lambda _: None)
    progress = Progress(first, bridge.plan,
                        Receipt(id="r1", protocol="xreserve", status=Status.SOURCE_CONFIRMING,
                                protocol_state={"routeId": USDC_ROUTE}))
    return live_cases._drive(bridge, progress, live_helpers.LiveState(route_id=USDC_ROUTE),
                             Path("/nonexistent/state.json"), spec=live_cases.CASES["evm-xreserve"],
                             benchmark=benchmark, save=lambda _: None, wait_timeout_seconds=1,
                             wait_poll_seconds=0, log=lambda _: None, **kwargs)


def test_drive_runs_wait_resume_complete_until_done():
    bridge = _ScriptedBridge(["resume", "wait", "complete", "wait", "done"])
    progress = _drive(bridge, "wait", secret_nonce="7scalar")
    assert progress.next == "done"
    assert bridge.calls == ["wait", "resume", "wait", "complete", "wait"]


def test_drive_raises_the_reported_error_on_failure():
    bridge = _ScriptedBridge(["failed"])
    with pytest.raises(live_helpers.LiveCaseError, match="scripted failure"):
        _drive(bridge, "wait", secret_nonce="7scalar")


def test_drive_refuses_a_private_mint_without_the_kept_nonce():
    bridge = _ScriptedBridge(["complete"])
    with pytest.raises(live_helpers.LiveCaseError, match="secret nonce"):
        _drive(bridge, "wait", secret_nonce=None)


def test_drive_gives_up_rather_than_looping_forever():
    bridge = _ScriptedBridge(["wait"] * 40)
    with pytest.raises(live_helpers.LiveCaseError, match="did not settle"):
        _drive(bridge, "wait", secret_nonce="7scalar")


# ══ scripts/rehearse.py ═══════════════════════════════════════════════════════

import importlib.util                                                    # noqa: E402

from aleo_bridge.errors import PollingTimeoutError                       # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rehearse.py"


@pytest.fixture(scope="module")
def rehearse():
    spec = importlib.util.spec_from_file_location("rehearse", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def state_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(STATE_DIR, str(tmp_path))
    return tmp_path


def _report(path):
    return json.loads(Path(path).read_text())


def test_cli_parses_the_documented_flags(rehearse):
    args = rehearse.parse_args(["--case", "evm-hyperlane"])
    assert args.case == "evm-hyperlane" and args.route is None
    assert args.quote_only is False and args.recover is None and args.report is None

    args = rehearse.parse_args(["--case", "aleo-xreserve", "--route", "xreserve:aleo/usdcx->ethereum/usdc",
                                "--quote-only", "--report", "/tmp/r.json"])
    assert args.route == "xreserve:aleo/usdcx->ethereum/usdc" and args.quote_only and args.report == "/tmp/r.json"

    args = rehearse.parse_args(["--recover", "/tmp/state.json"])
    assert args.recover == "/tmp/state.json" and args.case is None

    for bad in ([], ["--case", "not-a-case"]):
        with pytest.raises(SystemExit):
            rehearse.parse_args(bad)


def test_cli_has_no_reset_flag(rehearse):
    """veil's solana-deposit.ts --reset deletes a checkpoint without checking the chain; not ported."""
    with pytest.raises(SystemExit):
        rehearse.parse_args(["--case", "evm-hyperlane", "--reset"])


def test_quote_only_run_reports_every_route_of_the_case(rehearse, fake, state_dir, tmp_path):
    lines: list[str] = []
    report = tmp_path / "report.json"
    code = rehearse.run(["--case", "evm-hyperlane", "--quote-only", "--report", str(report)],
                        bridge_factory=lambda: fake, log=lines.append)

    assert code == rehearse.EXIT_OK
    payload = _report(report)
    rows = {row["route_id"]: row for row in payload["results"]}
    assert set(rows) == {route.id for route in live_cases.routes_for_case(DEFAULT_REGISTRY, "evm-hyperlane")}
    assert rows[ETH_ROUTE]["status"] == "quote-only"
    assert rows["hyperlane:ethereum/usad->aleo/usad"]["status"] == "skipped"
    assert "metadata-required" in rows["hyperlane:ethereum/usad->aleo/usad"]["reason"]
    assert payload["execute"] is False and payload["case"] == "evm-hyperlane"

    printed = "\n".join(lines)
    assert ETH_ROUTE in printed and "quote-only" in printed
    assert not any(event[0] in {"evm_send", "submit"} for event in fake.events)


def test_a_single_route_can_be_selected(rehearse, fake, state_dir, tmp_path):
    report = tmp_path / "report.json"
    rehearse.run(["--case", "evm-hyperlane", "--route", ETH_ROUTE, "--quote-only", "--report", str(report)],
                 bridge_factory=lambda: fake, log=lambda _: None)
    assert [row["route_id"] for row in _report(report)["results"]] == [ETH_ROUTE]


def test_without_the_acknowledgements_the_run_quotes_and_names_the_variable(rehearse, fake, state_dir,
                                                                            monkeypatch, tmp_path):
    """The gate is READ here; nothing prints a line that would set it in a subshell."""
    monkeypatch.setenv(FUNDS, "1")
    monkeypatch.setenv(ACK, "I_ACKNOWLEDGE_BRIDGE_MAINNET_FUNDS")
    monkeypatch.setenv(CASES, "evm-hyperlane")          # the case is acknowledged, submission is not
    lines: list[str] = []
    report = tmp_path / "report.json"
    code = rehearse.run(["--case", "evm-hyperlane", "--route", ETH_ROUTE, "--report", str(report)],
                        bridge_factory=lambda: fake, log=lines.append)

    payload = _report(report)
    assert code == rehearse.EXIT_OK and payload["execute"] is False
    assert "BRIDGE_LIVE_MAINNET_EXECUTE" in payload["reason"]
    printed = "\n".join(lines)
    assert "export " not in printed and "I_ACKNOWLEDGE" not in printed
    assert "BRIDGE_LIVE_MAINNET_EXECUTE" in printed


def test_acknowledged_runs_ask_the_case_to_execute(rehearse, fake, state_dir, monkeypatch, tmp_path):
    monkeypatch.setenv(FUNDS, "1")
    monkeypatch.setenv(ACK, "I_ACKNOWLEDGE_BRIDGE_MAINNET_FUNDS")
    monkeypatch.setenv(CASES, "evm-hyperlane")
    monkeypatch.setenv(EXECUTE, "I_ACKNOWLEDGE_THIS_SUBMITS_MAINNET_TRANSACTIONS")
    seen = {}

    def runner(bridge, route_id, **kwargs):
        seen.update(route_id=route_id, execute=kwargs["execute"], state_path=kwargs["state_path"])
        return live_helpers.LiveState(route_id=route_id, source_tx_id="0xsource", message_id="0xm",
                                      destination_tx_id="at1d", completed=True)

    monkeypatch.setitem(live_cases.RUNNERS, "evm-hyperlane", runner)
    report = tmp_path / "report.json"
    code = rehearse.run(["--case", "evm-hyperlane", "--route", ETH_ROUTE, "--report", str(report)],
                        bridge_factory=lambda: fake, log=lambda _: None)

    assert code == rehearse.EXIT_OK and seen["execute"] is True
    assert Path(seen["state_path"]) == state_dir / "mainnet" / f"{live_cases.state_name('evm-hyperlane', ETH_ROUTE)}.json"
    row = _report(report)["results"][0]
    assert row["status"] == "completed" and row["source_tx_id"] == "0xsource" and row["message_id"] == "0xm"


def test_an_underfunded_case_is_skipped_with_its_shortfall(rehearse, fake, state_dir, monkeypatch, tmp_path):
    def runner(bridge, route_id, **kwargs):
        raise live_helpers.Underfunded(asset_id="ethereum/eth", needed=1000, have=1)

    monkeypatch.setitem(live_cases.RUNNERS, "evm-hyperlane", runner)
    report = tmp_path / "report.json"
    code = rehearse.run(["--case", "evm-hyperlane", "--route", ETH_ROUTE, "--quote-only", "--report", str(report)],
                        bridge_factory=lambda: fake, log=lambda _: None)
    row = _report(report)["results"][0]
    assert code == rehearse.EXIT_OK and row["status"] == "skipped" and "999" in row["reason"]


def test_a_timeout_is_pending_with_the_resume_command_not_a_failure(rehearse, fake, state_dir, monkeypatch, tmp_path):
    def runner(bridge, route_id, **kwargs):
        raise PollingTimeoutError("still in flight", status=Status.DELIVERY_PENDING)

    monkeypatch.setitem(live_cases.RUNNERS, "evm-hyperlane", runner)
    report = tmp_path / "report.json"
    lines: list[str] = []
    code = rehearse.run(["--case", "evm-hyperlane", "--route", ETH_ROUTE, "--quote-only", "--report", str(report)],
                        bridge_factory=lambda: fake, log=lines.append)
    row = _report(report)["results"][0]
    assert code == rehearse.EXIT_PENDING and row["status"] == "pending"
    assert "--recover" in row["resume"] and row["resume"] in "\n".join(lines)


def test_a_failed_case_exits_one(rehearse, fake, state_dir, monkeypatch, tmp_path):
    def runner(bridge, route_id, **kwargs):
        raise live_helpers.LiveCaseError("the destination rejected it")

    monkeypatch.setitem(live_cases.RUNNERS, "evm-hyperlane", runner)
    report = tmp_path / "report.json"
    code = rehearse.run(["--case", "evm-hyperlane", "--route", ETH_ROUTE, "--quote-only", "--report", str(report)],
                        bridge_factory=lambda: fake, log=lambda _: None)
    row = _report(report)["results"][0]
    assert code == rehearse.EXIT_FAILED and row["status"] == "failed" and "rejected" in row["reason"]


def test_recover_resolves_the_case_from_the_state_file(rehearse, fake, state_dir, monkeypatch, tmp_path):
    state_path = state_dir / "mainnet" / "evm-hyperlane-resume.json"
    live_helpers.save_live_state(state_path, live_helpers.LiveState(route_id=ETH_ROUTE, source_tx_id="0xs"))
    seen = {}

    def runner(bridge, route_id, **kwargs):
        seen.update(route_id=route_id, state_path=kwargs["state_path"])
        return live_helpers.LiveState(route_id=route_id, source_tx_id="0xs", completed=True)

    monkeypatch.setitem(live_cases.RUNNERS, "evm-hyperlane", runner)
    code = rehearse.run(["--recover", str(state_path), "--quote-only"],
                        bridge_factory=lambda: fake, log=lambda _: None)
    assert code == rehearse.EXIT_OK
    assert seen["route_id"] == ETH_ROUTE and Path(seen["state_path"]) == state_path


def test_testnet_submission_is_gated_only_on_the_funds_variables(rehearse, monkeypatch, tmp_path):
    """I6: the mainnet acknowledgements gate MAINNET funds. Requiring them on testnet too made
    ``--recover`` unable to finish a testnet transfer at all — the checkpoint was already on disk
    and the funds already committed, and the run would only ever re-quote."""
    monkeypatch.setenv(FUNDS, "1")
    monkeypatch.setenv(STATE_DIR, str(tmp_path))

    allowed, reason = rehearse.execution_allowed("aleo-xreserve", quote_only=False, environment="testnet")
    assert allowed and "testnet" in reason
    allowed, reason = rehearse.execution_allowed("aleo-xreserve", quote_only=False, environment="mainnet")
    assert not allowed and ACK in reason and CASES in reason
    # --quote-only and the funds gate still win in both environments
    for environment in ("testnet", "mainnet"):
        assert rehearse.execution_allowed("aleo-xreserve", quote_only=True, environment=environment)[0] is False
    monkeypatch.delenv(FUNDS)
    allowed, reason = rehearse.execution_allowed("aleo-xreserve", quote_only=False, environment="testnet")
    assert not allowed and FUNDS in reason


def test_recover_builds_its_client_for_the_environment_the_state_file_names(rehearse, fake, monkeypatch,
                                                                            tmp_path):
    """I6: ``--recover`` used to build a bare ``Bridge.from_env()``, i.e. always the mainnet key —
    so a testnet state file was resumed with the wrong account. It now builds the client the way
    the live suite does, per environment, from ``tests/live/config.py``."""
    monkeypatch.setenv(STATE_DIR, str(tmp_path))
    testnet_route = "xreserve:aleo-testnet/usdcx->sepolia/usdc"
    state_path = tmp_path / "testnet" / "aleo-xreserve-resume.json"
    live_helpers.save_live_state(state_path, live_helpers.LiveState(route_id=testnet_route, source_tx_id="at1s"))
    built = []

    def fake_build(environment):
        built.append(environment)
        return fake

    def runner(bridge, route_id, **kwargs):
        return live_helpers.LiveState(route_id=route_id, source_tx_id="at1s", completed=True)

    monkeypatch.setattr(live_helpers, "build_bridge", fake_build)
    monkeypatch.setitem(live_cases.RUNNERS, "aleo-xreserve", runner)
    assert rehearse.run(["--recover", str(state_path), "--quote-only"], log=lambda _: None) == rehearse.EXIT_OK
    assert built == ["testnet"]                    # never Bridge.from_env(), never the mainnet key


def test_the_table_renders_one_line_per_route(rehearse):
    rows = [{"case": "evm-hyperlane", "route_id": ETH_ROUTE, "status": "quote-only", "reason": "",
             "source_tx_id": None, "message_id": None, "destination_tx_id": None, "resume": ""},
            {"case": "evm-hyperlane", "route_id": "hyperlane:ethereum/wbtc->aleo/wbtc", "status": "skipped",
             "reason": "registry availability: metadata-required", "source_tx_id": None,
             "message_id": None, "destination_tx_id": None, "resume": ""}]
    table = rehearse.render_table(rows)
    assert ETH_ROUTE in table and "quote-only" in table and "metadata-required" in table
    assert len(table.strip().splitlines()) >= 3        # header + two rows


# ══ 13b: environment aliases, the execute handover, the suite's parametrization ══

def test_key_and_rpc_variables_resolve_per_environment(monkeypatch):
    """A testnet run may never reach for the mainnet Aleo key, and an unset RPC has a default."""
    monkeypatch.setenv("BRIDGE_PRIVATE_KEY", "APrivateKey1zkpMainnet")
    monkeypatch.setenv("ALEO_E2E_PRIVATE_KEY", "APrivateKey1zkpTestnet")
    assert live_config.aleo_private_key("mainnet") == "APrivateKey1zkpMainnet"
    assert live_config.aleo_private_key("testnet") == "APrivateKey1zkpTestnet"

    monkeypatch.setenv("BRIDGE_LIVE_ALEO_TESTNET_PRIVATE_KEY", "APrivateKey1zkpAlias")
    assert live_config.aleo_private_key("testnet") == "APrivateKey1zkpAlias"      # the alias wins
    assert live_config.aleo_private_key("mainnet") == "APrivateKey1zkpMainnet"

    monkeypatch.delenv("BRIDGE_PRIVATE_KEY")
    with pytest.raises(live_config.LiveConfigError, match="BRIDGE_PRIVATE_KEY"):
        live_config.aleo_private_key("mainnet")
    with pytest.raises(live_config.LiveConfigError, match="environment"):
        live_config.aleo_private_key("devnet")


def test_evm_key_is_normalised_and_the_value_never_appears(monkeypatch):
    monkeypatch.setenv("BRIDGE_EVM_PRIVATE_KEY", "0x" + "AB" * 32)
    assert live_config.evm_private_key("mainnet") == "0x" + "ab" * 32
    assert live_config.evm_private_key("testnet") == "0x" + "ab" * 32

    monkeypatch.setenv("BRIDGE_LIVE_EVM_TESTNET_PRIVATE_KEY", "0x" + "cd" * 32)
    assert live_config.evm_private_key("testnet") == "0x" + "cd" * 32
    assert live_config.evm_private_key("mainnet") == "0x" + "ab" * 32

    monkeypatch.setenv("BRIDGE_LIVE_EVM_TESTNET_PRIVATE_KEY", "not-a-key")
    with pytest.raises(live_config.LiveConfigError) as excinfo:
        live_config.evm_private_key("testnet")
    assert "not-a-key" not in str(excinfo.value)


def test_rpc_urls_fall_back_to_the_public_defaults(monkeypatch):
    for name in ("SEPOLIA_RPC_URL", "BRIDGE_LIVE_SEPOLIA_RPC_URL", "ETHEREUM_RPC_URL",
                 "BRIDGE_LIVE_ETHEREUM_RPC_URL", "ALEO_ENDPOINT", "BRIDGE_LIVE_ALEO_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    assert live_config.evm_rpc_url("mainnet") == live_config.DEFAULT_ETHEREUM_RPC_URL
    assert live_config.evm_rpc_url("testnet") == live_config.DEFAULT_SEPOLIA_RPC_URL
    assert live_config.aleo_endpoint() == live_config.DEFAULT_ALEO_ENDPOINT

    monkeypatch.setenv("SEPOLIA_RPC_URL", "https://sepolia.example")
    monkeypatch.setenv("BRIDGE_LIVE_ETHEREUM_RPC_URL", "https://eth.example")
    monkeypatch.setenv("ALEO_ENDPOINT", "https://aleo.example/api")
    assert live_config.evm_rpc_url("testnet") == "https://sepolia.example"
    assert live_config.evm_rpc_url("mainnet") == "https://eth.example"
    assert live_config.aleo_endpoint() == "https://aleo.example/api"
    assert live_config.first_value(("ABSENT_A", "SEPOLIA_RPC_URL")) == ("SEPOLIA_RPC_URL", "https://sepolia.example")
    assert live_config.first_value(("ABSENT_A", "ABSENT_B")) is None


def test_amount_overrides_are_per_case_with_an_xreserve_shorthand(monkeypatch):
    monkeypatch.delenv("BRIDGE_LIVE_XRESERVE_AMOUNT", raising=False)
    assert live_config.case_amount_override("evm-xreserve") is None
    monkeypatch.setenv("BRIDGE_LIVE_XRESERVE_AMOUNT", "3")
    assert live_config.case_amount_override("evm-xreserve") == "3"
    assert live_config.case_amount_override("aleo-xreserve") == "3"
    assert live_config.case_amount_override("evm-hyperlane") is None       # never a Hyperlane amount
    monkeypatch.setenv("BRIDGE_LIVE_EVM_XRESERVE_AMOUNT", "5")
    assert live_config.case_amount_override("evm-xreserve") == "5"


def _refuse_execute(*args, **kwargs):
    raise AssertionError("execute must never be called for a transfer that already has a checkpoint")


def test_stop_after_execute_hands_the_transfer_over_through_the_state_file(fake, tmp_path):
    """Phase one executes and returns; phase two must reach done without executing again."""
    state_path = tmp_path / "handover.json"
    first = live_cases.run_case(fake, "evm-hyperlane", ETH_ROUTE, state_path=state_path, execute=True,
                                stop_after_execute=True, wait_timeout_seconds=1, wait_poll_seconds=0,
                                log=lambda _: None)
    assert first.checkpoint is not None and first.completed is False and first.source_tx_id
    assert state_path.exists()

    submitted = [event for event in fake.events if event[0] in {"evm_send", "submit"}]
    fake.execute = _refuse_execute                       # the handover may only use the recovery verbs
    fake.eth.recover_result = Receipt(
        id=first.checkpoint["receiptId"], protocol="hyperlane", status=Status.COMPLETED,
        source_tx_id=first.source_tx_id, destination_tx_id="at1delivered",
        protocol_state={"routeId": ETH_ROUTE, "messageId": "0x" + "ee" * 32})
    second = live_cases.run_case(fake, "evm-hyperlane", ETH_ROUTE, state_path=state_path, execute=True,
                                 wait_timeout_seconds=1, wait_poll_seconds=0, log=lambda _: None)
    assert second.completed and second.source_tx_id == first.source_tx_id
    assert [event for event in fake.events if event[0] in {"evm_send", "submit"}] == submitted


def test_the_suite_parametrizes_every_route_in_both_environments():
    from tests.live import test_lifecycle_live as suite

    for environment, buckets in (("mainnet", suite.MAINNET_ROUTES), ("testnet", suite.TESTNET_ROUTES)):
        listed = {route.id for routes in buckets.values() for route in routes}
        assert listed == {route.id for route in DEFAULT_REGISTRY.routes(environment=environment)}

    mainnet = {route.id: route for routes in suite.MAINNET_ROUTES.values() for route in routes}
    assert "hyperlane:ethereum/usad->aleo/usad" in mainnet                 # metadata-required, parametrized
    assert not mainnet["hyperlane:ethereum/usad->aleo/usad"].active
    assert suite.TESTNET_DEPOSIT_ROUTE in {r.id for r in suite.TESTNET_ROUTES["evm-xreserve"]}
    assert suite.TESTNET_RETURN_ROUTE in {r.id for r in suite.TESTNET_ROUTES["aleo-xreserve"]}
    assert suite.TESTNET_DEPOSIT_AMOUNT == "3" and suite.TESTNET_RETURN_AMOUNT == "2.000001"
    assert "--recover" in suite.resume_command("/tmp/state.json")


def test_only_the_aleo_to_evm_withdrawal_measures_delivery_by_balance():
    """The one leg with no delivery query anywhere: `wait` on it could only ever time out."""
    rise = {route.id for route in DEFAULT_REGISTRY.routes()
            if live_cases.delivery_is_a_balance_rise(route, DEFAULT_REGISTRY)}
    assert rise == {"xreserve:aleo/usdcx->ethereum/usdc", "xreserve:aleo-testnet/usdcx->sepolia/usdc"}
    assert not live_cases.delivery_is_a_balance_rise(DEFAULT_REGISTRY.route(USDC_ROUTE), DEFAULT_REGISTRY)
    assert not live_cases.delivery_is_a_balance_rise(DEFAULT_REGISTRY.route(ETH_ROUTE), DEFAULT_REGISTRY)


def test_the_suite_sets_no_acknowledgement_and_prints_no_settable_form():
    source = (SCRIPT.parent.parent / "tests" / "live" / "test_lifecycle_live.py").read_text(encoding="utf-8")
    assert "export " not in source
    assert live_config.MAINNET_ACK not in source and live_config.MAINNET_EXECUTE_ACK not in source
    assert "setenv" not in source and "os.environ[" not in source


def test_hyperlane_lookup_is_skipped_for_aleo_origin_ids():
    """An Aleo ``at1…`` source id has no explorer bytea form (bech32, not base58): the lookup
    returns None immediately, never posts, and never polls — the SDK's own delivery check
    (balance rise on the destination) is the verdict for Aleo-origin legs, as in veil."""
    def never_post(*_args, **_kwargs):
        raise AssertionError("the explorer must not be queried for an Aleo source id")

    aleo_id = "at1fhunxkgp2zgqmu4qv8848qmc9c3ytcgzmzm758860nyzlad2s58q0jr0ge"
    assert live_helpers._bytea(aleo_id) is None
    assert live_helpers.hyperlane_delivery(aleo_id, post=never_post) is None
    slept: list[float] = []
    assert live_helpers.wait_for_hyperlane_delivery(
        aleo_id, post=never_post, sleep=slept.append, now=lambda: 0.0, timeout_seconds=5, poll_seconds=1) is None
    assert slept == []
    assert live_helpers._bytea("not-base58-0OIl") is None          # a bad base58 string degrades the same way
    assert live_helpers._bytea("0x" + "ab" * 32) == "\\x" + "ab" * 32  # EVM hashes are unchanged
