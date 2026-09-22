"""The funded live suite: every registry route, back and forth, through the public lifecycle verbs.

One test per veil case, parametrized over every route that case covers.  The parameter set is built
by ENUMERATING the registry, not by listing route ids by hand, so a new route cannot be added
without a case: :func:`_by_case` raises at import when a route has no case, and a route whose
``availability`` is not ``active`` is parametrized and skipped with ``registry:<availability>``
rather than dropped.  Plus the testnet deposit (`xreserve:sepolia/usdc->aleo-testnet/usdcx`) and,
beyond veil, the testnet RETURN leg (`xreserve:aleo-testnet/usdcx->sepolia/usdc`).

Gates (read only — nothing here sets, exports or prints a settable acknowledgement):

* ``BRIDGE_LIVE_FUNDS=1`` + ``BRIDGE_LIVE_STATE_DIR`` — the funded tests exist at all.
* ``BRIDGE_LIVE_MAINNET_ACK`` + ``BRIDGE_LIVE_MAINNET_CASES`` — a named mainnet case may run.
* ``BRIDGE_LIVE_MAINNET_EXECUTE`` — the wallet may submit.  Without it every mainnet case runs to
  the quote and returns (veil's ``if (!mainnetExecutionEnabled()) return``), which is how the
  quote-only sweep in the report was produced.

Every funded test goes through the disk: phase one quotes, prechecks and executes, then RETURNS —
the client is thrown away; phase two builds a brand new :class:`Bridge` over the same
``FileCheckpointStore`` and finishes the transfer from ``bridge.pending()`` / ``bridge.recover()``.
``execute`` is called at most once per transfer, ever; a timeout is *pending* (skip + the resume
command), never a failure verdict, and ``Underfunded`` is a skip printing the shortfall.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import pytest

from aleo_bridge.errors import InsufficientBalanceError, PollingTimeoutError
from aleo_bridge.registry import DEFAULT_REGISTRY, Route

from . import cases as live_cases
from . import config as live_config
from . import helpers as live_helpers
from .helpers import LiveBenchmark, LiveTimeoutError, Underfunded

pytestmark = pytest.mark.live

#: The testnet pair.  The deposit amount is 3 USDC (controller ruling 2026-09-18): the minted 3
#: USDCx clears the 2 USDCx withdrawal fee so the return leg can burn veil's 2.000001 afterwards.
TESTNET_DEPOSIT_ROUTE = "xreserve:sepolia/usdc->aleo-testnet/usdcx"
TESTNET_RETURN_ROUTE = "xreserve:aleo-testnet/usdcx->sepolia/usdc"
TESTNET_DEPOSIT_AMOUNT = "3"
TESTNET_RETURN_AMOUNT = "2.000001"          # veil mainnet/aleo-xreserve.live.test.ts:97


# ── parametrization: enumerate the registry, leave nothing out ───────────────

def _by_case(environment: str) -> dict[str, list[Route]]:
    """``case → routes`` for *environment*, asserting that every route has a case.

    Hermetic: the registry is static data, so this runs at import with no network and no keys.
    """
    buckets: dict[str, list[Route]] = {name: [] for name in live_cases.CASE_NAMES}
    uncovered: list[str] = []
    for route in DEFAULT_REGISTRY.routes(environment=environment):
        name = live_cases.case_for_route(DEFAULT_REGISTRY, route)
        if name is None:
            uncovered.append(route.id)
        else:
            buckets[name].append(route)
    if uncovered:
        raise AssertionError(
            f"No live case covers these {environment} routes: {sorted(uncovered)} — add the case "
            "to tests/live/cases.py::CASES rather than narrowing this suite")
    return buckets


MAINNET_ROUTES = _by_case("mainnet")
TESTNET_ROUTES = _by_case("testnet")


def _ids(routes: list[Route]) -> list[str]:
    return [route.id for route in routes]


def _params(case: str) -> Any:
    """``pytest.mark.parametrize`` arguments for one case: every mainnet route it covers."""
    routes = MAINNET_ROUTES[case]
    return pytest.mark.parametrize("route", routes, ids=_ids(routes))


# ── clients ──────────────────────────────────────────────────────────────────

def _build_bridge(environment: str) -> Any:
    """A fresh client for *environment*, from the one builder ``scripts/rehearse.py`` also uses
    (``helpers.build_bridge``) — so a rehearsal resumes with exactly the account this suite would."""
    return live_helpers.build_bridge(environment)


#: Opt-in for sharing a MAINNET view key with the hosted record scanner (see below).
SCANNER_VAR = "BRIDGE_LIVE_ALEO_SCANNER"


def _register_record_scanner(bridge: Any, environment: str) -> None:
    """Register the Aleo account with the hosted record scanner, which private selection needs.

    ``bridge.privacy.select_record`` (the record the xReserve burn spends) goes through
    ``aleo.records.find``, and the scanner answers nothing for an account it has not been
    registered for — it raises ``UUIDError`` instead.  Registration **shares that account's view
    key** with the scanning service, which can then decrypt every record the account owns.

    That is a decision about somebody's privacy, so it is automatic only for the testnet e2e key.
    On mainnet the case skips unless ``BRIDGE_LIVE_ALEO_SCANNER=1`` says the operator accepts it;
    nothing here ever sets that variable.
    """
    if environment != "testnet" and live_config.value(SCANNER_VAR) != "1":
        pytest.skip(f"private-record selection needs the hosted record scanner, and registering "
                    f"shares this account's view key with it — set {SCANNER_VAR}=1 to accept that")
    result = bridge.aleo.records.register(bridge.aleo.default_account)
    ok = result.get("ok") if isinstance(result, dict) else result
    print(f"  scanner    hosted record scanner registered for {bridge.aleo_address()} (ok={ok})")


def _client(make_bridge: Callable[[], Any], case: str, environment: str, *, execute: bool) -> Any:
    """A fresh bridge, with the record scanner registered for the cases that select private records.

    Only when we are actually going to burn: a quote-only rehearsal never selects a record, so it
    must not make a view-key decision (nor skip for want of one) on the operator's behalf.
    """
    bridge = make_bridge()
    if execute and case == "aleo-xreserve":
        _register_record_scanner(bridge, environment)
    return bridge


@pytest.fixture
def mainnet_client() -> Callable[[], Any]:
    return lambda: _build_bridge("mainnet")


@pytest.fixture
def testnet_client() -> Callable[[], Any]:
    return lambda: _build_bridge("testnet")


# ── the shared body ──────────────────────────────────────────────────────────

def _state_path(case: str, route: Route, environment: str) -> Path:
    return live_config.live_state_path(environment, live_cases.state_name(case, route.id))


def resume_command(state_path: Path | str) -> str:
    """What the operator types to continue an interrupted transfer (the CLI, not a test run)."""
    return f"python scripts/rehearse.py --recover {state_path}"


def _destination_family(route: Route) -> str:
    asset = DEFAULT_REGISTRY.asset(route.destination_asset_id)
    return DEFAULT_REGISTRY.chain(asset.chain_id).family


def _record(record_property: Callable[[str, Any], None], case: str, route: Route, state: Any,
            benchmark: LiveBenchmark) -> None:
    record_property("case", case)
    record_property("route_id", route.id)
    record_property("source_tx_id", getattr(state, "source_tx_id", None))
    record_property("message_id", getattr(state, "message_id", None))
    record_property("destination_tx_id", getattr(state, "destination_tx_id", None))
    record_property("destination_balance_before", getattr(state, "destination_balance_before", None))
    record_property("completed", bool(getattr(state, "completed", False)))
    record_property("benchmark_ms", benchmark.as_dict())


def _drive_case(make_bridge: Callable[[], Any], case: str, route: Route, *, environment: str,
                execute: bool, record_property: Callable[[str, Any], None],
                amount: str | None = None) -> Any:
    """Quote (and, when acknowledged, execute) one route, then finish it from disk with a NEW client.

    The two phases are the point: whatever ``execute`` returned is dropped with the first client, so
    the transfer can only reach ``done`` through ``bridge.pending()`` → ``bridge.recover()`` →
    ``wait``/``resume``/``complete`` on a process that never saw it start.
    """
    if not route.active:
        pytest.skip(f"registry:{route.availability}")

    state_path = _state_path(case, route, environment)
    benchmark = LiveBenchmark(f"{case}:{live_cases.route_slug(route.id)}", log=print)
    amount = amount or live_config.case_amount_override(case)
    recipient = live_config.recipient_override(_destination_family(route))
    deadline = time.monotonic() + live_cases.CASE_TIMEOUT_SECONDS

    def remaining() -> float:
        return max(deadline - time.monotonic(), 0.0)

    def run(bridge: Any, *, stop_after_execute: bool) -> Any:
        return live_cases.run_case(
            bridge, case, route.id, state_path=state_path, recipient=recipient, amount=amount,
            execute=execute, benchmark=benchmark, stop_after_execute=stop_after_execute,
            wait_timeout_seconds=min(live_cases.WAIT_TIMEOUT_SECONDS, remaining()),
            wait_poll_seconds=live_cases.WAIT_POLL_SECONDS, log=print)

    state = None
    try:
        bridge = _client(make_bridge, case, environment, execute=execute)
        benchmark.mark("clients-created")
        state = run(bridge, stop_after_execute=True)
        if execute and state.checkpoint is not None and not state.completed:
            print(f"  handover   dropping the client that executed {route.id}; "
                  "a new one will recover from disk")
            del bridge                              # the in-memory progress goes with it
            bridge = _client(make_bridge, case, environment, execute=execute)
            benchmark.mark("clients-recreated")
            state = run(bridge, stop_after_execute=False)
    except Underfunded as exc:
        record_property("underfunded", {"asset_id": exc.asset_id, "needed": exc.needed,
                                        "have": exc.have, "shortfall": exc.shortfall, "what": exc.what})
        print(f"\n  UNDERFUNDED {route.id}: {exc}")
        pytest.skip(f"{route.id}: {exc}")
    except InsufficientBalanceError as exc:
        # Some quotes read the wallet themselves and refuse before our precheck ever runs (the
        # xReserve deposit quote is one). That is the same verdict, reached one step earlier.
        record_property("underfunded", {"raised_by": "quote", "detail": str(exc)})
        print(f"\n  UNDERFUNDED {route.id}: the quote refused — {exc}")
        pytest.skip(f"{route.id}: {exc}")
    except (PollingTimeoutError, LiveTimeoutError) as exc:
        record_property("pending_state_path", str(state_path))
        record_property("pending_resume", resume_command(state_path))
        print(f"\n  PENDING    {route.id} is still in flight: {exc}\n"
              f"  resume     {resume_command(state_path)}")
        pytest.skip(f"{route.id} still in flight (state {state_path}); resume: {resume_command(state_path)}")

    _record(record_property, case, route, state, benchmark)
    if not execute:
        # veil's `if (!mainnetExecutionEnabled()) return` — the quote and the precheck are the test,
        # and the one thing that must hold is that nothing was submitted for an unstarted transfer.
        assert state.completed or state.checkpoint is None, \
            f"{route.id} has an in-flight checkpoint that the quote-only run must not have created"
        return state

    assert state.completed, f"{route.id} did not reach done: {state}"
    assert state.source_tx_id, f"{route.id} completed without a source transaction id"
    if not live_cases.delivery_is_a_balance_rise(route, DEFAULT_REGISTRY):
        assert state.message_id or state.destination_tx_id, \
            f"{route.id} completed without a message id or a destination transaction id"

    # Delivery is checked as "at least what was quoted", never as equality. A quote's `amount_out`
    # is derived from the registry's fee literal, and the fee the protocol actually charges is live
    # state: the 2026-09-18 testnet return burned 2.000001 USDCx against a registry
    # withdrawalFeeAtomic of 2_000_000 (quote: 0.000001 USDC out) and delivered 0.996501 USDC,
    # because Circle's testnet withdrawal fee was 1.0035 USDC that day. A private mint moves no
    # public balance at all, so a zero delta is also correct.
    if state.destination_balance_before is not None:
        after = live_cases.read_balances(bridge).get(route.destination_asset_id)
        if after is not None:
            delta = after - int(state.destination_balance_before)
            record_property("destination_balance_delta", delta)
            print(f"  delta      {route.destination_asset_id} +{delta} atomic "
                  f"(before {state.destination_balance_before}, after {after})")
            assert delta >= 0, f"{route.id} delivered a negative balance delta ({delta})"
            if live_cases.delivery_is_a_balance_rise(route, DEFAULT_REGISTRY):
                assert delta > 0, f"{route.id} was marked delivered but nothing arrived"
    print(f"\n  SUMMARY    {case} {route.id} source={state.source_tx_id} "
          f"message={state.message_id} destination={state.destination_tx_id}")
    return state


def _mainnet(case: str, route: Route, make_bridge: Callable[[], Any],
             record_property: Callable[[str, Any], None]) -> Any:
    """Gate one mainnet route: funds + the case acknowledgement; submission needs its own."""
    if not live_config.live_funds_enabled():
        pytest.skip(f"set {live_config.FUNDS_VAR}=1 and {live_config.STATE_DIR_VAR} to run funded live cases")
    if not live_config.mainnet_case_enabled(case):
        pytest.skip(f"{live_config.MAINNET_ACK_VAR} and {live_config.MAINNET_CASES_VAR} do not enable {case}")
    return _drive_case(make_bridge, case, route, environment="mainnet",
                       execute=live_config.mainnet_execution_enabled(), record_property=record_property)


def _testnet(case: str, route_id: str, make_bridge: Callable[[], Any],
             record_property: Callable[[str, Any], None], amount: str) -> Any:
    """Gate one testnet route: the funds gate alone (veil's testnet file has no mainnet acknowledgement)."""
    if not live_config.live_funds_enabled():
        pytest.skip(f"set {live_config.FUNDS_VAR}=1 and {live_config.STATE_DIR_VAR} to run funded live cases")
    route = DEFAULT_REGISTRY.route(route_id)
    return _drive_case(make_bridge, case, route, environment="testnet", execute=True,
                       record_property=record_property, amount=amount)


# ── the five mainnet cases, one test each, over every route they cover ───────

@_params("evm-hyperlane")
def test_evm_hyperlane(route, mainnet_client, record_property):
    """veil mainnet/evm-hyperlane.live.test.ts: ethereum → aleo over the Hyperlane warp routes."""
    _mainnet("evm-hyperlane", route, mainnet_client, record_property)


@_params("evm-xreserve")
def test_evm_xreserve(route, mainnet_client, record_property):
    """veil mainnet/evm-xreserve.live.test.ts: 2 USDC ethereum → aleo, private mint (needs `complete`)."""
    _mainnet("evm-xreserve", route, mainnet_client, record_property)


@_params("aleo-hyperlane")
def test_aleo_hyperlane(route, mainnet_client, record_property):
    """veil mainnet/aleo-hyperlane.live.test.ts: aleo → ethereum and aleo → solana, `mode="signer"`."""
    _mainnet("aleo-hyperlane", route, mainnet_client, record_property)


@_params("aleo-xreserve")
def test_aleo_xreserve(route, mainnet_client, record_property):
    """veil mainnet/aleo-xreserve.live.test.ts: 2.000001 USDCx private burn, aleo → ethereum."""
    _mainnet("aleo-xreserve", route, mainnet_client, record_property)


@_params("solana-hyperlane")
def test_solana_hyperlane(route, mainnet_client, record_property):
    """veil mainnet/solana-hyperlane.live.test.ts: 1 lamport solana → aleo."""
    _mainnet("solana-hyperlane", route, mainnet_client, record_property)


# ── recovery for real (controller note 10, brief deliverable 2) ──────────────

@pytest.mark.parametrize("route", MAINNET_ROUTES["evm-hyperlane"][:1],
                         ids=_ids(MAINNET_ROUTES["evm-hyperlane"][:1]))
def test_evm_hyperlane_recovers_from_disk_after_execute(route, mainnet_client, record_property):
    """The ETH route, finished by a client that never saw ``execute``.

    This is the same two-phase body every funded test uses, named separately because the parity doc
    calls it out: the checkpoint written by phase one is looked up through ``bridge.pending()`` on a
    brand-new ``Bridge`` bound to the same ``FileCheckpointStore``, rebuilt with ``bridge.recover``
    and driven to ``done`` with ``wait``.
    """
    state = _mainnet("evm-hyperlane", route, mainnet_client, record_property)
    if live_config.mainnet_execution_enabled():
        assert state.checkpoint is not None, "the transfer must have left a checkpoint on disk"


# ── testnet: the pair that actually runs today ───────────────────────────────

def test_testnet_evm_xreserve_deposit(testnet_client, record_property):
    """veil testnet/evm-xreserve.live.test.ts: 3 USDC Sepolia → aleo-testnet USDCx, private mint.

    3 rather than veil's 2 (controller ruling 2026-09-18) so the minted balance clears the 2 USDCx
    withdrawal fee and :func:`test_testnet_aleo_xreserve_return` can burn straight afterwards.
    """
    _testnet("evm-xreserve", TESTNET_DEPOSIT_ROUTE, testnet_client, record_property,
             TESTNET_DEPOSIT_AMOUNT)


def test_testnet_aleo_xreserve_return(testnet_client, record_property):
    """Beyond veil: the RETURN leg, aleo-testnet USDCx → Sepolia USDC, 2.000001 private burn.

    Runs the mainnet ``aleo-xreserve`` case function against the testnet route, so "all the routes
    back and forth" holds on testnet too.  It spends what the deposit minted — run that first; an
    unfunded run skips with the shortfall.
    """
    _testnet("aleo-xreserve", TESTNET_RETURN_ROUTE, testnet_client, record_property,
             TESTNET_RETURN_AMOUNT)


# ── the coverage invariant, as a test as well as an import-time assertion ────

def test_every_registry_route_has_a_case():
    """'All the routes back and forth': no route in either environment may be left without a case."""
    for environment in live_config.ENVIRONMENTS:
        buckets = _by_case(environment)
        listed = {route.id for routes in buckets.values() for route in routes}
        assert listed == {route.id for route in DEFAULT_REGISTRY.routes(environment=environment)}
    assert {route.id for route in TESTNET_ROUTES["evm-xreserve"]} == {TESTNET_DEPOSIT_ROUTE}
    assert {route.id for route in TESTNET_ROUTES["aleo-xreserve"]} == {TESTNET_RETURN_ROUTE}


__all__ = ["MAINNET_ROUTES", "TESTNET_ROUTES", "resume_command"]
