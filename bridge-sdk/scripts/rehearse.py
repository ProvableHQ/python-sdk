#!/usr/bin/env python3
"""Operator front-end over the live cases in ``tests/live/cases.py``.

    python scripts/rehearse.py --case evm-hyperlane --quote-only        # price every route, submit nothing
    python scripts/rehearse.py --case evm-hyperlane --route hyperlane:ethereum/eth->aleo/eth
    python scripts/rehearse.py --recover <state.json>                   # continue one interrupted transfer
    python scripts/rehearse.py --case solana-hyperlane --report run.json

The CLI and the pytest suite call the SAME case functions, so a rehearsal and a test cannot drift.

This script never sets, exports, prints or suggests a value for an acknowledgement variable.  It
READS ``BRIDGE_LIVE_FUNDS`` / ``BRIDGE_LIVE_STATE_DIR`` / ``BRIDGE_LIVE_MAINNET_ACK`` /
``BRIDGE_LIVE_MAINNET_CASES`` / ``BRIDGE_LIVE_MAINNET_EXECUTE``; without all of them it runs to the
quote and reports what it would have submitted.  Keys, secret nonces, attestations and record
plaintexts never reach the output — addresses, amounts and transaction ids do.

Exit codes: ``0`` everything ran (or was quoted/skipped), ``1`` a case failed, ``2`` a case is
still pending (a timeout leaves the checkpoint on disk; re-run with ``--recover``).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "python") not in sys.path:
    sys.path.insert(0, str(ROOT / "python"))

from aleo_bridge.errors import BridgeError, PollingTimeoutError            # noqa: E402
from tests.live import cases as live_cases                                 # noqa: E402
from tests.live import config as live_config                               # noqa: E402
from tests.live.helpers import (LiveBenchmark, LiveCaseError, LiveTimeoutError,   # noqa: E402
                                Underfunded, load_live_state)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_PENDING = 2

_COLUMNS = (("case", 16), ("route_id", 40), ("status", 11), ("source_tx_id", 18), ("reason", 40))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rehearse.py", description="Run one live bridge case over its registry routes.",
        epilog="Mainnet submission additionally requires the acknowledgement variables described "
               "in the README's 'Live tests' section; this script only reads them.")
    parser.add_argument("--case", choices=list(live_cases.CASE_NAMES),
                        help="the veil case to run (omit only with --recover)")
    parser.add_argument("--route", help="restrict the run to one registry route id")
    parser.add_argument("--quote-only", action="store_true",
                        help="price and preflight every route; never submit, whatever is acknowledged")
    parser.add_argument("--recover", metavar="STATE.json",
                        help="continue the transfer recorded in one state file (its route picks the case)")
    parser.add_argument("--report", metavar="PATH", help="write the JSON report here")
    args = parser.parse_args(argv)
    if not args.case and not args.recover:
        parser.error("give --case NAME or --recover STATE.json")
    return args


# ── target selection ──────────────────────────────────────────────────────────

def resolve_routes(registry: Any, case: str, route: str | None = None,
                   environment: str = "mainnet") -> list[Any]:
    """The routes this invocation covers: every route of *case*, or the single one asked for."""
    routes = live_cases.routes_for_case(registry, case, environment)
    if route is None:
        return routes
    chosen = [candidate for candidate in routes if candidate.id == route]
    if not chosen:
        raise SystemExit(f"Route {route} is not one of the {case} routes for {environment}")
    return chosen


def execution_allowed(case: str, *, quote_only: bool) -> tuple[bool, str]:
    """Whether the wallet may submit, and the reason when it may not (variable names only)."""
    if quote_only:
        return False, "--quote-only was given"
    if not live_config.live_funds_enabled():
        return False, (f"{live_config.FUNDS_VAR} and {live_config.STATE_DIR_VAR} do not enable "
                       "funded live cases")
    if not live_config.mainnet_case_enabled(case):
        return False, (f"{live_config.MAINNET_ACK_VAR} and {live_config.MAINNET_CASES_VAR} do not "
                       f"enable the {case} case")
    if not live_config.mainnet_execution_enabled():
        return False, f"{live_config.MAINNET_EXECUTE_VAR} does not acknowledge mainnet submission"
    return True, "acknowledged"


def state_path_for(case: str, route_id: str, environment: str) -> Path:
    return live_config.live_state_path(environment, live_cases.state_name(case, route_id))


def resume_command(state_path: Path | str) -> str:
    return f"python scripts/rehearse.py --recover {state_path}"


# ── running ───────────────────────────────────────────────────────────────────

def _row(case: str, route_id: str, status: str, reason: str = "", state: Any = None,
         state_path: Path | str | None = None, benchmark: LiveBenchmark | None = None) -> dict[str, Any]:
    return {
        "case": case,
        "route_id": route_id,
        "status": status,
        "reason": reason,
        "source_tx_id": getattr(state, "source_tx_id", None),
        "message_id": getattr(state, "message_id", None),
        "destination_tx_id": getattr(state, "destination_tx_id", None),
        "state_path": str(state_path) if state_path else None,
        "resume": resume_command(state_path) if state_path else "",
        "benchmark": benchmark.as_dict() if benchmark is not None else {},
    }


def run_route(bridge: Any, case: str, route: Any, *, execute: bool, state_path: Path,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    """One route: run the case, and turn every outcome into a report row rather than a traceback."""
    if not route.active:
        return _row(case, route.id, "skipped", f"registry availability: {route.availability}")
    benchmark = LiveBenchmark(f"{case}:{live_cases.route_slug(route.id)}", log=log)
    runner = live_cases.RUNNERS[case]
    try:
        state = runner(bridge, route.id, state_path=state_path, execute=execute, benchmark=benchmark,
                       log=log)
    except Underfunded as exc:
        return _row(case, route.id, "skipped", str(exc), state_path=state_path, benchmark=benchmark)
    except (PollingTimeoutError, LiveTimeoutError) as exc:
        return _row(case, route.id, "pending", f"still in flight: {exc}", state_path=state_path,
                    benchmark=benchmark)
    except (LiveCaseError, BridgeError) as exc:
        return _row(case, route.id, "failed", f"{type(exc).__name__}: {exc}", state_path=state_path,
                    benchmark=benchmark)
    status = "completed" if state.completed else ("quote-only" if not execute else "pending")
    reason = "" if status != "pending" else "the transfer has not reached done yet"
    return _row(case, route.id, status, reason, state=state, state_path=state_path, benchmark=benchmark)


def _cell(value: Any, width: int) -> str:
    text = str(value or "")
    return text.ljust(width) if len(text) <= width else text[:width - 1] + "…"


def render_table(rows: Iterable[dict[str, Any]]) -> str:
    """A fixed-width table of what ran, was skipped, is pending, or failed."""
    header = "  ".join(name.replace("_", " ").upper().ljust(width) for name, width in _COLUMNS)
    lines = [header, "-" * len(header)]
    lines.extend("  ".join(_cell(row.get(name), width) for name, width in _COLUMNS) for row in rows)
    return "\n".join(lines) + "\n"


def exit_code(rows: list[dict[str, Any]]) -> int:
    statuses = {row["status"] for row in rows}
    if "failed" in statuses:
        return EXIT_FAILED
    if "pending" in statuses:
        return EXIT_PENDING
    return EXIT_OK


def _default_bridge() -> Any:
    from aleo_bridge import Bridge

    # Bridge.from_env() already reads BRIDGE_LIVE_ETHEREUM_RPC_URL / BRIDGE_LIVE_SOLANA_RPC_URL as
    # aliases of ETHEREUM_RPC_URL / SOLANA_RPC_URL, so veil's shell works unchanged.
    return Bridge.from_env()


def run(argv: list[str] | None = None, *, bridge_factory: Callable[[], Any] = _default_bridge,
        log: Callable[[str], None] = print) -> int:
    args = parse_args(argv)
    bridge = bridge_factory()
    environment = bridge.environment

    if args.recover:
        state_path = Path(args.recover).expanduser().resolve()
        route_id = json.loads(state_path.read_text(encoding="utf-8")).get("routeId")
        if not isinstance(route_id, str):
            raise SystemExit(f"{state_path} does not name a routeId")
        route = bridge.registry.route(route_id)
        case = args.case or live_cases.case_for_route(bridge.registry, route)
        if case is None:
            raise SystemExit(f"No live case covers {route_id}")
        load_live_state(state_path, route_id)                 # fail closed before anything else
        targets = [(route, state_path)]
    else:
        case = args.case
        targets = [(route, state_path_for(case, route.id, environment))
                   for route in resolve_routes(bridge.registry, case, args.route, environment)]

    execute, reason = execution_allowed(case, quote_only=args.quote_only)
    log(f"case {case} · environment {environment} · registry {bridge.registry.version}")
    log(f"submission: {'ENABLED' if execute else 'disabled'} ({reason})")
    if not execute and not args.quote_only:
        log("Nothing will be submitted. Mainnet submission is gated on the acknowledgement variables "
            f"({live_config.MAINNET_ACK_VAR}, {live_config.MAINNET_CASES_VAR}, "
            f"{live_config.MAINNET_EXECUTE_VAR}); set them yourself, for one command, in your own shell.")

    rows = [run_route(bridge, case, route, execute=execute, state_path=path, log=log)
            for route, path in targets]

    log("")
    log(render_table(rows))
    for row in rows:
        if row["status"] == "pending":
            log(f"pending {row['route_id']}: {row['resume']}")

    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "case": case,
               "environment": environment, "registry_version": bridge.registry.version,
               "execute": execute, "reason": reason, "results": rows}
    if args.report:
        report = Path(args.report).expanduser()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log(f"report written to {report}")
    return exit_code(rows)


def main() -> int:
    return run()


if __name__ == "__main__":      # pragma: no cover — exercised through run()
    raise SystemExit(main())
