"""The funded live cases — one function per veil case, driven only by the public ``Bridge`` verbs.

Ported from veil's `test/integration/live/mainnet/*.live.test.ts` (and `testnet/evm-xreserve`):
five cases, each parametrized over every registry route it covers, each resuming from its own
state file.  The pytest suite and ``scripts/rehearse.py`` both call these functions, so the two can
never drift.

Shape of every case (veil parity §4):

1. load the state file for this route (fails closed; a completed case re-asserts and returns),
2. ``quote`` → print the route/amount/fee table → balance precheck (:class:`Underfunded` when the
   wallet cannot cover it, which callers turn into a skip),
3. ``execute=False`` stops here — that is veil's ``if (!mainnetExecutionEnabled()) return``,
4. otherwise ``execute(plan, on_checkpoint=…)`` once, saving the checkpoint to the state file at
   every boundary, then drop the in-memory progress and ``recover`` from what is on disk,
5. drive ``wait`` / ``resume`` / ``complete`` until ``next == "done"`` (``failed`` raises),
6. record the source tx, message id, destination tx and balance delta; mark the state completed.

What this module will never do: retry ``execute`` after a broadcast (ambiguous or not), read or
set an acknowledgement variable (the caller passes ``execute=``), or print a key, a secret nonce,
an attestation or a record plaintext.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aleo_bridge.errors import InsufficientBalanceError
from aleo_bridge.privacy import record_amount
from aleo_bridge.registry import Asset, Registry, Route
from aleo_bridge.units import format_decimal_amount

from .config import one_atomic_unit
from .helpers import (LiveBenchmark, LiveCaseError, LiveState, Underfunded, ensure_secret_nonce,
                      load_live_state, load_secret_nonce, redacted, save_live_state, wait_for,
                      wait_for_aleo_transaction, wait_for_hyperlane_delivery)

#: ERC-20 ``Transfer(address,address,uint256)``.
_TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

#: veil's per-test budget (`30 * 60_000`).
CASE_TIMEOUT_SECONDS = 30 * 60.0
WAIT_TIMEOUT_SECONDS = 20 * 60.0
WAIT_POLL_SECONDS = 15.0
_MAX_TRANSITIONS = 12          # a drive loop that cannot settle is a bug, not a slow chain


@dataclass(frozen=True)
class CaseSpec:
    """One veil case: which routes it covers, what it sends, and how ``execute`` is parametrized."""

    name: str
    protocol: str                  # "hyperlane" | "xreserve"
    source_family: str             # "evm" | "aleo" | "solana"
    mint_mode: str = "public"      # "private" for the xReserve private mint into Aleo
    mode: str | None = None        # execute(mode=…): "signer" (Aleo Hyperlane) / "private" (Aleo burn)
    amount: str | None = None      # veil's literal, or None → one atomic unit of the source asset
    veil_source: str = ""

    @property
    def private_mint(self) -> bool:
        return self.mint_mode == "private"


CASES: dict[str, CaseSpec] = {
    "evm-hyperlane": CaseSpec(
        name="evm-hyperlane", protocol="hyperlane", source_family="evm",
        veil_source="mainnet/evm-hyperlane.live.test.ts:21-115"),
    "evm-xreserve": CaseSpec(
        name="evm-xreserve", protocol="xreserve", source_family="evm", mint_mode="private",
        amount="2",                                    # veil mainnet/evm-xreserve.live.test.ts:57
        veil_source="mainnet/evm-xreserve.live.test.ts:31-146"),
    "aleo-hyperlane": CaseSpec(
        name="aleo-hyperlane", protocol="hyperlane", source_family="aleo", mode="signer",
        veil_source="mainnet/aleo-hyperlane.live.test.ts:47-164"),
    "aleo-xreserve": CaseSpec(
        name="aleo-xreserve", protocol="xreserve", source_family="aleo", mode="private",
        amount="2.000001",                             # veil mainnet/aleo-xreserve.live.test.ts:97
        veil_source="mainnet/aleo-xreserve.live.test.ts:69-156"),
    "solana-hyperlane": CaseSpec(
        name="solana-hyperlane", protocol="hyperlane", source_family="solana",
        veil_source="mainnet/solana-hyperlane.live.test.ts:20-91"),
}

CASE_NAMES = tuple(CASES)


# ── routes, names, defaults ───────────────────────────────────────────────────

def routes_for_case(registry: Registry, case: str, environment: str = "mainnet") -> list[Route]:
    """Every registry route *case* covers — both directions are separate cases, so this is one way.

    Includes non-active routes so the caller can report them as skipped-by-registry rather than
    silently dropping them (veil parity §2).
    """
    spec = CASES[case]
    return [route for route in registry.routes(environment=environment, protocol=spec.protocol)
            if registry.chain(registry.asset(route.source_asset_id).chain_id).family == spec.source_family]


def case_for_route(registry: Registry, route: Route) -> str | None:
    """The case that covers *route*, or None — 13b asserts no active mainnet route returns None."""
    family = registry.chain(registry.asset(route.source_asset_id).chain_id).family
    for spec in CASES.values():
        if spec.protocol == route.protocol and spec.source_family == family:
            return spec.name
    return None


def route_slug(route_id: str) -> str:
    """``hyperlane:ethereum/eth->aleo/eth`` → ``hyperlane-ethereum-eth-aleo-eth`` (veil's state-name rule)."""
    return re.sub(r"^-|-$", "", re.sub(r"[^a-z0-9]+", "-", route_id, flags=re.IGNORECASE))


def state_name(case: str, route_id: str) -> str:
    """The state file's base name. Always route-qualified, so a case that covers several routes
    (evm-hyperlane over eth/wbtc/usdt, aleo-hyperlane over eth/wbtc/usdt/sol) can never resume one
    route's checkpoint under another's — veil only did this for aleo-hyperlane."""
    return f"{case}-{route_slug(route_id)}"


def asset_ref(asset: Asset) -> str:
    return f"{asset.chain_id}/{asset.key}"


def default_amount(registry: Registry, case: str, route: Route) -> str:
    """veil's literal for the xReserve cases, otherwise one atomic unit of the source asset."""
    spec = CASES[case]
    return spec.amount or one_atomic_unit(registry.asset(route.source_asset_id).decimals)


def sender_for(bridge: Any, route: Route) -> str | None:
    """The address that will sign the source leg, read from the configured connection."""
    family = bridge.registry.chain(bridge.registry.asset(route.source_asset_id).chain_id).family
    if family == "aleo":
        return bridge.aleo_address()
    connection = bridge.ethereum if family == "evm" else bridge.solana
    if connection is None:
        raise LiveCaseError(f"No {family} connection is configured for {route.id}")
    return connection.address


def default_recipient(bridge: Any, route: Route) -> str:
    """Our own address on the destination chain (the operator overrides it through the config vars)."""
    family = bridge.registry.chain(bridge.registry.asset(route.destination_asset_id).chain_id).family
    if family == "aleo":
        return bridge.aleo_address()
    # Ruling: probe the CONNECTION, never the bridge.eth/bridge.sol properties — those raise.
    connection = bridge.ethereum if family == "evm" else bridge.solana
    if connection is None or connection.address is None:
        raise LiveCaseError(f"No {family} address is configured to receive {route.id}")
    return connection.address


# ── printing (addresses and amounts only, never secrets) ──────────────────────

def _human(bridge: Any, atomic: int, asset_id: str) -> str:
    return f"{format_decimal_amount(atomic, bridge.registry.asset(asset_id).decimals)} " \
           f"{bridge.registry.asset(asset_id).symbol}"


def print_quote(bridge: Any, quote: Any, *, case: str, route_id: str, log: Callable[[str], None] = print) -> None:
    """veil's ``console.table``: what is about to move, where, and what it costs."""
    plan = quote.plan
    source, destination = bridge.registry.asset(plan.source_asset_id), bridge.registry.asset(plan.destination_asset_id)
    log(f"\n== {case} [{plan.environment}] {route_id}")
    log(f"  kind       {quote.kind} (registry {plan.registry_version})")
    log(f"  amount     {plan.amount} {source.symbol}  →  {quote.amount_out or plan.amount} {destination.symbol}")
    log(f"  sender     {plan.sender}")
    log(f"  recipient  {plan.recipient}")
    log(f"  mint mode  {plan.mint_mode}")
    for fee in quote.fees:
        estimated = " (estimated)" if fee.estimated else ""
        log(f"  fee        {fee.amount} {bridge.registry.asset(fee.asset_id).symbol} [{fee.kind}]{estimated}")
    # An EvmXReserveQuote carries no `fees`: its protocol cost is the max fee the deposit authorizes.
    max_fee = getattr(quote, "max_fee_atomic", None)
    if max_fee is not None:
        log(f"  fee        {_human(bridge, max_fee, source.id)} [xReserve max fee]")
    for name in ("native_value_atomic", "native_fee_atomic", "approval_required", "total_lamports",
                 "igp_lamports", "rent_lamports", "payment_microcredits", "balance_atomic",
                 "allowance_atomic", "withdrawal_fee_atomic", "max_fee_atomic"):
        if hasattr(quote, name):
            log(f"  {name:<24}{getattr(quote, name)}")
    steps = " → ".join(f"{step.id}{'*' if step.irreversible else ''}" for step in plan.steps)
    log(f"  steps      {steps}   (* irreversible)")


# ── balances ──────────────────────────────────────────────────────────────────

def read_balances(bridge: Any) -> dict[str, int]:
    """``asset_id → atomic`` across every configured chain, or ``{}`` when a chain cannot be read.

    ``Bridge.status()`` already walks exactly the configured connections, so an Aleo-only client
    simply reports no EVM/Solana rows instead of raising.
    """
    try:
        return {asset_id: atomic for chain in bridge.status().chains for asset_id, atomic in chain.balances.items()}
    except Exception:                                  # noqa: BLE001 — a precheck must never fail the case
        return {}


def _require(balances: dict[str, int], asset_id: str, needed: int, *, what: str = "balance",
             log: Callable[[str], None] = print) -> None:
    have = balances.get(asset_id)
    if have is None:
        log(f"  {what:<10} {asset_id}: unreadable — proceeding; the chain will enforce it")
        return
    log(f"  {what:<10} {asset_id}: {have} atomic (need {needed})")
    if have < needed:
        raise Underfunded(asset_id=asset_id, needed=needed, have=have, what=what)


def _native_asset_id(bridge: Any, chain_id: str) -> str | None:
    native = [a for a in bridge.registry.assets(chain=chain_id) if a.kind == "native"]
    return native[0].id if native else None


def precheck(bridge: Any, quote: Any, *, case: str, log: Callable[[str], None] = print) -> dict[str, int]:
    """Refuse to spend what the wallet does not have; raises :class:`Underfunded` with the shortfall."""
    plan = quote.plan
    balances = read_balances(bridge)
    source = bridge.registry.asset(plan.source_asset_id)
    native_id = _native_asset_id(bridge, source.chain_id)

    if quote.kind == "evm-hyperlane":
        if source.kind == "native":
            _require(balances, source.id, quote.native_value_atomic, log=log)
        else:
            _require(balances, source.id, plan.amount_atomic, log=log)
            if native_id:
                _require(balances, native_id, quote.native_fee_atomic, what="gas", log=log)
    elif quote.kind == "solana-hyperlane":
        if native_id:
            _require(balances, native_id, quote.total_lamports, log=log)
    elif quote.kind == "aleo-hyperlane":
        _require(balances, source.id, plan.amount_atomic, log=log)
        if native_id:
            _require(balances, native_id, quote.payment_microcredits, what="hook fee", log=log)
    elif quote.kind == "evm-xreserve":
        # The quote already read the wallet's USDC balance and allowance on chain.
        if quote.balance_atomic < plan.amount_atomic:
            raise Underfunded(asset_id=source.id, needed=plan.amount_atomic, have=quote.balance_atomic)
        log(f"  balance    {source.id}: {quote.balance_atomic} atomic (need {plan.amount_atomic})")
    elif quote.kind == "aleo-xreserve":
        pass            # the private record is selected (and checked) right before the burn
    return balances


# ── the shared engine ─────────────────────────────────────────────────────────

def _saver(state: LiveState, state_path: Path, benchmark: LiveBenchmark) -> Callable[[Any], None]:
    def save(checkpoint: Any) -> None:
        state.checkpoint = checkpoint.to_dict() if hasattr(checkpoint, "to_dict") else dict(checkpoint)
        source = state.checkpoint.get("source") or {}
        state.source_tx_id = source.get("transactionId") or state.source_tx_id
        destination = state.checkpoint.get("destination") or {}
        state.destination_tx_id = destination.get("transactionId") or state.destination_tx_id
        save_live_state(state_path, state)
        benchmark.mark("checkpoint-saved")
    return save


def _assert_completed(state: LiveState, case: str) -> LiveState:
    """Re-running a finished case is a no-op that re-asserts what was recorded (veil parity §4)."""
    if not state.source_tx_id:
        raise LiveCaseError(f"{case} state claims completion without a source transaction id")
    return state


def _recover_from_state(bridge: Any, state: LiveState, *, benchmark: LiveBenchmark,
                        log: Callable[[str], None]) -> Any:
    """Rebuild progress from the checkpoint ON DISK, dropping whatever ``execute`` returned.

    The user's instruction requires the recovery verbs proven for real: the checkpoint the store
    holds — found through ``bridge.pending()`` when a store is bound — is the only input here.
    """
    if state.checkpoint is None:
        raise LiveCaseError("No checkpoint was saved for this transfer; nothing to recover from")
    receipt_id = state.checkpoint.get("receiptId")
    pending = bridge.pending() or []
    if receipt_id and any(entry.receipt.id == receipt_id for entry in pending):
        log(f"  pending    {receipt_id} is in the bound checkpoint store ({len(pending)} in flight)")
    progress = bridge.recover(state.checkpoint)
    benchmark.mark("source-recovered")
    return progress


def _drive(bridge: Any, progress: Any, state: LiveState, state_path: Path, *, spec: CaseSpec,
           secret_nonce: str | None, benchmark: LiveBenchmark, save: Callable[[Any], None],
           wait_timeout_seconds: float, wait_poll_seconds: float,
           log: Callable[[str], None]) -> Any:
    """``wait`` / ``resume`` / ``complete`` until the transfer is done. Never calls ``execute``."""
    for _ in range(_MAX_TRANSITIONS):
        if progress.next == "wait":
            progress = bridge.wait(progress, timeout_seconds=wait_timeout_seconds,
                                   poll_seconds=wait_poll_seconds,
                                   on_error=lambda exc: log(f"  transient  {type(exc).__name__}: {exc}"))
            benchmark.mark("wait-returned")
            continue
        if progress.next == "resume":
            progress = bridge.resume(progress, on_checkpoint=save, secret_nonce=secret_nonce)
            benchmark.mark("resume-returned")
            continue
        if progress.next == "complete":
            if not secret_nonce:
                raise LiveCaseError(
                    "The private mint needs the secret nonce kept beside this case's state file; "
                    "it is missing, so the mint cannot be completed here")
            progress = bridge.complete(progress, secret_nonce=secret_nonce, on_checkpoint=save)
            benchmark.mark("complete-returned")
            continue
        if progress.next == "failed":
            raise LiveCaseError(f"{spec.name} failed: {progress.error}")
        if progress.next == "done":
            return progress
        raise LiveCaseError(f"Unexpected progress state {progress.next!r} for {spec.name}")
    raise LiveCaseError(f"{spec.name} did not settle after {_MAX_TRANSITIONS} lifecycle transitions")


def _select_private_record(bridge: Any, route: Route, amount_atomic: int,
                           log: Callable[[str], None]) -> str:
    """The USDCx record the burn will spend, logged by amount and digest — never by plaintext."""
    program = route.meta_str("remoteToken")
    try:
        record = bridge.privacy.select_record(program, amount_atomic)
    except InsufficientBalanceError as exc:
        raise Underfunded(asset_id=bridge.registry.asset(route.source_asset_id).id,
                          needed=amount_atomic, have=0, what="private record") from exc
    log(f"  record     {program} {redacted([record])} amount={record_amount(record)}")
    return record


def completion_has_no_destination_id(route: Route, registry: Registry) -> bool:
    """True for every Aleo-origin leg: the SDK proves delivery by the recipient's balance rising
    (``get_status`` branch 6 for Hyperlane, branch 8 for xReserve) and records neither a message
    id nor a destination transaction id — veil's ``aleo-hyperlane.live.test.ts:163`` likewise
    asserts only ``completed`` and ``sourceTxId`` there (the 2026-09-22 mainnet
    ``aleo/eth->ethereum/eth`` leg completed exactly this way, +1 wei on Ethereum)."""
    return registry.chain(registry.asset(route.source_asset_id).chain_id).family == "aleo"


def delivery_is_a_balance_rise(route: Route, registry: Registry) -> bool:
    """True for the Aleo→EVM xReserve withdrawal, the one leg with no delivery query anywhere.

    ``lifecycle.py`` says it in as many words ("xReserve Aleo→EVM: Circle exposes no canonical
    delivery query") and simply returns the receipt unchanged, so its status stays
    ``DELIVERY_PENDING`` for ever and ``wait`` can only ever time out.  veil does not drive this
    case to ``done`` either — ``aleo-xreserve.live.test.ts:146-155`` polls the recipient's ERC-20
    balance until it rises above what it was before the burn, and that is what delivery means here.
    """
    if route.protocol != "xreserve":
        return False
    source = registry.chain(registry.asset(route.source_asset_id).chain_id).family
    destination = registry.chain(registry.asset(route.destination_asset_id).chain_id).family
    return source == "aleo" and destination == "evm"


def _wait_for_balance_rise(bridge: Any, asset_id: str, before: int, *, timeout_seconds: float,
                           poll_seconds: float, log: Callable[[str], None]) -> int:
    """veil ``waitFor(... balanceOf > destinationBalanceBefore)``: the recipient's balance, once it rises."""
    def read() -> Any:
        after = read_balances(bridge).get(asset_id)
        return after if after is not None and after > before else None

    log(f"  awaiting   {asset_id} to rise above {before} atomic (no delivery query exists for this leg)")
    return wait_for(read, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds)


def _evm_transfer_tx(bridge: Any, asset: Asset, recipient: str, amount_atomic: int,
                     lookback_blocks: int = 5_000) -> str | None:
    """The transaction that moved *amount_atomic* of *asset* to *recipient*, or None.

    Best effort only: this is a convenience id for the report, so a public RPC that refuses the
    log range (or returns nothing) leaves ``destinationTxId`` unset rather than failing a leg whose
    funds have demonstrably arrived.
    """
    connection = getattr(bridge, "ethereum", None)
    if connection is None or asset.locator.kind != "evm-contract":
        return None
    try:
        w3 = connection.w3
        head = w3.eth.block_number
        entries = w3.eth.get_logs({
            "fromBlock": max(head - lookback_blocks, 0), "toBlock": head,
            "address": w3.to_checksum_address(asset.locator.value),
            "topics": [_TRANSFER_TOPIC, None, "0x" + "00" * 12 + recipient[2:].lower()]})
    except Exception:                                  # noqa: BLE001 — a missing id is not a failure
        return None
    for entry in reversed(list(entries)):
        raw = entry["data"]
        value = int(raw.hex() if hasattr(raw, "hex") else raw, 16)
        if value == amount_atomic:
            digest = entry["transactionHash"]
            return "0x" + (digest.hex() if hasattr(digest, "hex") else str(digest)).removeprefix("0x")
    return None


def run_case(bridge: Any, case: str, route_id: str, *, state_path: Path | str, recipient: str | None = None,
             amount: str | None = None, execute: bool, benchmark: LiveBenchmark | None = None,
             wait_timeout_seconds: float = WAIT_TIMEOUT_SECONDS, wait_poll_seconds: float = WAIT_POLL_SECONDS,
             stop_after_execute: bool = False, log: Callable[[str], None] = print) -> LiveState:
    """Run one case over one route, resuming from ``state_path``. See the module docstring for the flow.

    ``stop_after_execute=True`` returns as soon as the source transaction is on chain and its
    checkpoint is on disk, so the caller can throw the whole client away and prove that a *new*
    ``Bridge`` finishes the transfer from the state file alone.  Calling ``run_case`` again with the
    same ``state_path`` takes the resume branch; ``execute`` is never called twice for one transfer.
    """
    spec = CASES[case]
    state_path = Path(state_path)
    benchmark = benchmark if benchmark is not None else LiveBenchmark(case, log=log)
    state = load_live_state(state_path, route_id)
    if state.completed:
        log(f"  {case} {route_id} is already complete (source {state.source_tx_id})")
        return _assert_completed(state, case)

    route = bridge.registry.route(route_id)
    if not route.active:
        raise LiveCaseError(f"Route {route_id} is {route.availability}; the registry will not execute it")
    if route.protocol != spec.protocol:
        raise LiveCaseError(f"Route {route_id} is {route.protocol}, not a {spec.protocol} case")
    source, destination = bridge.registry.asset(route.source_asset_id), bridge.registry.asset(route.destination_asset_id)
    amount = amount or default_amount(bridge.registry, case, route)
    recipient = recipient or default_recipient(bridge, route)
    sender = sender_for(bridge, route)
    benchmark.mark("plan-prepared")

    # The private-mint nonce is created only when we are actually going to deposit: a quote-only
    # rehearsal leaves no secret file behind. An existing one is always reused.
    secret_nonce = None
    if spec.private_mint:
        secret_nonce = ensure_secret_nonce(state_path) if execute else load_secret_nonce(state_path)
        state.secret_nonce_present = secret_nonce is not None

    progress = None
    if state.checkpoint is None:
        quote = bridge.quote(asset_ref(source), asset_ref(destination), amount=amount, recipient=recipient,
                             sender=sender, protocol=route.protocol, mint_mode=spec.mint_mode,
                             secret_nonce=secret_nonce or "0scalar")
        benchmark.mark("quote-returned")
        print_quote(bridge, quote, case=case, route_id=route_id, log=log)
        balances = precheck(bridge, quote, case=case, log=log)
        before = balances.get(destination.id)
        if before is not None:
            state.destination_balance_before = str(before)
        if not execute:
            log(f"\n  quote only — nothing was submitted for {route_id}.")
            return state
        save_live_state(state_path, state)

        record = (_select_private_record(bridge, route, quote.plan.amount_atomic, log)
                  if case == "aleo-xreserve" else None)
        save = _saver(state, state_path, benchmark)
        gas = getattr(quote, "payment_microcredits", None) if spec.protocol == "hyperlane" else None
        progress = bridge.execute(quote.plan, on_checkpoint=save, mode=spec.mode, record=record,
                                  secret_nonce=secret_nonce, gas_payment_microcredits=gas)
        benchmark.mark("execute-returned")
        state.source_tx_id = progress.receipt.source_tx_id or state.source_tx_id
        save_live_state(state_path, state)
        if progress.receipt.protocol_state.get("blockhashExpired") is True:
            raise LiveCaseError(
                f"Solana source transaction {state.source_tx_id} expired; inspect it on chain before "
                "clearing the checkpoint — never re-run execute")
        if spec.source_family == "aleo" and state.source_tx_id:
            # veil aleo-hyperlane:153 / aleo-xreserve:136: confirm the source on chain before
            # recovering, so a rejected execution is reported as itself rather than as a timeout.
            wait_for_aleo_transaction(bridge, state.source_tx_id)
            benchmark.mark("source-confirmed")
        if stop_after_execute:
            log(f"  handover   source={state.source_tx_id} checkpoint on disk at {state_path}; "
                "a new client will recover it")
            return state
    else:
        log(f"  resuming {case} {route_id} from the saved checkpoint")

    save = _saver(state, state_path, benchmark)
    if not execute:
        # A saved checkpoint plus no acknowledgement: report what is pending, submit nothing.
        log(f"\n  quote only — {route_id} has a saved checkpoint; re-run with the acknowledgement to finish it.")
        return state

    progress = _recover_from_state(bridge, state, benchmark=benchmark, log=log)

    if delivery_is_a_balance_rise(route, bridge.registry):
        # No drive loop: `wait` on this leg can only time out (see delivery_is_a_balance_rise).
        before = int(state.destination_balance_before or 0)
        after = _wait_for_balance_rise(bridge, destination.id, before, timeout_seconds=wait_timeout_seconds,
                                       poll_seconds=wait_poll_seconds, log=log)
        benchmark.mark("destination-delivered")
        state.destination_tx_id = state.destination_tx_id or _evm_transfer_tx(
            bridge, destination, recipient, after - before)
        log(f"  delivered  {destination.id} +{after - before} atomic (before {before}, after {after}) "
            f"tx={state.destination_tx_id}")
        state.completed = True
        save_live_state(state_path, state)
        log(f"  done       source={state.source_tx_id} destination={state.destination_tx_id}")
        log(f"  {benchmark.summary()}")
        return state

    progress = _drive(bridge, progress, state, state_path, spec=spec, secret_nonce=secret_nonce,
                      benchmark=benchmark, save=save, wait_timeout_seconds=wait_timeout_seconds,
                      wait_poll_seconds=wait_poll_seconds, log=log)
    benchmark.mark("destination-delivered")

    receipt = progress.receipt
    state.source_tx_id = receipt.source_tx_id or state.source_tx_id
    state.message_id = receipt.protocol_state.get("messageId") or state.message_id
    state.destination_tx_id = receipt.destination_tx_id or state.destination_tx_id
    if spec.protocol == "hyperlane" and state.destination_tx_id is None and state.source_tx_id:
        delivery = wait_for_hyperlane_delivery(state.source_tx_id, timeout_seconds=300, poll_seconds=15)
        if delivery is not None:
            state.message_id = state.message_id or delivery.message_id
            state.destination_tx_id = delivery.destination_tx_id
        else:
            log("  explorer   unavailable; the destination transaction id was not recorded")
    state.completed = True
    save_live_state(state_path, state)
    after = read_balances(bridge).get(destination.id)
    if after is not None and state.destination_balance_before is not None:
        log(f"  delivered  {destination.id} +{after - int(state.destination_balance_before)} atomic "
            f"(before {state.destination_balance_before}, after {after})")
    log(f"  done       source={state.source_tx_id} message={state.message_id} "
        f"destination={state.destination_tx_id}")
    log(f"  {benchmark.summary()}")
    return state


def _runner(case: str) -> Callable[..., LiveState]:
    def run(bridge: Any, route_id: str, *, state_path: Path | str, recipient: str | None = None,
            amount: str | None = None, execute: bool, benchmark: LiveBenchmark | None = None,
            **kwargs: Any) -> LiveState:
        return run_case(bridge, case, route_id, state_path=state_path, recipient=recipient, amount=amount,
                        execute=execute, benchmark=benchmark, **kwargs)
    run.__name__ = f"run_{case.replace('-', '_')}"
    run.__doc__ = (f"veil {CASES[case].veil_source}: {case} over one route "
                   f"(amount {CASES[case].amount or 'one atomic unit'}"
                   f"{', private mint' if CASES[case].private_mint else ''}"
                   f"{f', execute mode {CASES[case].mode}' if CASES[case].mode else ''}).")
    return run


run_evm_hyperlane = _runner("evm-hyperlane")
run_evm_xreserve = _runner("evm-xreserve")
run_aleo_hyperlane = _runner("aleo-hyperlane")
run_aleo_xreserve = _runner("aleo-xreserve")
run_solana_hyperlane = _runner("solana-hyperlane")

RUNNERS: dict[str, Callable[..., LiveState]] = {
    "evm-hyperlane": run_evm_hyperlane,
    "evm-xreserve": run_evm_xreserve,
    "aleo-hyperlane": run_aleo_hyperlane,
    "aleo-xreserve": run_aleo_xreserve,
    "solana-hyperlane": run_solana_hyperlane,
}

__all__ = [
    "CASES", "CASE_NAMES", "CASE_TIMEOUT_SECONDS", "CaseSpec", "LiveCaseError", "RUNNERS", "Underfunded",
    "asset_ref", "case_for_route", "completion_has_no_destination_id", "default_amount", "default_recipient",
    "delivery_is_a_balance_rise",
    "precheck", "print_quote", "read_balances", "route_slug", "routes_for_case", "run_aleo_hyperlane",
    "run_aleo_xreserve", "run_case", "run_evm_hyperlane", "run_evm_xreserve", "run_solana_hyperlane",
    "sender_for", "state_name",
]
