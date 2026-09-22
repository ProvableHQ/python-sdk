"""Persistence, polling and timing for the funded live cases — a port of veil's
`test/integration/live/helpers.ts` (plus the secret-nonce file veil does not have).

Nothing here signs or submits anything.  What it does own is the memory of a run: the state file
that lets a case resume across processes, the private-mint secret that must survive a crash but
must never reach a log, a checkpoint or the state JSON, and the read-only lookups that answer
"did it arrive?".

State files are veil-shaped on disk (camelCase keys, one JSON object per case) and fail CLOSED:
a corrupt or wrong-route file raises rather than quietly starting a fresh transfer over funds that
may already be in flight.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from aleo_bridge._base58 import b58decode
from aleo_bridge.encoding import validate_scalar
from aleo_bridge.lifecycle import aleo_transaction_status

#: veil helpers.ts:85 — the Hyperlane explorer's public GraphQL endpoint (read-only).
HYPERLANE_EXPLORER_URL = "https://explorer4.hasura.app/v1/graphql"
HYPERLANE_QUERY = """query ByOrigin($hash: bytea!) {
  message_view(where: {origin_tx_hash: {_eq: $hash}}, limit: 1) {
    msg_id is_delivered destination_tx_hash
  }
}"""

DEFAULT_TIMEOUT_SECONDS = 1200.0       # veil helpers.ts:70 — 20 minutes
DEFAULT_POLL_SECONDS = 15.0

_SECRET_SUFFIX = ".secret"
_STATE_FIELDS = {
    "route_id": "routeId",
    "source_tx_id": "sourceTxId",
    "message_id": "messageId",
    "destination_tx_id": "destinationTxId",
    "destination_balance_before": "destinationBalanceBefore",
    "completed": "completed",
    "checkpoint": "checkpoint",
    "secret_nonce_present": "secretNoncePresent",
}
_STRING_FIELDS = ("source_tx_id", "message_id", "destination_tx_id", "destination_balance_before")


class LiveStateError(Exception):
    """A state file could not be trusted: corrupt, malformed, or bound to another route."""


class LiveTimeoutError(Exception):
    """A live verification ran out of time; the transfer is still in flight (never a failure verdict)."""


class ExplorerError(Exception):
    """The Hyperlane explorer answered with a GraphQL error rather than data."""


class LiveCaseError(Exception):
    """A live case cannot continue (a rejected source transaction, an expired blockhash, …)."""


class Underfunded(Exception):
    """A wallet cannot cover amount + fees. Callers turn this into a skip, printing the shortfall."""

    def __init__(self, *, asset_id: str, needed: int, have: int, what: str = "balance") -> None:
        self.asset_id, self.needed, self.have, self.what = asset_id, needed, have, what
        self.shortfall = max(needed - have, 0)
        super().__init__(f"Insufficient {asset_id} {what}: need {needed} atomic units, have {have} "
                         f"(short {self.shortfall})")


# ── state files ───────────────────────────────────────────────────────────────

@dataclass
class LiveState:
    """veil ``LiveState`` + ``secretNoncePresent``.

    The private-mint nonce itself is NOT a field: it lives in the sibling ``<state>.secret`` file
    (mode 0600) and only its presence is recorded here, so a state file can be pasted into a bug
    report without leaking the commitment secret.
    """

    route_id: str
    source_tx_id: str | None = None
    message_id: str | None = None
    destination_tx_id: str | None = None
    destination_balance_before: str | None = None
    completed: bool = False
    checkpoint: dict[str, Any] | None = None
    secret_nonce_present: bool = False

    def to_dict(self) -> dict[str, Any]:
        """veil's on-disk shape: camelCase keys, unset optionals omitted."""
        raw = asdict(self)
        out: dict[str, Any] = {}
        for attribute, key in _STATE_FIELDS.items():
            value = raw[attribute]
            if value is None or value is False:
                continue
            out[key] = value
        out["routeId"] = self.route_id
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LiveState":
        return cls(**{attribute: data.get(key) for attribute, key in _STATE_FIELDS.items()
                      if data.get(key) is not None})


def _validate(state: LiveState, route_id: str, path: Path) -> LiveState:
    if state.route_id != route_id:
        raise LiveStateError(f"Live state route {state.route_id!r} does not match {route_id!r}: {path}")
    for attribute in _STRING_FIELDS:
        value = getattr(state, attribute)
        if value is not None and not isinstance(value, str):
            raise LiveStateError(f"Live state {_STATE_FIELDS[attribute]} is invalid: {path}")
    if not isinstance(state.completed, bool):
        raise LiveStateError(f"Live state completed flag is invalid: {path}")
    if not isinstance(state.secret_nonce_present, bool):
        raise LiveStateError(f"Live state secretNoncePresent flag is invalid: {path}")
    if state.checkpoint is not None and not isinstance(state.checkpoint, dict):
        raise LiveStateError(f"Live state checkpoint is invalid: {path}")
    return state


def load_live_state(path: Path | str, route_id: str) -> LiveState:
    """veil ``loadLiveState()``: the saved state for *route_id*, or an empty one when absent.

    Fails closed on corrupt JSON, a non-object payload, a wrong route or a malformed field — a
    state file we cannot read is never treated as "no transfer in flight".
    """
    target = Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return LiveState(route_id=route_id)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LiveStateError(f"Live state is not valid JSON: {target} ({exc})") from exc
    if not isinstance(parsed, dict):
        raise LiveStateError(f"Live state is not an object: {target}")
    if not isinstance(parsed.get("routeId"), str):
        raise LiveStateError(f"Live state routeId is invalid: {target}")
    return _validate(LiveState.from_dict(parsed), route_id, target)


def save_live_state(path: Path | str, state: LiveState) -> Path:
    """veil ``saveLiveState()``: atomic (temp file + ``os.replace``), mode 0600, parents 0700."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = target.with_name(f".{target.name}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(state.to_dict(), indent=2, sort_keys=True))
            handle.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target


# ── the private-mint secret nonce (beside the state, never inside it) ─────────

def generate_secret_nonce() -> str:
    """A fresh Aleo scalar literal for one private mint (248 random bits, below the scalar modulus)."""
    import secrets

    return validate_scalar(f"{secrets.randbits(248)}scalar")


def secret_path(state_path: Path | str) -> Path:
    """``<state>.secret`` — the sibling file that holds the nonce for that case."""
    target = Path(state_path)
    return target.with_name(target.name + _SECRET_SUFFIX)


def save_secret_nonce(state_path: Path | str, nonce: str) -> Path:
    """Write *nonce* exclusively (``O_CREAT|O_EXCL``, 0600). An existing file raises rather than being
    overwritten: the nonce of a deposit already on chain is the only way to finish that mint."""
    validate_scalar(nonce)
    target = secret_path(state_path)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(f"{nonce}\n")
    return target


def load_secret_nonce(state_path: Path | str) -> str | None:
    """The kept nonce for this case, or None when there is no secret file."""
    try:
        text = secret_path(state_path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return validate_scalar(text)


def ensure_secret_nonce(state_path: Path | str) -> str:
    """The case's nonce: the kept one if the secret file exists, otherwise a fresh one, saved once."""
    existing = load_secret_nonce(state_path)
    if existing is not None:
        return existing
    nonce = generate_secret_nonce()
    save_secret_nonce(state_path, nonce)
    return nonce


# ── benchmark marks ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Mark:
    step: str
    elapsed_ms: int
    total_ms: int


@dataclass
class LiveBenchmark:
    """veil ``createLiveBenchmark()``: per-step and total elapsed milliseconds, printed as they happen.

    ``marks`` is kept so a pytest case can hand the whole timeline to ``record_property``.
    """

    label: str
    now: Callable[[], float] = time.monotonic
    log: Callable[[str], None] = print
    marks: list[Mark] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._started = self.now()
        self._previous = self._started

    def mark(self, step: str) -> Mark:
        current = self.now()
        entry = Mark(step, round((current - self._previous) * 1000), round((current - self._started) * 1000))
        self._previous = current
        self.marks.append(entry)
        self.log(f"[{self.label}] {step}: +{entry.elapsed_ms}ms (total {entry.total_ms}ms)")
        return entry

    def summary(self) -> str:
        if not self.marks:
            return f"{self.label}: no marks"
        steps = ", ".join(f"{m.step} +{m.elapsed_ms}ms" for m in self.marks)
        return f"{self.label}: {steps} (total {self.marks[-1].total_ms}ms)"

    def as_dict(self) -> dict[str, int]:
        return {m.step: m.elapsed_ms for m in self.marks}


# ── polling ───────────────────────────────────────────────────────────────────

def wait_for(read: Callable[[], Any], *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
             poll_seconds: float = DEFAULT_POLL_SECONDS, sleep: Callable[[float], None] = time.sleep,
             now: Callable[[], float] = time.monotonic) -> Any:
    """veil ``waitFor()``: poll *read* until it returns something that is not None.

    Always reads at least once, then checks the deadline — so a zero timeout still performs one
    read.  A timeout raises :class:`LiveTimeoutError`; the transfer is not failed, it is unfinished,
    and the state file on disk is how you resume it.
    """
    deadline = now() + timeout_seconds
    while True:
        value = read()
        if value is not None:
            return value
        if now() >= deadline:
            raise LiveTimeoutError(
                "Live bridge verification timed out; the transfer is still in flight — "
                "resume from the persisted state file")
        sleep(poll_seconds)


# ── Hyperlane explorer (read-only HTTP) ───────────────────────────────────────

@dataclass(frozen=True)
class HyperlaneDelivery:
    message_id: str
    destination_tx_id: str


def _post(url: str, payload: dict[str, Any], timeout: float) -> Any:
    import requests

    return requests.post(url, json=payload, timeout=timeout,
                         headers={"content-type": "application/json"})


def _bytea(source_tx_id: str) -> str:
    """veil helpers.ts:81-83: an EVM ``0x`` hash or a Solana base58 signature → PostgreSQL bytea."""
    if source_tx_id.startswith("0x"):
        return f"\\x{source_tx_id[2:]}"
    return f"\\x{b58decode(source_tx_id).hex()}"


def _normalize(value: str) -> str:
    return f"0x{value[2:]}" if value.startswith("\\x") else value


def hyperlane_delivery(source_tx_id: str, *, post: Callable[..., Any] = _post,
                       timeout: float = 30.0) -> HyperlaneDelivery | None:
    """One read of the Hyperlane explorer for the message dispatched by *source_tx_id*.

    Returns None while the message is undelivered AND when the explorer is throttled or broken
    (HTTP 429/5xx, or an unreachable host): a rate-limited explorer says nothing about the
    transfer.  A GraphQL error is a bug in the query and is raised (veil helpers.ts:104-106).
    """
    payload = {"query": HYPERLANE_QUERY, "variables": {"hash": _bytea(source_tx_id)}}
    try:
        response = post(HYPERLANE_EXPLORER_URL, payload, timeout)
    except Exception:                                   # noqa: BLE001 — network flake, never a verdict
        return None
    status = getattr(response, "status_code", 200)
    if status == 429 or status >= 500:
        return None
    if status >= 400:
        raise ExplorerError(f"Hyperlane explorer returned HTTP {status}")
    body = response.json()
    errors = body.get("errors") if isinstance(body, dict) else None
    if errors:
        joined = "; ".join(str(e.get("message", "unknown error")) for e in errors)
        raise ExplorerError(f"Hyperlane explorer query failed: {joined}")
    rows = ((body.get("data") or {}).get("message_view") or []) if isinstance(body, dict) else []
    message = rows[0] if rows else None
    if not message or not message.get("is_delivered") or not message.get("msg_id") \
            or not message.get("destination_tx_hash"):
        return None
    return HyperlaneDelivery(message_id=_normalize(message["msg_id"]),
                             destination_tx_id=_normalize(message["destination_tx_hash"]))


def wait_for_hyperlane_delivery(source_tx_id: str, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                                poll_seconds: float = DEFAULT_POLL_SECONDS,
                                sleep: Callable[[float], None] = time.sleep,
                                now: Callable[[], float] = time.monotonic,
                                post: Callable[..., Any] = _post) -> HyperlaneDelivery | None:
    """Poll the explorer for the destination transaction id, or give up quietly.

    This lookup is a convenience on top of a leg the SDK has already called ``done``, so a timeout
    or a throttled explorer returns None instead of failing the case (veil parity §8).
    """
    try:
        return wait_for(lambda: hyperlane_delivery(source_tx_id, post=post),
                        timeout_seconds=timeout_seconds, poll_seconds=poll_seconds, sleep=sleep, now=now)
    except LiveTimeoutError:
        return None


# ── Aleo confirmation ─────────────────────────────────────────────────────────

def wait_for_aleo_transaction(bridge: Any, tx_id: str, *, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
                              poll_seconds: float = DEFAULT_POLL_SECONDS,
                              sleep: Callable[[float], None] = time.sleep,
                              now: Callable[[], float] = time.monotonic) -> None:
    """veil ``waitForAleoTransaction()``: block until *tx_id* is accepted; a rejection raises."""

    def read() -> Any:
        status, error = aleo_transaction_status(bridge, tx_id)
        if status == "rejected":
            raise LiveCaseError(error or f"Aleo transaction {tx_id} was rejected")
        return True if status == "accepted" else None

    wait_for(read, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds, sleep=sleep, now=now)


def redacted(values: Iterable[str]) -> str:
    """A stable, non-reversible tag for a secret-bearing string (a record plaintext), for logs."""
    import hashlib

    digest = hashlib.sha256("".join(values).encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


def build_bridge(environment: str) -> Any:
    """A fresh client for *environment* — new facade, new connections, same checkpoint store.

    Keys and endpoints are resolved only through ``config``: the testnet bridge can never pick up
    the mainnet Aleo key, and an unset RPC variable falls back to the public default. The pytest
    suite and ``scripts/rehearse.py --recover`` share this one builder, so a rehearsal can never
    resume a transfer with a different account than the suite would have used.
    """
    from tests.live import config as live_config

    from aleo_bridge import Bridge
    from aleo_bridge.checkpoint import FileCheckpointStore
    from aleo_bridge.client import build_aleo
    from aleo_bridge.eth import Ethereum
    from aleo_bridge.sol import Solana

    aleo = build_aleo(live_config.aleo_endpoint(), live_config.ALEO_NETWORKS[environment],
                      live_config.aleo_private_key(environment))
    ethereum = Ethereum(live_config.evm_rpc_url(environment),
                        private_key=live_config.evm_private_key(environment))
    solana = Solana.from_env() if environment == "mainnet" else None
    store = FileCheckpointStore(live_config.state_dir() / environment / "checkpoints")
    bridge = Bridge(aleo, ethereum=ethereum, solana=solana, checkpoints=store)
    assert bridge.environment == environment
    return bridge


__all__ = [
    "build_bridge",
    "DEFAULT_POLL_SECONDS", "DEFAULT_TIMEOUT_SECONDS", "ExplorerError", "HYPERLANE_EXPLORER_URL",
    "HYPERLANE_QUERY", "HyperlaneDelivery", "LiveBenchmark", "LiveCaseError", "LiveState",
    "LiveStateError", "LiveTimeoutError", "Mark", "Underfunded", "ensure_secret_nonce",
    "generate_secret_nonce", "hyperlane_delivery", "load_live_state", "load_secret_nonce", "redacted",
    "save_live_state", "save_secret_nonce", "secret_path", "wait_for", "wait_for_aleo_transaction",
    "wait_for_hyperlane_delivery",
]
