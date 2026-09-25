"""Checkpoints — the allowlisted, versioned recovery record (spec §8, brief §2.10).

A checkpoint carries the public transfer intent, the route id + registry
version, and the transaction ids already submitted (plus, for Aleo legs, the
exact proved-but-unbroadcast transaction). It never carries keys, record
plaintext, the private-mint secret nonce, attestation bodies, payloads,
message hashes, nonces, or quote internals — ``create_checkpoint`` copies an
allowlist, not ``protocol_state``.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .errors import BridgeError, CheckpointInvalidError
from .registry import Registry
from .types import Plan, Receipt

CHECKPOINT_VERSION = 1
_DIGITS = re.compile(r"^\d+$")


@dataclass(frozen=True)
class Checkpoint:
    """Version-1 recovery record. Dict/JSON keys are veil's camelCase names."""

    version: int
    receipt_id: str
    intent: dict[str, Any]
    route: dict[str, str]
    source: dict[str, Any] | None = None
    destination: dict[str, Any] | None = None
    delivery_verification: dict[str, str] | None = None
    journal_id: str | None = None

    @property
    def id(self) -> str:
        """Stable journal key, or the receipt id for a legacy checkpoint."""
        return self.journal_id or self.receipt_id

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"version": self.version, "receiptId": self.receipt_id,
                                "intent": self.intent, "route": self.route}
        if self.journal_id is not None:
            out["journalId"] = self.journal_id
        if self.source:
            out["source"] = self.source
        if self.destination:
            out["destination"] = self.destination
        if self.delivery_verification:
            out["deliveryVerification"] = self.delivery_verification
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        if not isinstance(data, dict) or data.get("version") != CHECKPOINT_VERSION \
                or not isinstance(data.get("intent"), dict) or not isinstance(data.get("route"), dict):
            raise CheckpointInvalidError(
                "Bridge checkpoint format is invalid or unsupported: expected version 1 "
                "with 'intent' and 'route' objects")
        journal_id = data.get("journalId")
        if journal_id is not None and (not isinstance(journal_id, str) or not _JOURNAL_ID.fullmatch(journal_id)):
            raise CheckpointInvalidError("Invalid journalId: expected a dated transfer filename without .json")
        source = data.get("source") or None
        destination = data.get("destination") or None
        receipt_id = data.get("receiptId") or _derive_id(source, destination)
        return cls(version=1, receipt_id=receipt_id, intent=dict(data["intent"]),
                   route=dict(data["route"]), source=source, destination=destination,
                   delivery_verification=data.get("deliveryVerification") or None, journal_id=journal_id)

    @classmethod
    def from_json(cls, text: str) -> "Checkpoint":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as exc:
            raise CheckpointInvalidError(f"Bridge checkpoint is not valid JSON: {exc}") from exc


def _derive_id(source: dict[str, Any] | None, destination: dict[str, Any] | None) -> str:
    """veil-shaped checkpoints have no receiptId: use the most recent transaction id."""
    for container, keys in ((destination, ("transactionId", "preparedTransaction")),
                            (source, ("preparedTransaction", "transactionId"))):
        for key in keys:
            value = (container or {}).get(key)
            if isinstance(value, dict) and value.get("transactionId"):
                return str(value["transactionId"])
            if isinstance(value, str) and value:
                return value
    approvals = (source or {}).get("approvalTransactionIds") or []
    if approvals:
        return str(approvals[-1])
    raise CheckpointInvalidError(
        "Bridge checkpoint contains no submitted or prepared transaction to identify it by")


def _validated_prepared(value: Any, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CheckpointInvalidError(f"Bridge receipt contains an invalid {what}")
    return value


def create_checkpoint(plan: Plan, receipt: Receipt, registry: Registry) -> Checkpoint:
    """Reduce *receipt* to the documented recovery fields (allowlist, brief §2.10).

    Raises :class:`CheckpointInvalidError` when the receipt belongs to another
    route/protocol or carries malformed approval ids, prepared transactions,
    Solana blockhash lifetime, or delivery-verification state.
    """
    state = receipt.protocol_state
    if receipt.protocol != plan.protocol or state.get("routeId") != plan.route_id:
        raise CheckpointInvalidError("Bridge receipt does not match the prepared route")

    raw_approvals = state.get("approvalTxIds")
    if raw_approvals is not None and (not isinstance(raw_approvals, list)
                                       or any(not isinstance(v, str) for v in raw_approvals)):
        raise CheckpointInvalidError("Bridge receipt contains invalid approval transaction identifiers")
    approvals = list(raw_approvals or [])

    source_sender = state.get("sourceSender")
    if source_sender is not None and not isinstance(source_sender, str):
        raise CheckpointInvalidError("Bridge receipt contains an invalid source sender")
    sender = plan.sender or source_sender

    prepared = _validated_prepared(state.get("preparedTransaction"), "prepared transaction")
    prepared_destination = _validated_prepared(state.get("preparedDestinationTransaction"),
                                               "prepared destination transaction")

    blockhash = state.get("blockhash")
    last_valid = state.get("lastValidBlockHeight")
    if (blockhash is not None or last_valid is not None) and (
            not isinstance(blockhash, str) or not blockhash
            or not isinstance(last_valid, str) or not _DIGITS.match(last_valid)):
        raise CheckpointInvalidError("Bridge receipt contains an invalid Solana blockhash lifetime")

    before = state.get("destinationBalanceBeforeAtomic")
    expected = state.get("expectedDestinationIncreaseAtomic")
    if (before is not None or expected is not None) and (
            not isinstance(before, str) or not _DIGITS.match(before)
            or not isinstance(expected, str) or not _DIGITS.match(expected)):
        raise CheckpointInvalidError(
            "Bridge receipt contains invalid destination balance verification state")

    source: dict[str, Any] | None = None
    if approvals or receipt.source_tx_id or prepared:
        source = {}
        if approvals:
            source["approvalTransactionIds"] = approvals
        if receipt.source_tx_id:
            source["transactionId"] = receipt.source_tx_id
        if isinstance(state.get("hookData"), str):
            source["hookData"] = state["hookData"]
        if state.get("dropped") is True and isinstance(state.get("sourceError"), str):
            # The verdict that this source transaction can never mine. Kept as a pair so a listing
            # rebuilt offline (Bridge.pending) reports the transfer as failed-and-explained rather
            # than as still confirming a hash nothing will ever mine.
            source["dropped"] = True
            source["sourceError"] = state["sourceError"]
        if isinstance(state.get("sourceNonce"), str) and state["sourceNonce"].isdigit():
            # The EVM nonce the source transaction was broadcast at: recovery needs it to tell a
            # slow transaction from one that was dropped or replaced. Version stays 1 — a
            # checkpoint written before this existed simply has no sourceNonce.
            source["sourceNonce"] = state["sourceNonce"]
        if isinstance(blockhash, str) and isinstance(last_valid, str):
            source["blockhash"] = blockhash
            source["lastValidBlockHeight"] = last_valid
        if prepared is not None:
            source["preparedTransaction"] = {"transactionId": receipt.id,
                                             "serializedTransaction": prepared}

    destination: dict[str, Any] | None = None
    if receipt.destination_tx_id or prepared_destination is not None:
        destination = {}
        if receipt.destination_tx_id:
            destination["transactionId"] = receipt.destination_tx_id
        if prepared_destination is not None:
            destination["preparedTransaction"] = {"transactionId": receipt.id,
                                                  "serializedTransaction": prepared_destination}

    src_asset = registry.asset(plan.source_asset_id)
    dst_asset = registry.asset(plan.destination_asset_id)
    intent: dict[str, Any] = {
        "source": {"chain": src_asset.chain_id, "asset": src_asset.key},
        "destination": {"chain": dst_asset.chain_id, "asset": dst_asset.key},
        "bridgeProtocol": plan.protocol,
        "amount": plan.amount,
        "recipient": plan.recipient,
    }
    if sender:
        intent["sender"] = sender
    if dst_asset.locator is not None and dst_asset.locator.kind == "aleo-program":
        intent["mintMode"] = plan.mint_mode

    verification = ({"balanceBeforeAtomic": before, "expectedIncreaseAtomic": expected}
                    if isinstance(before, str) and isinstance(expected, str) else None)
    return Checkpoint(version=CHECKPOINT_VERSION, receipt_id=receipt.id, intent=intent,
                      route={"id": plan.route_id, "registryVersion": plan.registry_version},
                      source=source, destination=destination, delivery_verification=verification, journal_id=plan.journal_id)


@dataclass(frozen=True)
class CheckpointProblem:
    """One stored record that could not be read back as a checkpoint.

    A half-written file, a foreign JSON file dropped into the directory, or a record written by a
    future version is reported as one of these instead of aborting the listing — the transfers
    alongside it still come back.
    """

    path: str
    error: str
    error_type: str

    def to_dict(self) -> dict[str, str]:
        """The failure entry ``Bridge.pending()`` / ``bridge_pending`` report for this file."""
        return {"error": self.error, "error_type": self.error_type, "path": self.path}


@dataclass(frozen=True)
class CheckpointLoadResult:
    """Return readable checkpoints and errors encountered loading saved files.

    ``checkpoints`` are ordered oldest first by modification time. Each entry in
    ``errors`` identifies an unreadable file and the reason it could not load.
    """

    checkpoints: list[Checkpoint]
    errors: list[CheckpointProblem]


@runtime_checkable
class CheckpointStore(Protocol):
    """Where checkpoints live between processes. Implement all four methods."""

    def save(self, checkpoint: Checkpoint) -> None: ...
    def load(self, checkpoint_id: str) -> Checkpoint | None: ...
    def list(self) -> list[Checkpoint]: ...
    def delete(self, checkpoint_id: str) -> None: ...


_JOURNAL_ID = re.compile(r"\d{4}-\d{2}-\d{2}_\d{3,}_[A-Za-z0-9_-]+_to_[A-Za-z0-9_-]+_[0-9]+(?:\.[0-9]+)?")


@contextmanager
def _counter_lock(path: Path):
    # A separate, persistent lock inode survives atomic replacement of counter data.
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    with os.fdopen(fd, "r+b") as lock:
        if os.name == "nt":
            import msvcrt
            if not path.stat().st_size:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


class FileCheckpointStore:
    """One recovery JSON per transfer under *directory*; mode 0600; atomic rename.

    ``Bridge.execute`` reserves a UTC date/counter name before submission, for example
    ``2026-09-25_001_ethereum-wbtc_to_aleo-wbtc_0.001.json``. This journal id survives
    receipt changes and recovery. Legacy checkpoints retain their receipt-based filenames.

    ``list()`` returns oldest-first by mtime, skipping any file it cannot read
    back as a checkpoint. ``load_checkpoints().errors`` reports those files, so
    one bad file never hides the transfers beside it. ``delete()`` of a missing id is a no-op. Ids are sanitized for the
    filesystem; ``receiptId`` remains the on-chain receipt identity, separate from ``journalId``.
    Keep the hidden ``.journal-counter`` and ``.journal.lock`` files when moving the journal;
    counters are unique within a directory, not across independent machines.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory).expanduser()     # "~/.aleo-bridge/checkpoints" means the home directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, checkpoint_id: str) -> Path:
        safe = checkpoint_id if _JOURNAL_ID.fullmatch(checkpoint_id) else (_UNSAFE.sub("_", checkpoint_id) or "_")
        return self.directory / f"{safe}.json"

    def reserve(self, plan: Plan) -> str:
        """Reserve a UTC date/counter name before submission; never reuse deleted entries.

        The hidden counter and lock are journal metadata, not recovery checkpoints. A failed
        submission can leave a gap in numbering. Custom stores need not implement this method.
        """
        source = plan.source_asset_id.replace("/", "-")
        destination = plan.destination_asset_id.replace("/", "-")
        day = datetime.now(timezone.utc).date().isoformat()
        counter_path = self.directory / ".journal-counter"
        with _counter_lock(self.directory / ".journal.lock"):
            try:
                counters = json.loads(counter_path.read_text()) if counter_path.exists() else {}
            except (ValueError, UnicodeError) as exc:
                raise CheckpointInvalidError("Journal counter is unreadable; restore it before submitting") from exc
            if not isinstance(counters, dict) or any(type(v) is not int or v < 0 for v in counters.values()):
                raise CheckpointInvalidError("Journal counter is invalid; restore it before submitting a transfer")
            existing = [int(p.name.split("_")[1]) for p in self.directory.glob(f"{day}_*.json")
                        if _JOURNAL_ID.fullmatch(p.stem)]
            counter = max(counters.get(day, 0), max(existing, default=0)) + 1
            name = f"{day}_{counter:03d}_{source}_to_{destination}_{plan.amount}"
            if not _JOURNAL_ID.fullmatch(name):
                raise CheckpointInvalidError("Transfer details cannot form a journal filename")
            counters[day] = counter
            self._write(counter_path, json.dumps(counters))
        return name

    def save(self, checkpoint: Checkpoint) -> None:
        self._write(self._path(checkpoint.id), checkpoint.to_json())

    def _write(self, target: Path, text: str) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
            if os.name != "nt":
                directory_fd = os.open(self.directory, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    def load(self, checkpoint_id: str) -> Checkpoint | None:
        path = self._path(checkpoint_id)
        if not path.exists():
            return None
        return Checkpoint.from_json(path.read_text(encoding="utf-8"))

    def _paths(self) -> list[Path]:
        return sorted((p for p in self.directory.glob("*.json") if not p.name.startswith(".")),
                      key=lambda p: (p.stat().st_mtime_ns, p.name))

    def load_checkpoints(self) -> CheckpointLoadResult:
        """Load saved checkpoints and report files that could not be read or parsed.

        One unreadable file must never hide every in-flight transfer: a record this store cannot
        parse (bad JSON, a future version, no transaction to identify it by) or cannot even read
        (permissions, a half-written file, non-UTF-8 bytes) is skipped here and reported, so
        ``Bridge.pending()`` can list the healthy transfers AND say what it could not read.
        """
        checkpoints: list[Checkpoint] = []
        problems: list[CheckpointProblem] = []
        for path in self._paths():
            try:
                checkpoints.append(Checkpoint.from_json(path.read_text(encoding="utf-8")))
            except (BridgeError, OSError, UnicodeDecodeError) as exc:
                problems.append(CheckpointProblem(path=str(path), error=str(exc),
                                                  error_type=type(exc).__name__))
        return CheckpointLoadResult(checkpoints=checkpoints, errors=problems)

    def list(self) -> list[Checkpoint]:
        """Return readable checkpoints oldest first; use ``load_checkpoints`` for load errors."""
        return self.load_checkpoints().checkpoints

    def delete(self, checkpoint_id: str) -> None:
        try:
            self._path(checkpoint_id).unlink()
        except FileNotFoundError:
            pass


__all__ = ["CheckpointLoadResult", "Checkpoint", "CheckpointProblem", "CheckpointStore", "FileCheckpointStore", "create_checkpoint"]
