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

    @property
    def id(self) -> str:
        """The receipt id this checkpoint was created from (store key)."""
        return self.receipt_id

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"version": self.version, "receiptId": self.receipt_id,
                                "intent": self.intent, "route": self.route}
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
        source = data.get("source") or None
        destination = data.get("destination") or None
        receipt_id = data.get("receiptId") or _derive_id(source, destination)
        return cls(version=1, receipt_id=receipt_id, intent=dict(data["intent"]),
                   route=dict(data["route"]), source=source, destination=destination,
                   delivery_verification=data.get("deliveryVerification") or None)

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
                      source=source, destination=destination, delivery_verification=verification)


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


@runtime_checkable
class CheckpointStore(Protocol):
    """Where checkpoints live between processes. Implement all four methods."""

    def save(self, checkpoint: Checkpoint) -> None: ...
    def load(self, checkpoint_id: str) -> Checkpoint | None: ...
    def list(self) -> list[Checkpoint]: ...
    def delete(self, checkpoint_id: str) -> None: ...


_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


class FileCheckpointStore:
    """One ``<id>.json`` per receipt id under *directory*; mode 0600; atomic rename.

    ``list()`` returns oldest-first by mtime, skipping any file it cannot read
    back as a checkpoint — those come back from ``list_problems()`` /
    ``list_with_problems()`` instead, so one bad file never hides the transfers
    beside it. ``delete()`` of a missing id is a no-op. Ids are sanitized for the
    filesystem; the stored ``receiptId`` keeps the original.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, checkpoint_id: str) -> Path:
        safe = _UNSAFE.sub("_", checkpoint_id) or "_"
        return self.directory / f"{safe}.json"

    def save(self, checkpoint: Checkpoint) -> None:
        target = self._path(checkpoint.id)
        fd, tmp = tempfile.mkstemp(dir=self.directory, prefix=".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(checkpoint.to_json())
                fh.write("\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, target)
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

    def list_with_problems(self) -> tuple[list[Checkpoint], list[CheckpointProblem]]:
        """Every readable checkpoint, plus a :class:`CheckpointProblem` per file that is not one.

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
        return checkpoints, problems

    def list(self) -> list[Checkpoint]:
        """Oldest-first by mtime; files that are not readable checkpoints are skipped (see
        :meth:`list_problems`)."""
        return self.list_with_problems()[0]

    def list_problems(self) -> list[CheckpointProblem]:
        """The files ``list()`` skipped, one entry each."""
        return self.list_with_problems()[1]

    def delete(self, checkpoint_id: str) -> None:
        try:
            self._path(checkpoint_id).unlink()
        except FileNotFoundError:
            pass


__all__ = ["Checkpoint", "CheckpointProblem", "CheckpointStore", "FileCheckpointStore", "create_checkpoint"]
