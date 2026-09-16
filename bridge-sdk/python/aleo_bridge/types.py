"""Typed results shared by every module (contract §types.py). Atomic amounts are ``int``; human amounts are ``str``."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    PREPARED = "PREPARED"
    SOURCE_APPROVAL_PENDING = "SOURCE_APPROVAL_PENDING"
    SOURCE_SUBMISSION_PENDING = "SOURCE_SUBMISSION_PENDING"
    SOURCE_CONFIRMING = "SOURCE_CONFIRMING"
    ATTESTATION_PENDING = "ATTESTATION_PENDING"
    DESTINATION_ACTION_REQUIRED = "DESTINATION_ACTION_REQUIRED"
    DELIVERY_PENDING = "DELIVERY_PENDING"
    DESTINATION_CONFIRMING = "DESTINATION_CONFIRMING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"

    def __str__(self) -> str:  # json.dumps and f-strings print the bare name
        return self.value


TERMINAL = {Status.COMPLETED, Status.FAILED, Status.EXPIRED}
CALLER_BOUNDARIES = {Status.SOURCE_SUBMISSION_PENDING, Status.DESTINATION_ACTION_REQUIRED,
                     Status.COMPLETED, Status.FAILED, Status.EXPIRED}

# brief §2.5
_NEXT_BY_STATUS = {
    Status.SOURCE_SUBMISSION_PENDING: "resume",
    Status.DESTINATION_ACTION_REQUIRED: "complete",
    Status.COMPLETED: "done",
    Status.FAILED: "failed",
    Status.EXPIRED: "failed",
}


@dataclass(frozen=True)
class Step:
    id: str
    kind: str          # approve|deposit|burn|dispatch|wait-attestation|mint|withdraw|wait-delivery|confirm-delivery
    executor: str      # aleo-wallet|evm-wallet|solana-wallet|protocol
    irreversible: bool


@dataclass(frozen=True)
class Fee:
    kind: str
    chain_id: str
    asset_id: str
    amount: str
    estimated: bool


@dataclass(frozen=True)
class Plan:
    route_id: str
    registry_version: str
    protocol: str
    environment: str
    source_asset_id: str
    destination_asset_id: str
    amount: str
    amount_atomic: int
    recipient: str
    sender: str | None
    mint_mode: str     # "public" | "record" | "private"
    steps: tuple[Step, ...]

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["steps"] = [dataclasses.asdict(s) for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Plan":
        data = dict(d)
        data["steps"] = tuple(Step(**s) for s in data.get("steps", ()))
        return cls(**data)


@dataclass(frozen=True)
class Quote:
    kind: str          # evm-hyperlane | solana-hyperlane | aleo-hyperlane | evm-xreserve | aleo-xreserve
    plan: Plan
    fees: tuple[Fee, ...]
    amount_out: str | None


@dataclass(frozen=True)
class EvmHyperlaneQuote(Quote):
    recipient_bytes32: bytes
    native_value_atomic: int
    native_fee_atomic: int
    approval_required: bool | None


@dataclass(frozen=True)
class SolanaHyperlaneQuote(Quote):
    igp_lamports: int
    network_fee_lamports: int
    rent_lamports: int
    total_lamports: int
    unique_message_address: str


@dataclass(frozen=True)
class AleoHyperlaneQuote(Quote):
    gas_limit: int
    gas_overhead: int
    gas_price: int
    exchange_rate: int
    payment_microcredits: int


@dataclass(frozen=True)
class EvmXReserveQuote(Quote):
    hook_data: bytes
    remote_recipient_bytes32: bytes
    balance_atomic: int
    allowance_atomic: int
    approval_required: bool
    max_fee_atomic: int


@dataclass(frozen=True)
class AleoXReserveQuote(Quote):
    withdrawal_fee_atomic: int


@dataclass
class Receipt:
    id: str
    protocol: str
    status: Status
    source_tx_id: str | None = None
    destination_tx_id: str | None = None
    protocol_state: dict[str, Any] = field(default_factory=dict)   # MUST include "routeId"
    next_action: dict[str, Any] | None = None                      # {"kind": "xreserve-private-mint", "chainId": ...}

    def __post_init__(self) -> None:
        self.status = Status(self.status)

    def replace(self, **changes: Any) -> "Receipt":
        return dataclasses.replace(self, **changes)


@dataclass(frozen=True)
class Progress:
    next: str          # "wait" | "resume" | "complete" | "done" | "failed"
    plan: Plan
    receipt: Receipt
    error: str | None = None


def to_progress(plan: Plan, receipt: Receipt) -> Progress:
    """brief §2.5: SOURCE_SUBMISSION_PENDING→resume, DESTINATION_ACTION_REQUIRED→complete, COMPLETED→done,
    FAILED|EXPIRED→failed, everything else→wait."""
    if "routeId" not in receipt.protocol_state:
        raise ValueError("Receipt.protocol_state must carry routeId")
    return Progress(_NEXT_BY_STATUS.get(Status(receipt.status), "wait"), plan, receipt)


@dataclass(frozen=True)
class GasQuote:
    route_id: str
    gas_limit: int
    gas_overhead: int
    gas_price: int
    exchange_rate: int
    payment_microcredits: int


@dataclass(frozen=True)
class Attestation:
    payload: bytes         # 305 bytes
    message_hash: bytes    # 32 bytes
    attestation: bytes     # 65 bytes
    status: str            # "complete"


@dataclass(frozen=True)
class DispatchReceipt:
    transaction_id: str
    route_id: str
    message_id: str | None
    amount_atomic: int
    receipt: Receipt


@dataclass(frozen=True)
class BurnReceipt:
    transaction_id: str
    route_id: str
    mode: str
    amount_atomic: int
    receipt: Receipt


@dataclass(frozen=True)
class MintReceipt:
    transaction_id: str
    route_id: str
    receipt: Receipt


@dataclass(frozen=True)
class DepositReceipt:
    transaction_id: str
    route_id: str
    message_hash: str
    nonce: str
    receipt: Receipt


@dataclass(frozen=True)
class PrivacyReceipt:
    transaction_id: str
    asset_id: str
    amount: str
    amount_atomic: int
    direction: str         # "shield" | "unshield"


@dataclass(frozen=True)
class PreparedTx:
    transaction_id: str
    serialized: str        # Transaction JSON; rebroadcast via network.submit_transaction(serialized)


@dataclass
class ChainStatus:
    chain_id: str
    address: str | None
    can_sign: bool
    balances: dict[str, int]   # asset_id -> atomic


@dataclass
class BridgeStatus:
    environment: str
    registry_version: str
    chains: list[ChainStatus]
    pending: list[Progress]


__all__ = [
    "CALLER_BOUNDARIES", "TERMINAL", "AleoHyperlaneQuote", "AleoXReserveQuote", "Attestation", "BridgeStatus",
    "BurnReceipt", "ChainStatus", "DepositReceipt", "DispatchReceipt", "EvmHyperlaneQuote", "EvmXReserveQuote",
    "Fee", "GasQuote", "MintReceipt", "Plan", "PreparedTx", "PrivacyReceipt", "Progress", "Quote", "Receipt",
    "SolanaHyperlaneQuote", "Status", "Step", "to_progress",
]
