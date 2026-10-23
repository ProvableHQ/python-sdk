"""Private record inventory: inspect, join, split, plan, rebalance, reserve, recover.

A private balance is a set of encrypted records. One large spend needs a
record that covers it; several concurrent spends need several records. This
package reads one asset's spendable records, plans the joins and splits that
reach a target layout, executes that plan one confirmed transaction at a
time, keeps participating writers from spending the same record, and
recovers an interrupted run without rebuilding a proof.

Bind it to a client with ``Aleo(provider, record_management=...)`` and use
``aleo.inventory``, or construct :class:`RecordInventory` directly.
"""
from __future__ import annotations

from ._engine import (
    PreparedTransaction as PreparedTransaction,
    inventory_output_commitments as inventory_output_commitments,
    record_input_ids as record_input_ids,
    transaction_fee as transaction_fee,
)
from .assets import (
    CREDITS_SPLIT_DEDUCTION as CREDITS_SPLIT_DEDUCTION,
    AssetCapabilities as AssetCapabilities,
    decode_record as decode_record,
    inventory_record_id as inventory_record_id,
    scope_for as scope_for,
    validate_asset as validate_asset,
)
from .interface import (
    ProgramFunction as ProgramFunction,
    ProgramInterface as ProgramInterface,
    ProgramParam as ProgramParam,
    ProgramRecord as ProgramRecord,
    RecordField as RecordField,
    parse_program_interface as parse_program_interface,
)
from .join_programs import (
    RECORD_JOIN_PROGRAMS as RECORD_JOIN_PROGRAMS,
    validate_record_join_program as validate_record_join_program,
    validate_token_join_router as validate_token_join_router,
)
from .client import (
    AsyncRecordInventory as AsyncRecordInventory,
    RecordInventory as RecordInventory,
    SpendCheckpoint as SpendCheckpoint,
    managed_inventory as managed_inventory,
    prepared_from_payload as prepared_from_payload,
    prepared_from_result as prepared_from_result,
)
from .planner import build_inventory_plan as build_inventory_plan
from .store import (
    JsonFileRecordInventoryStore as JsonFileRecordInventoryStore,
    MemoryRecordInventoryStore as MemoryRecordInventoryStore,
    SqliteRecordInventoryStore as SqliteRecordInventoryStore,
    held_record_ids as held_record_ids,
    holds_records as holds_records,
)
from .types import (
    InventoryError as InventoryError,
    InventoryPlan as InventoryPlan,
    InventoryPlanError as InventoryPlanError,
    InventoryRecord as InventoryRecord,
    InventoryResult as InventoryResult,
    InventoryStep as InventoryStep,
    InventoryTarget as InventoryTarget,
    PlannedRecord as PlannedRecord,
    RecordAsset as RecordAsset,
    RecordInventorySnapshot as RecordInventorySnapshot,
    RecordInventoryStore as RecordInventoryStore,
    RecordJoinProgram as RecordJoinProgram,
    RecordManagementConfig as RecordManagementConfig,
    RecordReservationConflictError as RecordReservationConflictError,
    RecordSpend as RecordSpend,
    RecordSpendPatch as RecordSpendPatch,
    ResolvedRecordAsset as ResolvedRecordAsset,
    TokenJoinRouter as TokenJoinRouter,
)

__all__ = [
    "AssetCapabilities", "AsyncRecordInventory", "CREDITS_SPLIT_DEDUCTION", "InventoryError", "InventoryPlan",
    "InventoryPlanError", "InventoryRecord", "InventoryResult", "InventoryStep", "InventoryTarget",
    "JsonFileRecordInventoryStore", "MemoryRecordInventoryStore", "PlannedRecord", "PreparedTransaction",
    "ProgramFunction", "ProgramInterface", "ProgramParam", "ProgramRecord", "RECORD_JOIN_PROGRAMS", "RecordAsset",
    "RecordField", "RecordInventory", "RecordInventorySnapshot", "RecordInventoryStore", "RecordJoinProgram",
    "RecordManagementConfig", "RecordReservationConflictError", "RecordSpend", "RecordSpendPatch",
    "ResolvedRecordAsset", "SpendCheckpoint", "SqliteRecordInventoryStore", "TokenJoinRouter", "build_inventory_plan",
    "decode_record", "held_record_ids", "holds_records",
    "inventory_output_commitments", "inventory_record_id", "managed_inventory", "parse_program_interface",
    "prepared_from_payload", "prepared_from_result", "record_input_ids", "scope_for", "transaction_fee",
    "validate_asset", "validate_record_join_program", "validate_token_join_router",
]
