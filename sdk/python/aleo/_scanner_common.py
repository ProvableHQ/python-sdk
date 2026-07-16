"""Shared error types, models, pure helpers for RecordScanner."""
from __future__ import annotations

from typing import Any, TypedDict

from ._client_common import AleoError


# ---------------------------------------------------------------------------
# Error types (all subclass AleoError so `except AleoError` catches them)
# ---------------------------------------------------------------------------

class RecordScannerRequestError(AleoError):
    """Raised when a RecordScanner HTTP request returns a non-2xx status."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class DecryptionNotEnabledError(AleoError):
    """Raised when decryption is required but decrypt_enabled=False."""


class ViewKeyNotStoredError(AleoError):
    """Raised when a view key is needed but not stored for a given UUID."""

    def __init__(self, message: str, uuid: str | None = None) -> None:
        super().__init__(message)
        self.uuid = uuid


class RecordNotFoundError(AleoError):
    """Raised when no matching record is found."""


class UUIDError(AleoError):
    """Raised for UUID resolution or validation failures."""

    def __init__(
        self,
        message: str,
        uuid: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.uuid = uuid
        self.filter = filter


# ---------------------------------------------------------------------------
# TypedDict models
# ---------------------------------------------------------------------------

class RecordsResponseFilter(TypedDict, total=False):
    block_height: bool
    block_timestamp: bool
    checksum: bool
    commitment: bool
    record_ciphertext: bool
    sender_ciphertext: bool
    function_name: bool
    nonce: bool
    output_index: bool
    owner: bool
    program_name: bool
    record_name: bool
    transaction_id: bool
    transition_id: bool
    transaction_index: bool
    transition_index: bool


class RecordsFilter(TypedDict, total=False):
    commitments: list[str]
    response: RecordsResponseFilter
    start: int
    end: int
    programs: list[str]
    records: list[str]
    functions: list[str]
    results_per_page: int
    page: int
    spent: bool
    program: str
    record: str


class OwnedRecordsResponseFilter(TypedDict, total=False):
    commitment: bool
    owner: bool
    tag: bool
    sender: bool
    spent: bool
    record_ciphertext: bool
    block_height: bool
    block_timestamp: bool
    output_index: bool
    record_name: bool
    function_name: bool
    program_name: bool
    transition_id: bool
    transaction_id: bool
    transaction_index: bool
    transition_index: bool


class OwnedFilter(TypedDict, total=False):
    uuid: str
    unspent: bool
    decrypt: bool
    filter: RecordsFilter
    responseFilter: OwnedRecordsResponseFilter
    nonces: list[str]


class OwnedRecord(TypedDict, total=False):
    block_height: int
    block_timestamp: int
    commitment: str
    function_name: str
    output_index: int
    owner: str
    program_name: str
    record_ciphertext: str
    record_plaintext: str
    record_name: str
    sender: str
    spent: bool
    tag: str
    transaction_id: str
    transition_id: str
    transaction_index: int
    transition_index: int


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def net_module(network: str = "mainnet") -> Any:
    """Return the compiled network module ('mainnet' or 'testnet').

    The type classes (Field, Poseidon4, RecordCiphertext, RecordPlaintext)
    come from distinct compiled extensions per network and are NOT
    interchangeable, so callers must select the module matching the
    view-key/record network.
    """
    if network == "testnet":
        from . import testnet as _mod  # type: ignore[attr-defined]
    else:
        from . import mainnet as _mod  # type: ignore[attr-defined]
    return _mod


def compute_uuid(view_key: Any, network: str = "mainnet") -> Any:
    """Compute the RecordScanner UUID from a ViewKey using Poseidon4.

    Returns a Field with domain separator "RecordScannerV0".
    This is the golden KAT:
        APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH
        -> 7884164224800444110633570141944665301008802280502652120359195870264061098703field
    """
    _mod = net_module(network)
    Field, Poseidon4 = _mod.Field, _mod.Poseidon4
    domain_sep = Field.domain_separator("RecordScannerV0")
    vk_field = view_key.to_field()
    one = Field.one()
    hasher = Poseidon4()
    return hasher.hash([domain_sep, vk_field, one])


def build_owned_filter(
    uuid: str | None,
    *,
    program: str | None = None,
    record: str | None = None,
    unspent: bool = True,
    nonces: list[str] | None = None,
) -> OwnedFilter:
    """Build an :class:`OwnedFilter` from the common record-query parameters.

    Shared by the sync and async facade record modules so the filter shape is
    defined in exactly one place.
    """
    record_filter: RecordsFilter = {}
    if program is not None:
        record_filter["program"] = program
    if record is not None:
        record_filter["record"] = record

    owned: OwnedFilter = {"unspent": unspent}
    if uuid is not None:
        owned["uuid"] = uuid
    if record_filter:
        owned["filter"] = record_filter
    if nonces is not None:
        owned["nonces"] = nonces
    return owned


def enforce_record_filter(
    records: list[OwnedRecord],
    *,
    program: str | None = None,
    record: str | None = None,
) -> list[OwnedRecord]:
    """Apply the ``program``/``record`` subfilter to scanned records locally.

    The hosted scanner ignores the ``filter`` subobject of an
    :class:`OwnedFilter` and returns every record the account owns, so the
    filter contract must be enforced on the results client-side.
    """
    if program is not None:
        records = [r for r in records if r.get("program_name") == program]
    if record is not None:
        records = [r for r in records if r.get("record_name") == record]
    return records


def uuid_is_valid(uuid: str, network: str = "mainnet") -> bool:
    """Return True if uuid is a valid Field string (e.g. '1234...field')."""
    try:
        Field = net_module(network).Field
        Field.from_string(uuid)
        return True
    except Exception:
        return False


def normalize_api_key(
    api_key: str | tuple[str, str] | dict[str, str] | None,
) -> dict[str, str] | None:
    """Normalize api_key to {'header': str, 'value': str} or None."""
    if api_key is None:
        return None
    if isinstance(api_key, str):
        return {"header": "X-Provable-API-Key", "value": api_key}
    if isinstance(api_key, tuple):
        header, value = api_key
        return {"header": header, "value": value}
    # dict
    return dict(api_key)
