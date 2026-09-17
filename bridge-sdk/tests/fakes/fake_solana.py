"""Fake solana-py ``Client`` for SolModule tests.

Mirrors solana-py 0.40's shapes: every method returns an object with ``.value``;
``get_latest_blockhash().value`` has ``blockhash``/``last_valid_block_height``;
``get_account_info().value`` is ``None`` or has ``.data: bytes``; signature statuses are a
list with ``err``/``confirmation_status``; ``get_transaction().value.transaction.meta.log_messages``.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from solders.hash import Hash
from solders.signature import Signature

from aleo_bridge.registry import DEFAULT_REGISTRY
from tests.fakes.sealevel_fixtures import (
    DISPATCHED_MESSAGE_RENT_LAMPORTS,
    FEE_PAYER_RENT_LAMPORTS,
    GAS_PAYMENT_RENT_LAMPORTS,
    IGP,
    NETWORK_FEE_LAMPORTS,
    TRANSFER,
    WARP_PROGRAM_ADDRESS,
    igp_account_data,
)

# Any 32-byte base58 string is a valid blockhash for compile/sign purposes (veil uses the warp program id).
BLOCKHASH = Hash.from_string(WARP_PROGRAM_ADDRESS)
LAST_VALID_BLOCK_HEIGHT = 100
RENTS = {141: GAS_PAYMENT_RENT_LAMPORTS, 194: DISPATCHED_MESSAGE_RENT_LAMPORTS, 0: FEE_PAYER_RENT_LAMPORTS}
STUB_SIGNATURE = Signature.from_bytes(bytes([7]) * 64)


class _Resp:
    def __init__(self, value: Any) -> None:
        self.value = value


@dataclass
class FakeSignatureStatus:
    err: Any = None
    confirmation_status: str | None = "confirmed"


@dataclass
class _Blockhash:
    blockhash: Hash
    last_valid_block_height: int


@dataclass
class _Account:
    data: bytes


class FakeSolanaClient:
    """``statuses`` is consumed one entry per ``get_signature_statuses`` call (the last entry repeats);
    an entry may be ``None`` (unknown signature), a ``FakeSignatureStatus``, or an ``Exception`` to raise."""

    def __init__(self, *, balance: int = 800_000_000_000, accounts: dict[str, bytes] | None = None,
                 fee: int = NETWORK_FEE_LAMPORTS, rents: dict[int, int] | None = None,
                 statuses: list[Any] | None = None, blockhash_valid: Any = True,
                 logs: list[str] | None = None, no_logs: bool = False,
                 signature: Signature = STUB_SIGNATURE) -> None:
        self.balance = balance
        self.accounts = {IGP["address"]: igp_account_data()} if accounts is None else accounts
        self.fee = fee
        self.rents = RENTS if rents is None else rents
        self.statuses = list(statuses) if statuses is not None else [FakeSignatureStatus()]
        self.blockhash_valid = blockhash_valid
        # logs=None → the recorded mainnet logs; logs=[] → confirmed but no dispatch line; no_logs → transaction not found
        self.logs = None if no_logs else (list(TRANSFER["logMessages"]) if logs is None else list(logs))
        self.signature = signature
        self.calls: list[str] = []
        self.fee_messages: list[Any] = []
        self.sent: list[bytes] = []
        self.sent_opts: list[Any] = []
        self.status_calls: list[bool] = []
        self.transaction_calls: list[tuple[Any, Any]] = []

    def get_latest_blockhash(self, commitment=None):
        self.calls.append("get_latest_blockhash")
        return _Resp(_Blockhash(BLOCKHASH, LAST_VALID_BLOCK_HEIGHT))

    def get_balance(self, pubkey, commitment=None):
        self.calls.append("get_balance")
        return _Resp(self.balance)

    def get_account_info(self, pubkey, commitment=None, encoding="base64", data_slice=None):
        self.calls.append("get_account_info")
        data = self.accounts.get(str(pubkey))
        return _Resp(None if data is None else _Account(data))

    def get_fee_for_message(self, message, commitment=None):
        self.calls.append("get_fee_for_message")
        self.fee_messages.append(message)
        return _Resp(self.fee)

    def get_minimum_balance_for_rent_exemption(self, usize, commitment=None):
        self.calls.append("get_minimum_balance_for_rent_exemption")
        return _Resp(self.rents[usize])

    def send_raw_transaction(self, txn, opts=None):
        self.calls.append("send_raw_transaction")
        self.sent.append(bytes(txn))
        self.sent_opts.append(opts)
        return _Resp(self.signature)

    def get_signature_statuses(self, signatures, search_transaction_history=False):
        self.calls.append("get_signature_statuses")
        self.status_calls.append(search_transaction_history)
        item = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if isinstance(item, Exception):
            raise item
        return _Resp([item])

    def is_blockhash_valid(self, blockhash, commitment=None):
        self.calls.append("is_blockhash_valid")
        if isinstance(self.blockhash_valid, Exception):
            raise self.blockhash_valid
        return _Resp(self.blockhash_valid)

    def get_transaction(self, tx_sig, encoding="json", commitment=None, max_supported_transaction_version=None):
        self.calls.append("get_transaction")
        self.transaction_calls.append((commitment, max_supported_transaction_version))
        if self.logs is None:
            return _Resp(None)
        meta = SimpleNamespace(log_messages=list(self.logs))
        return _Resp(SimpleNamespace(transaction=SimpleNamespace(meta=meta)))


def stub_bridge(environment: str = "mainnet") -> SimpleNamespace:
    """What SolModule needs from a Bridge without constructing one: registry + environment."""
    return SimpleNamespace(registry=DEFAULT_REGISTRY, environment=environment)
