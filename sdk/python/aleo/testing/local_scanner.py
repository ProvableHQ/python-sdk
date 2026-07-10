"""Client-side record scanner — the self-custodial :class:`RecordProvider`.

:class:`LocalRecordScanner` finds an account's records by walking a node's
blocks locally and decrypting each candidate ciphertext with the account's
**view key** — the view key never leaves the process.  It is the
self-hosted counterpart to :class:`~aleo.facade.records.RecordsModule`
(which delegates scanning to a hosted service and therefore shares the view
key); it satisfies the same :class:`~aleo._facade_common.RecordProvider`
protocol, so it can be assigned to ``aleo.record_provider`` to auto-source
private fee records without any hosted dependency.

Spend detection is done **via tags** — the on-chain spend marker.  For each
owned record a ``tag`` is derived from the account's :class:`GraphKey` and the
record's commitment; a record is *spent* iff the node can resolve that tag to a
transition (``aleo.network.get_transition_id`` returns instead of raising
:class:`~aleo._client_common.AleoNetworkError`).  This is lighter than serial
numbers and needs no private key.

Example
-------
::

    from aleo.testing import Devnode, LocalRecordScanner

    with Devnode() as dn:
        aleo = dn.aleo
        sender = dn.accounts[0]
        # ... broadcast a transfer_public_to_private, advance, wait ...
        scanner = LocalRecordScanner(aleo, sender)
        rec = scanner.get_unspent(program="credits.aleo", record="credits")
"""
from __future__ import annotations

import json
from typing import Any, cast

from .._client_common import AleoNetworkError


class _Owned:
    """An owned record plus the context needed to filter and spend-check it.

    Attributes
    ----------
    plaintext:
        The decrypted network ``RecordPlaintext``.
    commitment:
        The record's commitment ``Field`` (from ``records()``), needed to
        derive the spend ``tag``.
    program:
        The ``program_id`` of the transition that produced the record
        (e.g. ``"credits.aleo"``), used for the ``program`` filter.
    function:
        The ``function_name`` of the producing transition (advisory).
    """

    __slots__ = ("plaintext", "commitment", "program", "function")

    def __init__(
        self, plaintext: Any, commitment: Any, program: str, function: str
    ) -> None:
        self.plaintext = plaintext
        self.commitment = commitment
        self.program = program
        self.function = function


class LocalRecordScanner:
    """Client-side, view-key-local record finder implementing ``RecordProvider``.

    Parameters
    ----------
    aleo:
        An :class:`~aleo.facade.client.Aleo` client bound to the node to scan.
    account:
        The account whose records to find.  Must expose ``.view_key``.

    Notes
    -----
    A ``RecordPlaintext`` produced by ``Transaction.records()`` carries no
    program/record-type metadata of its own, so this scanner traverses at the
    **transition** level and tags each owned record with the producing
    transition's ``program_id``.  The ``program`` filter is matched against that
    ``program_id``; the ``record`` argument is advisory (for ``credits.aleo`` the
    only record type is ``credits``, which is the case the fee ladder and the
    public/private roundtrip depend on).
    """

    def __init__(self, aleo: Any, account: Any) -> None:
        self._aleo = aleo
        self._account = account

    def __repr__(self) -> str:
        return f"LocalRecordScanner(network={self._aleo._provider.network!r})"

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _net(self) -> Any:
        """Return the network module (``aleo.mainnet`` or ``aleo.testnet``).

        Mirrors the ``_net()`` selection used across the facade modules.
        """
        network: str = self._aleo._provider.network
        if network == "testnet":
            import aleo.testnet as _testnet  # type: ignore[attr-defined]
            return _testnet
        import aleo.mainnet as _mainnet
        return _mainnet

    def _graph_key(self) -> Any:
        net = self._net()
        return net.GraphKey.from_view_key(self._account.view_key)

    @staticmethod
    def _tx_objects(block: Any) -> list[Any]:
        """Pull each transaction JSON object out of a block dict, defensively.

        The block JSON's ``transactions`` is typically a list of
        *confirmed-transaction* dicts, each wrapping the real transaction under
        an inner ``"transaction"`` key.  Some shapes put the transaction object
        directly in the list.  Try ``item["transaction"]`` first, else use the
        item itself.  A missing/oddly-shaped ``transactions`` yields ``[]``.
        """
        if not isinstance(block, dict):
            return []
        txs: Any = cast("dict[str, Any]", block).get("transactions")
        if not isinstance(txs, list):
            return []
        out: list[Any] = []
        for item in cast("list[Any]", txs):
            if isinstance(item, dict) and "transaction" in item:
                out.append(item["transaction"])
            else:
                out.append(item)
        return out

    def _owned_in_transaction(self, net: Any, tx_obj: Any) -> list[_Owned]:
        """Decrypt every owned record in *tx_obj*, keeping producing-program context.

        Per-transaction and per-record work is wrapped in ``try/except`` so
        rejected transactions, deploy transactions, and ciphertexts we do not
        own are silently skipped.
        """
        owned: list[_Owned] = []
        try:
            tx = net.Transaction.from_json(json.dumps(tx_obj))
        except Exception:
            return owned
        vk = self._account.view_key
        # Traverse at the transition level so each record keeps its producing
        # program_id (used for the program filter).  Fall back to tx.records()
        # with an unknown program if transitions are unavailable.
        transitions: list[Any]
        try:
            transitions = list(tx.transitions())
        except Exception:
            transitions = []
        collected: list[tuple[Any, Any, str, str]] = []
        if transitions:
            for tr in transitions:
                try:
                    program = str(tr.program_id)
                except Exception:
                    program = ""
                try:
                    function = str(tr.function_name)
                except Exception:
                    function = ""
                try:
                    records: list[Any] = list(tr.records())
                except Exception:
                    continue
                for pair in records:
                    commitment, ciphertext = pair[0], pair[1]
                    collected.append((ciphertext, commitment, program, function))
        else:
            try:
                fallback: list[Any] = list(tx.records())
            except Exception:
                return owned
            for pair in fallback:
                commitment, ciphertext = pair[0], pair[1]
                collected.append((ciphertext, commitment, "", ""))
        for ciphertext, commitment, program, function in collected:
            try:
                if not ciphertext.is_owner(vk):
                    continue
                plaintext: Any = ciphertext.decrypt(vk)
            except Exception:
                continue
            owned.append(_Owned(plaintext, commitment, program, function))
        return owned

    def _is_spent(self, owned: _Owned, graph_key: Any) -> bool:
        """Return ``True`` iff *owned*'s tag resolves to a transition on-chain.

        The tag is the on-chain spend marker; a resolvable tag means the record
        has been consumed.  A ``404``/not-found surfaces as
        :class:`AleoNetworkError`, which is treated as *unspent*.
        """
        try:
            tag = owned.plaintext.tag(graph_key, owned.commitment)
        except Exception:
            # Cannot derive a tag (unexpected record shape) — treat as unspent
            # so we never hide a record from the caller on a derivation glitch.
            return False
        try:
            self._aleo.network.get_transition_id(str(tag))
            return True
        except AleoNetworkError:
            return False
        except Exception:
            # Any other transport error: conservatively report unspent so a
            # transient failure does not silently drop a usable record.
            return False

    # ── Scanning ─────────────────────────────────────────────────────────────

    def scan(self, start: int = 0, end: int | None = None) -> list[Any]:
        """Return every owned ``RecordPlaintext`` in blocks ``[start, end]``.

        Includes both spent and unspent records.  When *end* is ``None`` the
        current latest height is used.

        Parameters
        ----------
        start:
            First block height to scan (inclusive, default ``0``).
        end:
            Last block height to scan (inclusive).  ``None`` = latest height.

        Returns
        -------
        list
            Owned ``RecordPlaintext`` objects in block order.
        """
        return [o.plaintext for o in self._scan_owned(start, end)]

    def _scan_owned(self, start: int = 0, end: int | None = None) -> list[_Owned]:
        """Scan blocks and return the owned records with their filter context."""
        net = self._net()
        if end is None:
            end = int(self._aleo.network.get_latest_height())
        result: list[_Owned] = []
        for height in range(start, end + 1):
            try:
                block = self._aleo.network.get_block(height)
            except AleoNetworkError:
                continue
            for tx_obj in self._tx_objects(block):
                result.extend(self._owned_in_transaction(net, tx_obj))
        return result

    # ── RecordProvider protocol ──────────────────────────────────────────────

    def find(
        self,
        *,
        program: str | None = None,
        record: str | None = None,
        unspent: bool = True,
        **_extra: Any,
    ) -> list[Any]:
        """Return owned ``RecordPlaintext`` objects matching the filters.

        Parameters
        ----------
        program:
            Restrict to records produced by this program (matched against the
            producing transition's ``program_id``, e.g. ``"credits.aleo"``).
        record:
            Advisory record-type filter (see class notes).  Not enforced beyond
            *program* because the plaintext carries no record-type name.
        unspent:
            When ``True`` (default) drop records whose tag resolves on-chain.

        Returns
        -------
        list
            Matching ``RecordPlaintext`` objects.
        """
        graph_key = self._graph_key() if unspent else None
        out: list[Any] = []
        for owned in self._scan_owned():
            if program is not None and owned.program != program:
                continue
            if unspent and graph_key is not None and self._is_spent(owned, graph_key):
                continue
            out.append(owned.plaintext)
        return out

    def get_unspent(
        self,
        *,
        program: str,
        record: str,
        min_microcredits: int | None = None,
        exclude_nonces: tuple[str, ...] = (),
    ) -> Any | None:
        """Return one unspent ``RecordPlaintext`` for *program*/*record*, or ``None``.

        Scans, drops spent records (tag check) and excluded nonces, requires
        ``microcredits >= min_microcredits`` for credits records, and returns the
        first survivor.

        Parameters
        ----------
        program:
            The record's program (e.g. ``"credits.aleo"``).
        record:
            The record type (advisory; e.g. ``"credits"``).
        min_microcredits:
            Minimum microcredits the record must cover (credits records only).
        exclude_nonces:
            Record nonce strings to skip (records already used this batch).

        Returns
        -------
        RecordPlaintext | None
            The first qualifying unspent record, or ``None``.
        """
        graph_key = self._graph_key()
        excluded = set(exclude_nonces)
        is_credits = program == "credits.aleo" and record == "credits"
        for owned in self._scan_owned():
            if owned.program != program:
                continue
            plaintext = owned.plaintext
            try:
                if str(plaintext.nonce) in excluded:
                    continue
            except Exception:
                pass
            if is_credits and min_microcredits is not None:
                try:
                    if int(plaintext.microcredits) < int(min_microcredits):
                        continue
                except Exception:
                    continue
            if self._is_spent(owned, graph_key):
                continue
            return plaintext
        return None


__all__ = ["LocalRecordScanner"]
