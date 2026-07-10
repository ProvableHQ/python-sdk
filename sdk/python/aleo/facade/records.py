"""Aleo — facade records module (F6).

Attached to the :class:`~aleo.facade.client.Aleo` client as ``aleo.records``.
Wraps the delegated :class:`~aleo.record_scanner.RecordScanner` behind a clean,
Web3.py-style interface and implements the
:class:`~aleo._facade_common.RecordProvider` protocol so the F5 verb ladder can
auto-source a credits record for a private fee.

.. warning::

    **Delegated scanning shares your view key.**  Aleo records are encrypted;
    finding *your* records normally means scanning the whole chain.  The default
    scanner is a *hosted* service, so :meth:`RecordsModule.register` sends your
    account's **view key** (sealed-box encrypted in transit) to that service —
    which can then decrypt every record you own.  This is a real privacy
    tradeoff.  If you do not want to share your view key with a hosted scanner,
    point ``aleo.records.scanner`` at a **self-hosted** endpoint, or assign your
    own :class:`~aleo._facade_common.RecordProvider` implementation to
    ``aleo.record_provider``.
"""
from __future__ import annotations

from typing import Any

from .._scanner_common import OwnedRecord, RecordNotFoundError


class RecordsModule:
    """Namespaced record operations attached to an :class:`~aleo.facade.client.Aleo` client.

    Access via ``aleo.records``, not by direct construction.  Implements the
    :class:`~aleo._facade_common.RecordProvider` protocol (``get_unspent`` +
    ``find``) over a lazily-built :class:`~aleo.record_scanner.RecordScanner`.

    .. warning::

        This module talks to a **delegated record scanner**.  Registering an
        account (:meth:`register`) shares that account's **view key** with the
        scanning service, which can then decrypt every record the account owns.
        For a self-custodial alternative, repoint :attr:`scanner` at a
        self-hosted endpoint or assign a custom
        :class:`~aleo._facade_common.RecordProvider` to ``aleo.record_provider``.

    Parameters
    ----------
    client:
        The parent :class:`~aleo.facade.client.Aleo` instance.
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._scanner: Any = None  # lazily built from provider config
        self._account: Any = None  # last account passed to register()

    def __repr__(self) -> str:
        return f"RecordsModule(network={self._client._provider.network!r})"

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _net(self) -> Any:
        """Return the network module (``aleo.mainnet`` or ``aleo.testnet``)."""
        network: str = self._client._provider.network
        if network == "testnet":
            import aleo.testnet as _testnet  # type: ignore[attr-defined]
            return _testnet
        import aleo.mainnet as _mainnet
        return _mainnet

    def _build_scanner(self) -> Any:
        """Construct a :class:`~aleo.record_scanner.RecordScanner` from provider config.

        The scanner base URL, api key, network and transport are read from the
        client's :class:`~aleo.facade.provider.HTTPProvider`.  The provider does
        not expose a dedicated scanner URI, so the versioned API root is used as
        the scanner base (with any trailing network suffix stripped, which the
        scanner constructor requires).  Point :attr:`scanner` elsewhere to use a
        self-hosted scanning endpoint.
        """
        from ..record_scanner import RecordScanner

        provider = self._client._provider
        base = str(provider.url).rstrip("/")
        for suffix in ("/mainnet", "/testnet"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return RecordScanner(
            base,
            network=provider.network,
            api_key=provider.api_key,
            transport=getattr(provider, "_transport", None),
        )

    @property
    def scanner(self) -> Any:
        """The underlying :class:`~aleo.record_scanner.RecordScanner` (escape hatch).

        Built lazily on first access from the client's provider config.  Assign a
        pre-configured scanner (e.g. one pointed at a self-hosted endpoint) to
        override the default hosted service and keep your view key private.
        """
        if self._scanner is None:
            self._scanner = self._build_scanner()
        return self._scanner

    @scanner.setter
    def scanner(self, scanner: Any) -> None:
        self._scanner = scanner

    # ── Registration / lifecycle ─────────────────────────────────────────────

    def register(self, account: Any, start: int = 0) -> dict[str, Any]:
        """Register *account* for delegated scanning from block *start*.

        Sets the account on the scanner, enables in-place decryption, and calls
        :meth:`~aleo.record_scanner.RecordScanner.register` with the account's
        view key.

        .. warning::

            **This shares your view key.**  Registration transmits *account*'s
            **view key** (sealed-box encrypted in transit) to the scanning
            service so it can decrypt the records you own on your behalf.  The
            service can therefore see every record belonging to *account*.  If
            that is not acceptable, repoint :attr:`scanner` at a self-hosted
            endpoint before calling :meth:`register`, or supply your own
            :class:`~aleo._facade_common.RecordProvider` via
            ``aleo.record_provider``.

        Parameters
        ----------
        account:
            The :class:`Account` whose records should be scanned.
        start:
            First block height to scan from (default ``0``).

        Returns
        -------
        dict
            The scanner's register result (``{"ok": bool, "data"/"error": ...}``).
        """
        scanner = self.scanner
        self._account = account
        scanner.set_account(account)
        scanner.set_decrypt_enabled(True)
        return scanner.register(account.view_key, start)

    def revoke(self) -> dict[str, Any]:
        """Revoke the registered account's scanning registration.

        Delegates to :meth:`~aleo.record_scanner.RecordScanner.revoke` using the
        scanner's configured UUID (derived from the registered account's view key).
        """
        return self.scanner.revoke()

    def status(self) -> dict[str, Any]:
        """Return the scanning status for the registered account.

        Delegates to :meth:`~aleo.record_scanner.RecordScanner.status`.
        """
        return self.scanner.status()

    # ── Queries ──────────────────────────────────────────────────────────────

    def find(
        self,
        account: Any = None,
        *,
        program: str | None = None,
        record: str | None = None,
        unspent: bool = True,
        amounts: list[int] | None = None,
        nonces: list[str] | None = None,
        **_extra: Any,
    ) -> list[OwnedRecord]:
        """Find owned records for *account* matching the given filters.

        Builds an :class:`~aleo._scanner_common.OwnedFilter` (uuid derived from
        the account's view key, ``unspent`` flag, optional ``program``/``record``
        subfilter, optional ``amounts``/``nonces``) and calls
        :meth:`~aleo.record_scanner.RecordScanner.find_records`.

        Parameters
        ----------
        account:
            The account whose records to find.  Defaults to the account passed to
            :meth:`register`.
        program:
            Restrict to records emitted by this program (e.g. ``"credits.aleo"``).
        record:
            Restrict to this record type (e.g. ``"credits"``).
        unspent:
            Only return unspent records (default ``True``).
        amounts:
            Optional list of microcredit amounts to match (credits records).
        nonces:
            Optional list of record ``_nonce`` strings to match.

        Returns
        -------
        list[OwnedRecord]
            The scanner's matching records.
        """
        acct = account if account is not None else self._account
        scanner = self.scanner
        if acct is not None:
            scanner.set_account(acct)
            scanner.set_decrypt_enabled(True)

        from .._scanner_common import build_owned_filter, compute_uuid

        uuid = str(compute_uuid(acct.view_key)) if acct is not None else None
        owned_filter = build_owned_filter(
            uuid, program=program, record=record, unspent=unspent, nonces=nonces
        )

        if amounts is not None:
            return scanner.find_credits_records(amounts, owned_filter)
        return scanner.find_records(owned_filter)

    def find_credits(
        self, account: Any = None, at_least: int | None = None
    ) -> list[OwnedRecord]:
        """Find unspent ``credits.aleo``/``credits`` records for *account*.

        When *at_least* is given, only records covering that many microcredits are
        returned (the first such record, wrapped in a list); otherwise all
        unspent credits records are returned.

        Parameters
        ----------
        account:
            The account whose credits records to find.  Defaults to the account
            passed to :meth:`register`.
        at_least:
            Minimum microcredits a returned record must cover.

        Returns
        -------
        list[OwnedRecord]
            Matching credits records (``record_plaintext`` populated).
        """
        acct = account if account is not None else self._account
        scanner = self.scanner
        if acct is not None:
            scanner.set_account(acct)
            scanner.set_decrypt_enabled(True)

        from .._scanner_common import build_owned_filter, compute_uuid

        uuid = str(compute_uuid(acct.view_key)) if acct is not None else None

        if at_least is not None:
            try:
                rec = scanner.find_credits_record(
                    int(at_least), build_owned_filter(uuid)
                )
            except RecordNotFoundError:
                return []
            return [rec]

        owned_filter = build_owned_filter(
            uuid, program="credits.aleo", record="credits"
        )
        return scanner.find_records(owned_filter)

    # ── RecordProvider protocol ────────────────────────────────────────────────

    def get_unspent(
        self,
        *,
        program: str,
        record: str,
        min_microcredits: int | None = None,
        exclude_nonces: tuple[str, ...] = (),
    ) -> Any | None:
        """Return one unspent record as a network ``RecordPlaintext`` (or ``None``).

        This is the :class:`~aleo._facade_common.RecordProvider` method the F5 fee
        ladder calls to auto-source a private fee record.  It finds the first
        unspent ``program``/``record`` covering *min_microcredits* (for credits
        records), skips any whose ``_nonce`` is in *exclude_nonces*, parses the
        scanner's ``record_plaintext`` string into a network ``RecordPlaintext``
        and returns it — or ``None`` when nothing qualifies.

        Parameters
        ----------
        program:
            The record's program (e.g. ``"credits.aleo"``).
        record:
            The record type (e.g. ``"credits"``).
        min_microcredits:
            Minimum microcredits the record must cover (credits records only).
        exclude_nonces:
            Record ``_nonce`` strings to skip (e.g. records already spent in this
            transaction batch).
        """
        net = self._net()

        # Gather candidate OwnedRecords.
        if program == "credits.aleo" and record == "credits":
            if exclude_nonces:
                # find_credits(at_least=...) returns only the first covering
                # record; if that one is excluded we'd wrongly report None even
                # when other covering records exist.  With exclusions in play,
                # pull the full unspent set and let the loop below apply both
                # the min-microcredits and exclude-nonce filters.
                candidates = self.find_credits(at_least=None)
            else:
                candidates = self.find_credits(at_least=min_microcredits)
        else:
            candidates = self.find(program=program, record=record, unspent=True)

        excluded = set(exclude_nonces)
        for owned in candidates:
            pt_str = owned.get("record_plaintext", "")
            if not pt_str:
                continue
            try:
                plaintext = net.RecordPlaintext.from_string(pt_str)
            except Exception:
                continue
            if excluded and str(plaintext.nonce) in excluded:
                continue
            if (
                min_microcredits is not None
                and program == "credits.aleo"
                and record == "credits"
                and int(plaintext.microcredits) < int(min_microcredits)
            ):
                continue
            return plaintext
        return None


__all__ = ["RecordsModule"]
