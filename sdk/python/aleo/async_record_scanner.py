# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Asynchronous RecordScanner (httpx-based) for delegated record scanning."""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import urlparse

from ._client_common import jwt_expired
from ._scanner_common import (
    DecryptionNotEnabledError,
    OwnedFilter,
    OwnedRecord,
    RecordNotFoundError,
    RecordScannerRequestError,
    RecordsFilter,
    UUIDError,
    ViewKeyNotStoredError,
    net_module,
    compute_uuid,
    normalize_api_key,
    uuid_is_valid,
)
from .security import encrypt_registration_request


class AsyncRecordScanner:
    """Asynchronous record scanner client (httpx-based).

    Parameters
    ----------
    url:
        Base URL for the record scanner service (without network suffix).
        Must NOT end with ``/mainnet`` or ``/testnet``.
    network:
        Network name (default ``"mainnet"``).
    api_key:
        API key as a string, ``(header, value)`` tuple, or
        ``{"header": ..., "value": ...}`` dict.
    consumer_id:
        Consumer ID paired with *api_key* for JWT refresh.
    jwt_data:
        Pre-populated JWT dict ``{"jwt": str, "expiration": int (ms)}``.
    view_keys:
        Initial list of ViewKey objects to store.
    account:
        Account whose view key is used for UUID and scanning.
    cache_view_keys_on_register:
        If True, store the view key when registering (default True).
    auto_re_register:
        If True, re-register on 422 responses and retry once.
    decrypt_enabled:
        If True, decrypt records in-place for ``owned()`` calls.
    transport:
        Optional ``httpx.AsyncBaseTransport`` instance passed to
        ``httpx.AsyncClient(transport=...)``.
    """

    def __init__(
        self,
        url: str,
        *,
        network: str = "mainnet",
        api_key: str | tuple[str, str] | dict[str, str] | None = None,
        consumer_id: str | None = None,
        jwt_data: dict[str, Any] | None = None,
        view_keys: list[Any] | None = None,
        account: Any | None = None,
        cache_view_keys_on_register: bool = True,
        auto_re_register: bool = False,
        decrypt_enabled: bool = False,
        transport: Any = None,
    ) -> None:
        try:
            import httpx  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "httpx is required for AsyncRecordScanner. "
                "Install with: pip install aleo[async]"
            ) from None

        # Validate: URL must not end with a network suffix
        stripped = url.rstrip("/")
        if stripped.endswith("/mainnet") or stripped.endswith("/testnet"):
            raise ValueError(
                "The record scanning url should not include the specific network. "
                f"Remove '/{stripped.rsplit('/', 1)[-1]}' from the url."
            )

        self.url: str = f"{url}/{network}"
        self._origin: str = urlparse(url).scheme + "://" + urlparse(url).netloc
        self._network: str = network

        self._api_key: dict[str, str] | None = normalize_api_key(api_key)
        self.consumer_id: str | None = consumer_id
        self.jwt_data: dict[str, Any] | None = jwt_data
        self.cache_view_keys_on_register: bool = cache_view_keys_on_register
        self.auto_re_register: bool = auto_re_register
        self.decrypt_enabled: bool = decrypt_enabled

        # UUID and view key storage
        self._uuid: Any | None = None          # Field | None
        self._view_keys: dict[str, Any] = {}   # uuid_str -> ViewKey
        self._account: Any | None = None

        # Build httpx client
        if transport is not None:
            self._client: Any = httpx.AsyncClient(transport=transport)
        else:
            self._client = httpx.AsyncClient()

        # Load initial view keys
        if view_keys:
            for vk in view_keys:
                self.add_view_key(vk)

        # Load account (mirrors TS: set, setUuid, addViewKey)
        if account is not None:
            self._account = account
            self.set_uuid(account.view_key)
            self.add_view_key(account.view_key)

    # ── Mutators ─────────────────────────────────────────────────────────

    def set_api_key(self, api_key: str | tuple[str, str] | dict[str, str]) -> None:
        self._api_key = normalize_api_key(api_key)

    def set_consumer_id(self, consumer_id: str) -> None:
        self.consumer_id = consumer_id

    def set_jwt_data(self, jwt_data: dict[str, Any] | None) -> None:
        self.jwt_data = jwt_data

    def set_auto_re_register(self, enabled: bool) -> None:
        self.auto_re_register = enabled

    def set_decrypt_enabled(self, enabled: bool) -> None:
        self.decrypt_enabled = enabled

    def add_view_key(self, view_key: Any) -> None:
        """Store view_key keyed by its computed UUID string."""
        uuid_field = compute_uuid(view_key, self._network)
        self._view_keys[str(uuid_field)] = view_key

    def remove_view_key(self, uuid: str) -> None:
        self._view_keys.pop(uuid, None)

    def set_uuid(self, key_material: Any) -> None:
        """Set UUID from a Field or ViewKey."""
        Field = net_module(self._network).Field
        if isinstance(key_material, Field):
            self._uuid = key_material
        else:
            self._uuid = compute_uuid(key_material, self._network)

    def set_account(self, account: Any) -> None:
        """Set account: mirrors TS (set, setUuid, addViewKey, remove old uuid)."""
        old_uuid: str | None = None
        if self._account is not None:
            try:
                old_uuid = str(compute_uuid(self._account.view_key, self._network))
            except Exception:
                pass

        self._account = account
        self.set_uuid(account.view_key)
        self.add_view_key(account.view_key)

        if old_uuid and old_uuid != str(self._uuid):
            self.remove_view_key(old_uuid)

    # ── Internal HTTP ─────────────────────────────────────────────────────

    async def _build_headers(self) -> dict[str, str]:
        """Build authentication headers, refreshing JWT if needed."""
        hdrs: dict[str, str] = {"Content-Type": "application/json"}

        if self._api_key:
            hdrs[self._api_key["header"]] = self._api_key["value"]

        needs_refresh = not self.jwt_data or jwt_expired(self.jwt_data)
        if needs_refresh:
            if self._api_key and self.consumer_id:
                jwt_url = f"{self._origin}/jwts/{self.consumer_id}"
                jwt_hdrs = {self._api_key["header"]: self._api_key["value"]}
                resp = await self._client.post(jwt_url, headers=jwt_hdrs)
                if resp.is_success:
                    auth = resp.headers.get("Authorization") or resp.headers.get("authorization")
                    body = resp.json()
                    if auth:
                        self.jwt_data = {"jwt": auth, "expiration": body["exp"] * 1000}

        if self.jwt_data and self.jwt_data.get("jwt"):
            hdrs["Authorization"] = self.jwt_data["jwt"]

        return hdrs

    async def _get_json(self, url: str) -> Any:
        hdrs = await self._build_headers()
        resp = await self._client.get(url, headers=hdrs)
        if not resp.is_success:
            raise RecordScannerRequestError(resp.text, resp.status_code)
        return resp.json()

    async def _post_json(self, url: str, body: str) -> Any:
        hdrs = await self._build_headers()
        resp = await self._client.post(url, content=body.encode(), headers=hdrs)
        if not resp.is_success:
            raise RecordScannerRequestError(resp.text, resp.status_code)
        return resp

    # ── UUID helpers ─────────────────────────────────────────────────────

    def _resolve_and_validate_uuid(self, uuid_param: str | None = None) -> str:
        candidate: str | None = uuid_param
        if candidate is None and self._uuid is not None:
            candidate = str(self._uuid)
        if candidate is None:
            raise UUIDError("No UUID configured. Call register() or set_uuid() first.")
        if not uuid_is_valid(candidate, self._network):
            raise UUIDError(
                f"UUID '{candidate}' is invalid (not a valid field string).",
                uuid=candidate,
            )
        return candidate

    # ── Registration ─────────────────────────────────────────────────────

    async def register_encrypted(
        self,
        view_key: Any,
        start_block: int = 0,
    ) -> dict[str, Any]:
        """Register a view key for scanning using sealed-box encryption."""
        try:
            pubkey_data = await self._get_json(f"{self.url}/pubkey")
            key_id: str = pubkey_data["key_id"]
            public_key: str = pubkey_data["public_key"]

            ciphertext = encrypt_registration_request(public_key, view_key, start_block)
            payload = json.dumps({"key_id": key_id, "ciphertext": ciphertext})

            resp = await self._post_json(f"{self.url}/register/encrypted", payload)
            data: dict[str, Any] = resp.json()

            if self._uuid is None:
                Field = net_module(self._network).Field
                try:
                    self._uuid = Field.from_string(data["uuid"])
                except Exception:
                    pass

            if self.cache_view_keys_on_register:
                self.add_view_key(view_key)

            return {"ok": True, "data": data}

        except RecordScannerRequestError as exc:
            return {
                "ok": False,
                "status": exc.status,
                "error": {"message": str(exc), "status": exc.status},
            }

    async def register(self, view_key: Any, start_block: int = 0) -> dict[str, Any]:
        """Alias for register_encrypted."""
        return await self.register_encrypted(view_key, start_block)

    # ── Revoke ───────────────────────────────────────────────────────────

    async def revoke(self, uuid: str | None = None) -> dict[str, Any]:
        """Revoke a registration."""
        resolved = self._resolve_and_validate_uuid(uuid)
        body = json.dumps(resolved)
        await self._post_json(f"{self.url}/revoke", body)

        if self._uuid is not None and str(self._uuid) == resolved:
            self._uuid = None
        self._view_keys.pop(resolved, None)
        if self._account is not None:
            try:
                if str(compute_uuid(self._account.view_key, self._network)) == resolved:
                    self._account = None
            except Exception:
                pass

        return {"ok": True, "data": {"status": "OK"}}

    # ── Status ───────────────────────────────────────────────────────────

    async def status(self, uuid: str | None = None) -> dict[str, Any]:
        """Get scanning status for a UUID."""
        resolved = self._resolve_and_validate_uuid(uuid)
        body = json.dumps(resolved)
        resp = await self._post_json(f"{self.url}/status", body)
        return {"ok": True, "data": resp.json()}

    # ── Owned records ─────────────────────────────────────────────────────

    async def owned(self, filter: OwnedFilter) -> dict[str, Any]:
        """Fetch owned records matching the given filter."""
        filter_uuid = filter.get("uuid", "")
        if filter_uuid and uuid_is_valid(filter_uuid, self._network):
            resolved_uuid = filter_uuid
        elif self._uuid is not None:
            resolved_uuid = str(self._uuid)
        else:
            raise UUIDError(
                "No UUID in filter or configured. Call register() or set_uuid() first.",
                filter=dict(filter),
            )

        filter["uuid"] = resolved_uuid
        body = json.dumps(filter)

        hdrs = await self._build_headers()
        resp = await self._client.post(
            f"{self.url}/records/owned", content=body.encode(), headers=hdrs
        )

        if resp.status_code == 422 and self.auto_re_register:
            vk = self._view_keys.get(resolved_uuid)
            if vk is not None:
                await self.register_encrypted(vk, 0)
                hdrs = await self._build_headers()
                resp = await self._client.post(
                    f"{self.url}/records/owned", content=body.encode(), headers=hdrs
                )

        if not resp.is_success:
            return {
                "ok": False,
                "status": resp.status_code,
                "error": {"message": resp.text, "status": resp.status_code},
            }

        records: list[OwnedRecord] = resp.json()

        if self.decrypt_enabled:
            vk = self._view_keys.get(resolved_uuid)
            if vk is not None:
                self.decrypt(vk, records)

        return {"ok": True, "data": records}

    # ── Encrypted records ─────────────────────────────────────────────────

    async def encrypted(self, records_filter: RecordsFilter) -> dict[str, Any]:
        """Fetch encrypted records matching the given filter."""
        body = json.dumps(records_filter)
        try:
            resp = await self._post_json(f"{self.url}/records/encrypted", body)
            return {"ok": True, "data": resp.json()}
        except RecordScannerRequestError as exc:
            return {
                "ok": False,
                "status": exc.status,
                "error": {"message": str(exc), "status": exc.status},
            }

    async def encrypted_records(self, records_filter: RecordsFilter) -> list[Any]:
        """Throwing wrapper around encrypted()."""
        result = await self.encrypted(records_filter)
        if result["ok"]:
            return result["data"]  # type: ignore[return-value]
        err = result.get("error", {})
        raise RecordScannerRequestError(
            err.get("message", "encrypted_records failed"),
            result.get("status", 500),
        )

    # ── Serial numbers & tags ─────────────────────────────────────────────

    async def check_serial_numbers(
        self, serial_numbers: list[str]
    ) -> dict[str, Any]:
        """Check whether serial numbers are spent."""
        body = json.dumps(serial_numbers)
        try:
            resp = await self._post_json(f"{self.url}/records/sns", body)
            return {"ok": True, "data": resp.json()}
        except RecordScannerRequestError as exc:
            return {
                "ok": False,
                "status": exc.status,
                "error": {"message": str(exc), "status": exc.status},
            }

    async def serial_numbers(self, serial_numbers: list[str]) -> dict[str, Any]:
        """Alias for check_serial_numbers."""
        return await self.check_serial_numbers(serial_numbers)

    async def check_tags(self, tags: list[str]) -> dict[str, Any]:
        """Check whether tags are spent."""
        body = json.dumps(tags)
        try:
            resp = await self._post_json(f"{self.url}/records/tags", body)
            return {"ok": True, "data": resp.json()}
        except RecordScannerRequestError as exc:
            return {
                "ok": False,
                "status": exc.status,
                "error": {"message": str(exc), "status": exc.status},
            }

    async def tags(self, tags: list[str]) -> dict[str, Any]:
        """Alias for check_tags."""
        return await self.check_tags(tags)

    # ── Convenience ───────────────────────────────────────────────────────

    def decrypt(self, view_key: Any, records: list[Any]) -> None:
        """Decrypt record_ciphertext fields in-place."""
        RecordCiphertext = net_module(self._network).RecordCiphertext
        for record in records:
            ct = record.get("record_ciphertext", "").strip()
            if not ct:
                continue
            try:
                plaintext = RecordCiphertext.from_string(ct).decrypt(view_key)
                record["record_plaintext"] = str(plaintext)
            except Exception:
                pass

    async def find_records(self, filter: OwnedFilter) -> list[OwnedRecord]:
        """Call owned() and return the data list, or raise."""
        result = await self.owned(filter)
        if result["ok"]:
            return result["data"]  # type: ignore[return-value]
        err = result.get("error", {})
        raise RecordScannerRequestError(
            err.get("message", "find_records failed"),
            result.get("status", 500),
        )

    async def find_record(self, filter: OwnedFilter) -> OwnedRecord:
        """Return the first record from find_records(), or raise RecordNotFoundError."""
        records = await self.find_records(filter)
        if not records:
            raise RecordNotFoundError("No matching record found.")
        return records[0]

    async def find_credits_record(
        self, microcredits: int, filter: OwnedFilter
    ) -> OwnedRecord:
        """Find first credits.aleo record with >= microcredits."""
        filter_uuid = filter.get("uuid", "")
        if filter_uuid and uuid_is_valid(filter_uuid, self._network):
            resolved_uuid = filter_uuid
        elif self._uuid is not None:
            resolved_uuid = str(self._uuid)
        else:
            raise UUIDError("No UUID configured.")

        if not self.decrypt_enabled:
            raise DecryptionNotEnabledError(
                "decrypt_enabled must be True to use find_credits_record."
            )

        vk = self._view_keys.get(resolved_uuid)
        if vk is None:
            raise ViewKeyNotStoredError(
                f"No view key stored for uuid '{resolved_uuid}'.",
                uuid=resolved_uuid,
            )

        credits_filter: OwnedFilter = {
            "unspent": filter.get("unspent", True),  # type: ignore[typeddict-item]
            "filter": {  # type: ignore[typeddict-item]
                "program": "credits.aleo",
                "record": "credits",
                **filter.get("filter", {}),  # type: ignore[arg-type]
            },
            "uuid": resolved_uuid,
        }
        if "responseFilter" in filter:
            credits_filter["responseFilter"] = filter["responseFilter"]  # type: ignore[typeddict-item]

        records = await self.find_records(credits_filter)

        RecordPlaintext = net_module(self._network).RecordPlaintext
        for record in records:
            pt_str = record.get("record_plaintext", "")
            if not pt_str:
                continue
            try:
                pt = RecordPlaintext.from_string(pt_str)
                if pt.microcredits >= microcredits:
                    return record
            except Exception:
                pass

        raise RecordNotFoundError(
            f"No credits record found with >= {microcredits} microcredits."
        )

    async def find_credits_records(
        self, microcredit_amounts: list[int], filter: OwnedFilter
    ) -> list[OwnedRecord]:
        """Find credits.aleo records whose microcredits are in microcredit_amounts.

        Requires decrypt_enabled=True and a stored view key for the UUID.
        """
        # Resolve UUID
        filter_uuid = filter.get("uuid", "")
        if filter_uuid and uuid_is_valid(filter_uuid, self._network):
            resolved_uuid = filter_uuid
        elif self._uuid is not None:
            resolved_uuid = str(self._uuid)
        else:
            raise UUIDError("No UUID configured.")

        if not self.decrypt_enabled:
            raise DecryptionNotEnabledError(
                "decrypt_enabled must be True to use find_credits_records."
            )

        vk = self._view_keys.get(resolved_uuid)
        if vk is None:
            raise ViewKeyNotStoredError(
                f"No view key stored for uuid '{resolved_uuid}'.",
                uuid=resolved_uuid,
            )

        credits_filter: OwnedFilter = {
            "unspent": filter.get("unspent", True),  # type: ignore[typeddict-item]
            "filter": {  # type: ignore[typeddict-item]
                "program": "credits.aleo",
                "record": "credits",
                **filter.get("filter", {}),  # type: ignore[arg-type]
            },
            "uuid": resolved_uuid,
        }
        if "responseFilter" in filter:
            credits_filter["responseFilter"] = filter["responseFilter"]  # type: ignore[typeddict-item]

        records = await self.find_records(credits_filter)

        RecordPlaintext = net_module(self._network).RecordPlaintext
        amounts_set = set(microcredit_amounts)
        result: list[OwnedRecord] = []
        for record in records:
            pt_str = record.get("record_plaintext", "")
            if not pt_str:
                continue
            try:
                pt = RecordPlaintext.from_string(pt_str)
                if pt.microcredits in amounts_set:
                    result.append(record)
            except Exception:
                pass

        return result
