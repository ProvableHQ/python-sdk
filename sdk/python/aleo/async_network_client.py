# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Asynchronous Aleo network client (httpx-based).

Transport contract (async)
--------------------------
When ``transport`` is supplied it must be an instance of
:class:`httpx.AsyncBaseTransport`.  It is passed directly to
``httpx.AsyncClient(transport=...)``, giving the caller full control over the
underlying connection (useful for mocking and custom proxies).

When transport is provided, SDK-internal headers (``X-Aleo-SDK-Version``,
``X-Aleo-environment``, ``X-ALEO-METHOD``) are suppressed so the caller has
full control over the wire format.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

from ._client_common import (
    DEFAULT_HOST,
    DEFAULT_NETWORK,
    AleoNetworkError,
    AleoProvingError,
    is_provable_host,
    jwt_expired,
    jwt_origin,
    make_default_headers,
    method_headers,
    strip_quotes,
    validate_block_range,
    async_retry_with_backoff,
)
from .security import encrypt_proving_request


class AsyncAleoNetworkClient:
    """Asynchronous client for the Aleo REST API (httpx-based).

    Parameters
    ----------
    host:
        Versioned API root, e.g. ``"https://api.provable.com/v2"``.
    network:
        Network name appended to ``host`` for all node endpoints (default
        ``"mainnet"``).
    headers:
        Additional request headers merged on top of the SDK defaults.
    prover_uri:
        Base URI for the DPS prover (without network suffix).
    record_scanner_uri:
        Base URI for the record scanner service.
    transport:
        Optional :class:`httpx.AsyncBaseTransport` instance passed directly
        to ``httpx.AsyncClient``.  SDK-internal telemetry headers are
        suppressed in this mode.
    api_key:
        Provable API key used to refresh JWTs.
    consumer_id:
        Consumer ID paired with *api_key* for JWT refresh.
    jwt_data:
        Pre-populated JWT dict ``{"jwt": str, "expiration": int}``.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        *,
        network: str = DEFAULT_NETWORK,
        headers: dict[str, str] | None = None,
        prover_uri: str | None = None,
        record_scanner_uri: str | None = None,
        transport: Any = None,
        api_key: str | None = None,
        consumer_id: str | None = None,
        jwt_data: dict[str, Any] | None = None,
    ) -> None:
        try:
            import httpx  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "httpx is required for AsyncAleoNetworkClient. "
                "Install with: pip install aleo[async]"
            ) from None

        self._network: str = network
        # See AleoNetworkClient._resolve_urls: hosted Provable API → /v2 reads +
        # /prove + /scanner off the origin; any other host → literal read base
        # with no hosted prover/scanner.
        read_host, origin, prover_default, scanner_default = self._resolve_urls(host)
        self._origin: str = origin
        self._base_url: str = origin  # compat alias (now the origin)
        self._host: str = read_host
        self._has_custom_transport: bool = transport is not None
        self._transport: Any = transport
        self._account: Any = None
        self._verbose_errors: bool = True
        self.api_key: str | None = api_key
        self.consumer_id: str | None = consumer_id
        self.jwt_data: dict[str, Any] | None = jwt_data
        self._prover_uri: str | None = (
            f"{prover_uri}/{network}" if prover_uri else prover_default
        )
        self._record_scanner_uri: str | None = (
            f"{record_scanner_uri}/{network}" if record_scanner_uri else scanner_default
        )

        if self._has_custom_transport:
            self.headers: dict[str, str] = dict(headers) if headers else {}
        elif headers is not None:
            self.headers = dict(headers)
        else:
            self.headers = make_default_headers()

        # Pass the transport to httpx when provided (e.g. httpx.MockTransport
        # for tests, or a custom httpx.AsyncBaseTransport for production use).
        if transport is not None:
            self._client: Any = httpx.AsyncClient(transport=transport)
        else:
            self._client = httpx.AsyncClient()

    def _resolve_urls(
        self, host: str
    ) -> tuple[str, str, str | None, str | None]:
        """Resolve ``(read_host, origin, prover_default, scanner_default)`` for *host*.

        Hosted Provable API → reads under ``/v2`` with the delegated prover
        (``/prove``) and hosted scanner (``/scanner``) off the same origin; any
        other host → literal read base with no hosted prover/scanner. (``/jwts``
        and ``/consumers`` always live at the bare origin — handled in
        :meth:`_refresh_jwt`, not here.) Mirrors ``AleoNetworkClient._resolve_urls``.
        """
        origin = jwt_origin(host)
        network = self._network
        if is_provable_host(host):
            return (
                f"{origin}/v2/{network}",
                origin,
                f"{origin}/prove/{network}",
                f"{origin}/scanner/{network}",
            )
        return (f"{host.rstrip('/')}/{network}", origin, None, None)

    # ── Network module selection ──────────────────────────────────────────

    def _net(self) -> Any:
        """Return the network extension module (``aleo.mainnet`` or ``aleo.testnet``).

        Node responses must be parsed into types from the module matching
        ``self._network``; mixing mainnet/testnet types raises cross-extension
        ``TypeError``s.
        """
        try:
            if self._network == "testnet":
                from . import testnet as _mod  # type: ignore[attr-defined]
            else:
                from . import mainnet as _mod  # type: ignore[attr-defined]
        except ImportError:
            raise ImportError(
                f"aleo {self._network} module not available"
            ) from None
        return _mod

    # ── Mutators ──────────────────────────────────────────────────────────

    def set_host(self, host: str) -> None:
        read_host, origin, prover_default, scanner_default = self._resolve_urls(host)
        self._origin = origin
        self._base_url = origin
        self._host = read_host
        self._prover_uri = prover_default
        self._record_scanner_uri = scanner_default

    def set_prover_uri(self, prover_uri: str) -> None:
        self._prover_uri = f"{prover_uri}/{self._network}"

    def set_record_scanner_uri(self, record_scanner_uri: str) -> None:
        self._record_scanner_uri = f"{record_scanner_uri}/{self._network}"

    def set_account(self, account: Any) -> None:
        self._account = account

    def get_account(self) -> Any:
        return self._account

    @property
    def origin(self) -> str:
        """The API origin (``scheme://host``) all services derive from."""
        return self._origin

    @property
    def prover_uri(self) -> str | None:
        """DPS prover base (``{origin}/prove/{network}`` on the hosted API), or None."""
        return self._prover_uri

    @property
    def scanner_uri(self) -> str | None:
        """Hosted-scanner base (``{origin}/scanner/{network}`` on the hosted API), or None."""
        return self._record_scanner_uri

    def set_header(self, name: str, value: str) -> None:
        self.headers[name] = value

    def remove_header(self, name: str) -> None:
        self.headers.pop(name, None)

    def set_verbose_errors(self, verbose: bool) -> None:
        self._verbose_errors = verbose

    # ── Internal HTTP ─────────────────────────────────────────────────────

    def _request_headers(self, method_name: str) -> dict[str, str]:
        return method_headers(self.headers, method_name, self._has_custom_transport)

    async def _get(self, path: str, method_name: str) -> Any:
        url = self._host + path
        hdrs = self._request_headers(method_name)

        async def _do() -> Any:
            resp = await self._client.get(url, headers=hdrs)
            if not resp.is_success:
                raise AleoNetworkError(
                    f"GET {url} returned {resp.status_code}: {resp.text}",
                    status=resp.status_code,
                )
            return resp.json()

        return await async_retry_with_backoff(_do)

    async def _get_raw(self, path: str, method_name: str) -> str:
        url = self._host + path
        hdrs = self._request_headers(method_name)

        async def _do() -> str:
            resp = await self._client.get(url, headers=hdrs)
            if not resp.is_success:
                raise AleoNetworkError(
                    f"GET {url} returned {resp.status_code}: {resp.text}",
                    status=resp.status_code,
                )
            return resp.text

        return await async_retry_with_backoff(_do)

    async def _post(
        self,
        url: str,
        body: str,
        method_name: str,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        hdrs = {
            **self._request_headers(method_name),
            "Content-Type": "application/json",
            **(extra_headers or {}),
        }

        async def _do() -> Any:
            resp = await self._client.post(url, content=body.encode(), headers=hdrs)
            if not resp.is_success:
                raise AleoNetworkError(
                    f"POST {url} returned {resp.status_code}: {resp.text}",
                    status=resp.status_code,
                )
            return resp

        return await async_retry_with_backoff(_do)

    # ── JWT refresh ───────────────────────────────────────────────────────

    async def _refresh_jwt(self, api_key: str, consumer_id: str) -> dict[str, Any]:
        url = f"{self._origin}/jwts/{consumer_id}"
        hdrs = {
            **self._request_headers("refreshJwt"),
            "X-Provable-API-Key": api_key,
        }
        resp = await self._client.post(url, headers=hdrs)
        if not resp.is_success:
            raise AleoNetworkError(
                f"JWT refresh failed: {resp.status_code}: {resp.text}",
                status=resp.status_code,
            )
        auth = resp.headers.get("authorization") or resp.headers.get("Authorization")
        if not auth:
            raise AleoNetworkError("No authorization header in JWT refresh response")
        body = resp.json()
        return {"jwt": auth, "expiration": body["exp"] * 1000}

    async def _ensure_jwt(
        self,
        api_key: str | None,
        consumer_id: str | None,
        jwt_data: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if jwt_data and not jwt_expired(jwt_data):
            return jwt_data
        resolved_key = api_key or self.api_key
        resolved_cid = consumer_id or self.consumer_id
        if resolved_key and resolved_cid:
            new_jwt = await self._refresh_jwt(resolved_key, resolved_cid)
            self.jwt_data = new_jwt
            return new_jwt
        return jwt_data

    # ── Block endpoints ───────────────────────────────────────────────────

    async def get_block(self, height: int) -> Any:
        return await self._get(f"/block/{height}", "getBlock")

    async def get_block_by_hash(self, block_hash: str) -> Any:
        return await self._get(f"/block/{block_hash}", "getBlockByHash")

    async def get_block_range(self, start: int, end: int) -> list[Any]:
        validate_block_range(start, end)
        return await self._get(f"/blocks?start={start}&end={end}", "getBlockRange")

    async def get_latest_block(self) -> Any:
        return await self._get("/block/latest", "getLatestBlock")

    async def get_latest_height(self) -> int:
        return int(await self._get("/block/height/latest", "getLatestHeight"))

    async def get_latest_block_hash(self) -> str:
        return str(await self._get("/block/hash/latest", "getLatestBlockHash"))

    async def get_latest_committee(self) -> Any:
        return await self._get("/committee/latest", "getLatestCommittee")

    async def get_committee_by_height(self, height: int) -> Any:
        return await self._get(f"/committee/{height}", "getCommitteeByHeight")

    async def get_state_root(self) -> str:
        return str(await self._get("/stateRoot/latest", "getStateRoot"))

    async def get_state_paths(self, commitments: list[str]) -> list[Any]:
        csv = ",".join(quote(c, safe="") for c in commitments)
        return await self._get(f"/statePaths?commitments={csv}", "getStatePaths")

    # ── Program endpoints ─────────────────────────────────────────────────

    async def get_program(self, program_id: str, edition: int | None = None) -> str:
        if edition is not None:
            return await self._get(f"/program/{program_id}/{edition}", "getProgramVersion")
        return await self._get(f"/program/{program_id}", "getProgramVersion")

    async def get_latest_program_edition(self, program_id: str) -> int:
        raw = await self._get_raw(f"/program/{program_id}/latest_edition", "getLatestProgramEdition")
        return int(json.loads(raw))

    async def get_program_amendment_count(self, program_id: str) -> Any:
        raw = await self._get_raw(f"/program/{program_id}/amendment_count", "getProgramAmendmentCount")
        return json.loads(raw)

    async def get_program_object(self, program_id: str, edition: int | None = None) -> Any:
        Program = self._net().Program
        source = await self.get_program(program_id, edition)
        return Program.from_source(source)

    async def get_program_imports(
        self,
        program_id: str,
        imports: dict[str, str] | None = None,
    ) -> dict[str, str]:
        if imports is None:
            imports = {}
        source = await self.get_program(program_id)
        return await self._collect_program_imports(source, imports)

    async def _collect_program_imports(
        self,
        source: str,
        imports: dict[str, str],
    ) -> dict[str, str]:
        """Async DFS import collection — source already fetched, no re-fetch."""
        Program = self._net().Program
        prog = Program.from_source(source)
        for imp_id_obj in prog.imports:
            imp_id = str(imp_id_obj)
            if imp_id not in imports:
                imp_source = await self.get_program(imp_id)
                # Recurse into nested imports before recording this one
                await self._collect_program_imports(imp_source, imports)
                imports[imp_id] = imp_source
        return imports

    async def get_program_import_names(self, program_id: str) -> list[str]:
        Program = self._net().Program
        source = await self.get_program(program_id)
        prog = Program.from_source(source)
        return [str(imp) for imp in prog.imports]

    async def get_program_mapping_plaintext(
        self, program_id: str, mapping_name: str, key: str
    ) -> Any:
        Plaintext = self._net().Plaintext
        raw = await self._get_raw(
            f"/program/{program_id}/mapping/{mapping_name}/{key}",
            "getProgramMappingPlaintext",
        )
        import json as _json
        return Plaintext.from_string(_json.loads(raw))

    async def get_transaction_object(self, tx_id: str) -> Any:
        Transaction = self._net().Transaction
        raw = await self._get_raw(f"/transaction/{tx_id}", "getTransactionObject")
        return Transaction.from_json(raw)

    async def get_program_mapping_names(self, program_id: str) -> list[str]:
        return await self._get(f"/program/{program_id}/mappings", "getProgramMappingNames")

    async def get_program_mapping_value(
        self, program_id: str, mapping_name: str, key: str
    ) -> str:
        return await self._get(
            f"/program/{program_id}/mapping/{mapping_name}/{key}",
            "getProgramMappingValue",
        )

    async def get_public_balance(self, address: str) -> int:
        try:
            val = await self.get_program_mapping_value("credits.aleo", "account", address)
            return int(val) if val else 0
        except AleoNetworkError:
            return 0

    # ── Transaction endpoints ─────────────────────────────────────────────

    async def get_transaction(self, tx_id: str) -> Any:
        return await self._get(f"/transaction/{tx_id}", "getTransaction")

    async def get_confirmed_transaction(self, tx_id: str) -> Any:
        return await self._get(f"/transaction/confirmed/{tx_id}", "getConfirmedTransaction")

    async def get_transactions(self, block_height: int) -> list[Any]:
        return await self._get(f"/block/{block_height}/transactions", "getTransactions")

    async def get_transactions_in_mempool(self) -> list[Any]:
        return await self._get("/memoryPool/transactions", "getTransactionsInMempool")

    async def get_transition_id(self, input_or_output_id: str) -> str:
        return await self._get(
            f"/find/transitionID/{input_or_output_id}", "getTransitionId"
        )

    async def get_deployment_transaction_id_for_program(self, program_id: str) -> str:
        raw = await self._get(
            f"/find/transactionID/deployment/{program_id}",
            "getDeploymentTransactionIDForProgram",
        )
        return strip_quotes(str(raw))

    async def get_deployment_transaction_for_program(self, program_id: str) -> Any:
        tx_id = await self.get_deployment_transaction_id_for_program(program_id)
        return await self.get_transaction(tx_id)

    # ── POST endpoints ────────────────────────────────────────────────────

    async def submit_transaction(self, transaction: Any) -> str:
        tx_str = str(transaction)
        endpoint = (
            f"{self._host}/transaction/broadcast?check_transaction=true"
            if self._verbose_errors
            else f"{self._host}/transaction/broadcast"
        )
        resp = await self._post(endpoint, tx_str, "submitTransaction")
        return json.loads(resp.text)

    async def submit_solution(self, solution: str) -> str:
        resp = await self._post(
            f"{self._host}/solution/broadcast", solution, "submitSolution"
        )
        return json.loads(resp.text)

    # ── Wait for confirmation ─────────────────────────────────────────────

    async def wait_for_transaction_confirmation(
        self,
        tx_id: str,
        check_interval: float = 2.0,
        timeout: float = 45.0,
    ) -> Any:
        import asyncio
        import time as _time

        start = _time.monotonic()
        url = f"{self._host}/transaction/confirmed/{tx_id}"
        hdrs = self._request_headers("waitForTransactionConfirmation")

        while True:
            elapsed = _time.monotonic() - start
            if elapsed > timeout:
                raise TimeoutError(
                    f"Transaction {tx_id} did not appear after {timeout}s"
                )
            try:
                resp = await self._client.get(url, headers=hdrs)
                if not resp.is_success:
                    text = resp.text
                    if resp.status_code >= 400 and resp.status_code < 500 and "Invalid URL" in text:
                        raise AleoNetworkError(
                            f"Malformed transaction ID: {text}", status=resp.status_code
                        )
                    await asyncio.sleep(check_interval)
                    continue
                data = resp.json()
                status = data.get("status")
                if status == "accepted":
                    return data
                if status == "rejected":
                    raise AleoNetworkError(
                        f"Transaction {tx_id} was rejected by the network"
                    )
            except AleoNetworkError:
                raise
            except Exception:
                pass
            await asyncio.sleep(check_interval)

    # ── DPS ───────────────────────────────────────────────────────────────

    async def submit_proving_request_safe(
        self,
        proving_request: Any,
        *,
        url: str | None = None,
        api_key: str | None = None,
        consumer_id: str | None = None,
        jwt_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Prover base: {origin}/prove/{network} on the hosted API (set at
        # construction). Off the hosted API there is no prover unless configured.
        prover_uri = url or self._prover_uri
        if not prover_uri:
            return {
                "ok": False,
                "status": None,
                "error": {
                    "message": (
                        "No delegated prover configured for host "
                        f"{self._origin!r}. Delegated proving is available on the "
                        "Provable API (api.provable.com); for another endpoint pass "
                        "HTTPProvider(prover_uri=...)."
                    )
                },
            }
        # Build auth headers, optionally forcing a fresh JWT mint. The prover and
        # the hosted scanner share ONE consumer, and the auth server keeps a
        # single active JWT per consumer — so a scanner JWT mint invalidates the
        # prover's cached one out-of-band. On a 401 we drop the cached JWT and
        # re-mint (force_refresh) before retrying, so the prover self-heals.
        async def _build_hdrs(force_refresh: bool) -> dict[str, str]:
            if force_refresh:
                self.jwt_data = None
            rj = None if force_refresh else (jwt_data or self.jwt_data)
            rj = await self._ensure_jwt(api_key, consumer_id, rj)
            h: dict[str, str] = {
                **self._request_headers("submitProvingRequest"),
                "Content-Type": "application/json",
            }
            if rj and rj.get("jwt"):
                h["Authorization"] = rj["jwt"]
            return h

        if isinstance(proving_request, str):
            pr_obj = self._net().ProvingRequest.from_string(proving_request)
        else:
            pr_obj = proving_request

        kind = pr_obj.kind()
        endpoint = "/prove/request" if kind == "request" else "/prove/authorization"

        class _AuthInvalidated(Exception):
            """The JWT was rejected (401/403) — force a fresh mint and retry."""

        async def _send_once(hdrs: dict[str, str]) -> dict[str, Any]:
            # Prover affinity: the ephemeral X25519 private key lives ONLY on the
            # backend that served this /pubkey, so /prove must hit the same one.
            # The persistent httpx.AsyncClient owns a cookie jar that captures
            # the affinity cookie from this GET and auto-attaches it to the POST
            # below — including through a custom transport, which httpx wraps at
            # the client level (the jar sits above the transport). So we rely on
            # the jar rather than hand-building a Cookie header from set-cookie
            # (which comma-joins cookies, carries attributes, and bypasses the
            # jar — silently dropping affinity).
            pk_resp = await self._client.get(f"{prover_uri}/pubkey", headers=hdrs)
            if pk_resp.status_code in (401, 403):
                raise _AuthInvalidated()
            if not pk_resp.is_success:
                raise AleoNetworkError(
                    f"Failed to fetch pubkey: {pk_resp.status_code}",
                    status=pk_resp.status_code,
                )
            pk_data = pk_resp.json()
            key_id = pk_data["key_id"]
            public_key = pk_data["public_key"]

            pr_bytes = bytes(pr_obj.bytes())
            ciphertext = encrypt_proving_request(public_key, pr_bytes)
            payload = json.dumps({"key_id": key_id, "ciphertext": ciphertext})

            resp = await self._client.post(
                f"{prover_uri}{endpoint}",
                content=payload.encode(),
                headers=hdrs,
            )

            if resp.status_code == 200:
                return {"ok": True, "data": resp.json()}
            elif resp.status_code in (401, 403):
                raise _AuthInvalidated()
            elif resp.status_code in (400, 500, 503):
                try:
                    err_body = resp.json()
                    msg = err_body.get("message", resp.text)
                except Exception:
                    msg = resp.text
                if resp.status_code in (500, 503):
                    raise AleoNetworkError(msg, status=resp.status_code)
                return {"ok": False, "status": resp.status_code, "error": {"message": msg}}
            else:
                return {"ok": False, "status": resp.status_code, "error": {"message": resp.text}}

        async def _send_with_auth_retry() -> dict[str, Any]:
            try:
                return await _send_once(await _build_hdrs(force_refresh=False))
            except _AuthInvalidated:
                # JWT invalidated out-of-band (shared-consumer rotation) — mint a
                # fresh one and retry once.
                return await _send_once(await _build_hdrs(force_refresh=True))

        try:
            return await async_retry_with_backoff(_send_with_auth_retry)
        except _AuthInvalidated:
            return {"ok": False, "status": 401, "error": {"message": "JWT rejected (401) after refresh"}}
        except AleoNetworkError as exc:
            return {"ok": False, "status": exc.status or 500, "error": {"message": str(exc)}}

    async def submit_proving_request(
        self,
        proving_request: Any,
        *,
        url: str | None = None,
        api_key: str | None = None,
        consumer_id: str | None = None,
        jwt_data: dict[str, Any] | None = None,
    ) -> Any:
        result = await self.submit_proving_request_safe(
            proving_request,
            url=url,
            api_key=api_key,
            consumer_id=consumer_id,
            jwt_data=jwt_data,
        )
        if result["ok"]:
            return result["data"]
        err = result.get("error", {})
        raise AleoProvingError(
            err.get("message", "Proving failed"),
            status=result.get("status"),
        )
