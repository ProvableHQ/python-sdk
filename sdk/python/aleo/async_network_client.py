"""Asynchronous Aleo network client (httpx-based)."""
from __future__ import annotations

import json
from typing import Any

from ._client_common import (
    DEFAULT_HOST,
    DEFAULT_NETWORK,
    AleoNetworkError,
    AleoProvingError,
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
    """Asynchronous client for the Aleo REST API (httpx-based)."""

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

        self._base_url: str = host
        self._network: str = network
        self._host: str = f"{host}/{network}"
        self._has_custom_transport: bool = transport is not None
        self._transport = transport
        self._account: Any = None
        self._verbose_errors: bool = True
        self.api_key: str | None = api_key
        self.consumer_id: str | None = consumer_id
        self.jwt_data: dict[str, Any] | None = jwt_data
        self._prover_uri: str | None = f"{prover_uri}/{network}" if prover_uri else None
        self._record_scanner_uri: str | None = (
            f"{record_scanner_uri}/{network}" if record_scanner_uri else None
        )

        if self._has_custom_transport:
            self.headers: dict[str, str] = dict(headers) if headers else {}
        elif headers is not None:
            self.headers = dict(headers)
        else:
            self.headers = make_default_headers()

        self._client: Any = httpx.AsyncClient()

    # ── Mutators ──────────────────────────────────────────────────────────

    def set_host(self, host: str) -> None:
        self._base_url = host
        self._host = f"{host}/{self._network}"

    def set_prover_uri(self, prover_uri: str) -> None:
        self._prover_uri = f"{prover_uri}/{self._network}"

    def set_record_scanner_uri(self, record_scanner_uri: str) -> None:
        self._record_scanner_uri = f"{record_scanner_uri}/{self._network}"

    def set_account(self, account: Any) -> None:
        self._account = account

    def get_account(self) -> Any:
        return self._account

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
        origin = jwt_origin(self._base_url)
        url = f"{origin}/jwts/{consumer_id}"
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
        csv = ",".join(commitments)
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
        prover_uri = url or self._prover_uri or self._host
        resolved_jwt = jwt_data or self.jwt_data
        resolved_jwt = await self._ensure_jwt(api_key, consumer_id, resolved_jwt)

        hdrs: dict[str, str] = {
            **self._request_headers("submitProvingRequest"),
            "Content-Type": "application/json",
        }
        if resolved_jwt and resolved_jwt.get("jwt"):
            hdrs["Authorization"] = resolved_jwt["jwt"]

        try:
            from .mainnet import ProvingRequest  # type: ignore[attr-defined]
        except ImportError:
            raise ImportError("aleo mainnet module not available") from None

        if isinstance(proving_request, str):
            pr_obj = ProvingRequest.from_string(proving_request)
        else:
            pr_obj = proving_request

        kind = pr_obj.kind()
        endpoint = "/prove/request" if kind == "request" else "/prove/authorization"

        async def _send_once() -> dict[str, Any]:
            pk_resp = await self._client.get(f"{prover_uri}/pubkey", headers=hdrs)
            if not pk_resp.is_success:
                raise AleoNetworkError(
                    f"Failed to fetch pubkey: {pk_resp.status_code}",
                    status=pk_resp.status_code,
                )
            pk_data = pk_resp.json()
            key_id = pk_data["key_id"]
            public_key = pk_data["public_key"]
            cookie = pk_resp.headers.get("set-cookie")

            pr_bytes = bytes(pr_obj.bytes())
            ciphertext = encrypt_proving_request(public_key, pr_bytes)
            payload = json.dumps({"key_id": key_id, "ciphertext": ciphertext})
            post_hdrs = dict(hdrs)
            if cookie:
                post_hdrs["Cookie"] = cookie

            resp = await self._client.post(
                f"{prover_uri}{endpoint}",
                content=payload.encode(),
                headers=post_hdrs,
            )

            if resp.status_code == 200:
                return {"ok": True, "data": resp.json()}
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

        try:
            return await async_retry_with_backoff(_send_once)
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
