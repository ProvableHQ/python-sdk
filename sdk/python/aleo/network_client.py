"""Synchronous Aleo network client (requests-based).

Transport contract (sync)
-------------------------
When ``transport`` is supplied it must be a callable with the signature::

    transport(method: str, url: str, **kwargs) -> requests.Response

``method`` is the HTTP verb (``"GET"`` or ``"POST"``); ``url`` is the full
URL string; ``kwargs`` are the same keyword arguments that would normally be
passed to ``requests.Session.request`` (e.g. ``headers``, ``data``).

When transport is provided, SDK-internal headers (``X-Aleo-SDK-Version``,
``X-Aleo-environment``, ``X-ALEO-METHOD``) are suppressed so the caller has
full control over the wire format.
"""
from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote

import requests

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
    retry_with_backoff,
)
from .security import encrypt_proving_request


class AleoNetworkClient:
    """Synchronous client for the Aleo REST API.

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
        Optional callable ``(method, url, **kwargs) -> requests.Response``.
        When provided every HTTP request is routed through it instead of the
        internal :class:`requests.Session`.  SDK-internal telemetry headers
        are suppressed in this mode.
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
        self._base_url: str = host  # versioned API root (e.g. https://api.provable.com/v2)
        self._network: str = network
        self._host: str = f"{host}/{network}"  # full base for node endpoints
        self._has_custom_transport: bool = transport is not None
        self._transport: Any = transport
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
            self.headers: dict[str, str] = (
                dict(headers) if headers else {}
            )
        elif headers is not None:
            self.headers = dict(headers)
        else:
            self.headers = make_default_headers()

        self._session: requests.Session = requests.Session()

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

    # ── Internal HTTP core ────────────────────────────────────────────────

    def _http(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Route an HTTP request through the custom transport or the shared session.

        All public GET/POST helpers call this method so that a custom
        *transport* is honoured consistently.  If *transport* is callable it is
        invoked as ``transport(method, url, **kwargs)``; otherwise the shared
        :class:`requests.Session` is used (transport was provided only to
        signal custom-transport mode for header suppression).
        """
        if callable(self._transport):
            resp: Any = self._transport(method, url, **kwargs)
        else:
            resp = self._session.request(method, url, **kwargs)
        return resp  # type: ignore[return-value]

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

    def _get(self, path: str, method_name: str) -> Any:
        url = self._host + path
        hdrs = self._request_headers(method_name)

        def _do() -> Any:
            resp = self._http("GET", url, headers=hdrs)
            if not resp.ok:
                raise AleoNetworkError(
                    f"GET {url} returned {resp.status_code}: {resp.text}",
                    status=resp.status_code,
                )
            return resp.json()

        return retry_with_backoff(_do)

    def _get_raw(self, path: str, method_name: str) -> str:
        url = self._host + path
        hdrs = self._request_headers(method_name)

        def _do() -> str:
            resp = self._http("GET", url, headers=hdrs)
            if not resp.ok:
                raise AleoNetworkError(
                    f"GET {url} returned {resp.status_code}: {resp.text}",
                    status=resp.status_code,
                )
            return resp.text

        return retry_with_backoff(_do)

    def _post(
        self,
        url: str,
        body: str,
        method_name: str,
        extra_headers: dict[str, str] | None = None,
    ) -> requests.Response:
        hdrs = {
            **self._request_headers(method_name),
            "Content-Type": "application/json",
            **(extra_headers or {}),
        }

        def _do() -> requests.Response:
            resp = self._http("POST", url, data=body.encode(), headers=hdrs)
            if not resp.ok:
                raise AleoNetworkError(
                    f"POST {url} returned {resp.status_code}: {resp.text}",
                    status=resp.status_code,
                )
            return resp

        return retry_with_backoff(_do)

    # ── JWT refresh ───────────────────────────────────────────────────────

    def _refresh_jwt(self, api_key: str, consumer_id: str) -> dict[str, Any]:
        origin = jwt_origin(self._base_url)
        url = f"{origin}/jwts/{consumer_id}"
        hdrs = {
            **self._request_headers("refreshJwt"),
            "X-Provable-API-Key": api_key,
        }
        resp = self._http("POST", url, headers=hdrs)
        if not resp.ok:
            raise AleoNetworkError(
                f"JWT refresh failed: {resp.status_code}: {resp.text}",
                status=resp.status_code,
            )
        auth = resp.headers.get("authorization") or resp.headers.get("Authorization")
        if not auth:
            raise AleoNetworkError("No authorization header in JWT refresh response")
        body = resp.json()
        return {"jwt": auth, "expiration": body["exp"] * 1000}

    def _ensure_jwt(
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
            new_jwt = self._refresh_jwt(resolved_key, resolved_cid)
            self.jwt_data = new_jwt
            return new_jwt
        return jwt_data

    # ── Block endpoints ───────────────────────────────────────────────────

    def get_block(self, height: int) -> Any:
        return self._get(f"/block/{height}", "getBlock")

    def get_block_by_hash(self, block_hash: str) -> Any:
        return self._get(f"/block/{block_hash}", "getBlockByHash")

    def get_block_range(self, start: int, end: int) -> list[Any]:
        validate_block_range(start, end)
        return self._get(f"/blocks?start={start}&end={end}", "getBlockRange")

    def get_latest_block(self) -> Any:
        return self._get("/block/latest", "getLatestBlock")

    def get_latest_height(self) -> int:
        return int(self._get("/block/height/latest", "getLatestHeight"))

    def get_latest_block_hash(self) -> str:
        return str(self._get("/block/hash/latest", "getLatestBlockHash"))

    def get_latest_committee(self) -> Any:
        return self._get("/committee/latest", "getLatestCommittee")

    def get_committee_by_height(self, height: int) -> Any:
        return self._get(f"/committee/{height}", "getCommitteeByHeight")

    def get_state_root(self) -> str:
        return str(self._get("/stateRoot/latest", "getStateRoot"))

    def get_state_paths(self, commitments: list[str]) -> list[Any]:
        csv = ",".join(quote(c, safe="") for c in commitments)
        return self._get(f"/statePaths?commitments={csv}", "getStatePaths")

    # ── Program endpoints ─────────────────────────────────────────────────

    def get_program(self, program_id: str, edition: int | None = None) -> str:
        if edition is not None:
            return self._get(f"/program/{program_id}/{edition}", "getProgramVersion")
        return self._get(f"/program/{program_id}", "getProgramVersion")

    def get_latest_program_edition(self, program_id: str) -> int:
        raw = self._get_raw(f"/program/{program_id}/latest_edition", "getLatestProgramEdition")
        return int(json.loads(raw))

    def get_program_amendment_count(self, program_id: str) -> Any:
        raw = self._get_raw(f"/program/{program_id}/amendment_count", "getProgramAmendmentCount")
        return json.loads(raw)

    def get_program_object(self, program_id: str, edition: int | None = None) -> Any:
        Program = self._net().Program
        source = self.get_program(program_id, edition)
        return Program.from_source(source)

    def get_program_imports(
        self,
        program_id: str,
        imports: dict[str, str] | None = None,
    ) -> dict[str, str]:
        if imports is None:
            imports = {}
        source = self.get_program(program_id)
        return self._collect_program_imports(source, imports)

    def _collect_program_imports(
        self,
        source: str,
        imports: dict[str, str],
    ) -> dict[str, str]:
        """DFS import collection — source already fetched, no re-fetch."""
        Program = self._net().Program
        prog = Program.from_source(source)
        for imp_id_obj in prog.imports:
            imp_id = str(imp_id_obj)
            if imp_id not in imports:
                imp_source = self.get_program(imp_id)
                # Recurse into nested imports before recording this one
                self._collect_program_imports(imp_source, imports)
                imports[imp_id] = imp_source
        return imports

    def get_program_import_names(self, program_id: str) -> list[str]:
        Program = self._net().Program
        source = self.get_program(program_id)
        prog = Program.from_source(source)
        return [str(imp) for imp in prog.imports]

    def get_program_mapping_names(self, program_id: str) -> list[str]:
        return self._get(f"/program/{program_id}/mappings", "getProgramMappingNames")

    def get_program_mapping_value(
        self, program_id: str, mapping_name: str, key: str
    ) -> str:
        return self._get(
            f"/program/{program_id}/mapping/{mapping_name}/{key}",
            "getProgramMappingValue",
        )

    def get_program_mapping_plaintext(
        self, program_id: str, mapping_name: str, key: str
    ) -> Any:
        Plaintext = self._net().Plaintext
        raw = self._get_raw(
            f"/program/{program_id}/mapping/{mapping_name}/{key}",
            "getProgramMappingPlaintext",
        )
        return Plaintext.from_string(json.loads(raw))

    def get_public_balance(self, address: str) -> int:
        try:
            val = self.get_program_mapping_value("credits.aleo", "account", address)
            return int(val) if val else 0
        except AleoNetworkError:
            return 0

    # ── Transaction endpoints ─────────────────────────────────────────────

    def get_transaction(self, tx_id: str) -> Any:
        return self._get(f"/transaction/{tx_id}", "getTransaction")

    def get_confirmed_transaction(self, tx_id: str) -> Any:
        return self._get(f"/transaction/confirmed/{tx_id}", "getConfirmedTransaction")

    def get_transaction_object(self, tx_id: str) -> Any:
        Transaction = self._net().Transaction
        raw = self._get_raw(f"/transaction/{tx_id}", "getTransactionObject")
        return Transaction.from_json(raw)

    def get_transactions(self, block_height: int) -> list[Any]:
        return self._get(f"/block/{block_height}/transactions", "getTransactions")

    def get_transactions_in_mempool(self) -> list[Any]:
        return self._get("/memoryPool/transactions", "getTransactionsInMempool")

    def get_transition_id(self, input_or_output_id: str) -> str:
        return self._get(
            f"/find/transitionID/{input_or_output_id}", "getTransitionId"
        )

    def get_deployment_transaction_id_for_program(self, program_id: str) -> str:
        raw = self._get(
            f"/find/transactionID/deployment/{program_id}",
            "getDeploymentTransactionIDForProgram",
        )
        return strip_quotes(str(raw))

    def get_deployment_transaction_for_program(self, program_id: str) -> Any:
        tx_id = self.get_deployment_transaction_id_for_program(program_id)
        return self.get_transaction(tx_id)

    # ── POST endpoints ────────────────────────────────────────────────────

    def submit_transaction(self, transaction: Any) -> str:
        tx_str = str(transaction)
        endpoint = (
            f"{self._host}/transaction/broadcast?check_transaction=true"
            if self._verbose_errors
            else f"{self._host}/transaction/broadcast"
        )
        resp = self._post(endpoint, tx_str, "submitTransaction")
        return json.loads(resp.text)

    def submit_solution(self, solution: str) -> str:
        resp = self._post(
            f"{self._host}/solution/broadcast", solution, "submitSolution"
        )
        return json.loads(resp.text)

    # ── Wait for confirmation ─────────────────────────────────────────────

    def wait_for_transaction_confirmation(
        self,
        tx_id: str,
        check_interval: float = 2.0,
        timeout: float = 45.0,
    ) -> Any:
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
                resp = self._http("GET", url, headers=hdrs)
                if not resp.ok:
                    text = resp.text
                    if resp.status_code >= 400 and resp.status_code < 500 and "Invalid URL" in text:
                        raise AleoNetworkError(f"Malformed transaction ID: {text}", status=resp.status_code)
                    _time.sleep(check_interval)
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
            _time.sleep(check_interval)

    # ── DPS: submit_proving_request_safe / submit_proving_request ─────────

    def submit_proving_request_safe(
        self,
        proving_request: Any,
        *,
        url: str | None = None,
        api_key: str | None = None,
        consumer_id: str | None = None,
        jwt_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a proving request, returning {ok, data|error, status}. Never raises on HTTP errors."""
        # Determine the prover URI
        prover_uri = url or self._prover_uri or self._host

        # Resolve JWT
        resolved_jwt = jwt_data or self.jwt_data
        resolved_jwt = self._ensure_jwt(api_key, consumer_id, resolved_jwt)

        # Build headers
        hdrs: dict[str, str] = {
            **self._request_headers("submitProvingRequest"),
            "Content-Type": "application/json",
        }
        if resolved_jwt and resolved_jwt.get("jwt"):
            hdrs["Authorization"] = resolved_jwt["jwt"]

        # Determine routing. A ProvingRequest object is used as-is (no import —
        # this is what delegate()/callers pass). Only a serialized string needs
        # parsing; import lazily so an object never forces the module load.
        if isinstance(proving_request, str):
            pr_obj = self._net().ProvingRequest.from_string(proving_request)
        else:
            pr_obj = proving_request

        kind = pr_obj.kind()
        endpoint = "/prove/request" if kind == "request" else "/prove/authorization"

        def _send_once() -> dict[str, Any]:
            # Fetch the ephemeral pubkey + the prover's session/affinity cookie.
            # The prover is load-balanced and holds the ephemeral X25519 private
            # key ONLY on the backend that served this /pubkey, so the follow-up
            # /prove POST MUST land on that same backend. The shared
            # requests.Session persists the affinity cookie in its jar across
            # both calls; we ALSO forward the parsed cookies explicitly via the
            # ``cookies=`` kwarg so a custom transport (which may not share a jar)
            # keeps affinity too. We do NOT hand-build a ``Cookie`` header from
            # ``set-cookie`` — that string comma-joins multiple cookies and
            # carries attributes (Path/Secure/…), and setting it manually makes
            # requests SKIP the jar, silently dropping affinity.
            pk_resp = self._http("GET", f"{prover_uri}/pubkey", headers=hdrs)
            if not pk_resp.ok:
                raise AleoNetworkError(
                    f"Failed to fetch pubkey: {pk_resp.status_code}",
                    status=pk_resp.status_code,
                )
            pk_data = pk_resp.json()
            key_id = pk_data["key_id"]
            public_key = pk_data["public_key"]

            # Encrypt
            pr_bytes = bytes(pr_obj.bytes())
            ciphertext = encrypt_proving_request(public_key, pr_bytes)

            payload = json.dumps({"key_id": key_id, "ciphertext": ciphertext})

            resp = self._http(
                "POST",
                f"{prover_uri}{endpoint}",
                data=payload.encode(),
                headers=hdrs,
                cookies=pk_resp.cookies,
            )

            if resp.status_code == 200:
                body = resp.json()
                return {"ok": True, "data": body}
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
            return retry_with_backoff(_send_once)
        except AleoNetworkError as exc:
            return {"ok": False, "status": exc.status or 500, "error": {"message": str(exc)}}

    def submit_proving_request(
        self,
        proving_request: Any,
        *,
        url: str | None = None,
        api_key: str | None = None,
        consumer_id: str | None = None,
        jwt_data: dict[str, Any] | None = None,
    ) -> Any:
        """Submit a proving request, raising AleoProvingError on failure."""
        result = self.submit_proving_request_safe(
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
