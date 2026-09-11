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
    service_root,
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
                "Install with: pip install aleo-sdk[async]"
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
        network = self._network
        if is_provable_host(host):
            root = service_root(host)       # keeps the edge's /api prefix
            return (
                f"{root}/v2/{network}",
                root,
                f"{root}/prove/{network}",
                f"{root}/scanner/{network}",
            )
        return (f"{host.rstrip('/')}/{network}", jwt_origin(host), None, None)

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
        """Re-point the client at a different API host.

        Re-derives the read base, origin, and the delegated prover/scanner bases
        from *host*, discarding any prover or scanner URI set earlier — call
        :meth:`set_prover_uri` / :meth:`set_record_scanner_uri` again afterwards
        if you had overridden them. Local only; opens no connection.

        Args:
            host: Versioned API root, e.g. ``"https://api.provable.com/v2"``.
                Off the hosted Provable API the prover and scanner bases are
                left unset, since those services exist only there.
        """
        read_host, origin, prover_default, scanner_default = self._resolve_urls(host)
        self._origin = origin
        self._base_url = origin
        self._host = read_host
        self._prover_uri = prover_default
        self._record_scanner_uri = scanner_default

    def set_prover_uri(self, prover_uri: str) -> None:
        """Point delegated proving at a specific prover host.

        Use this to prove against a non-default prover; the current network name
        is appended for you, so pass the base without it.

        Args:
            prover_uri: Prover base without the network suffix, e.g.
                ``"https://api.provable.com/prove"``.
        """
        self._prover_uri = f"{prover_uri}/{self._network}"

    def set_record_scanner_uri(self, record_scanner_uri: str) -> None:
        """Point record scanning at a specific scanner host.

        The current network name is appended for you, so pass the base without
        it. Note that delegating a scan shares your view key with the service.

        Args:
            record_scanner_uri: Scanner base without the network suffix, e.g.
                ``"https://api.provable.com/scanner"``.
        """
        self._record_scanner_uri = f"{record_scanner_uri}/{self._network}"

    def set_account(self, account: Any) -> None:
        """Attach an account for calls that need one to sign or decrypt.

        Args:
            account: The account to associate with this client.
        """
        self._account = account

    def get_account(self) -> Any:
        """Return the account attached by :meth:`set_account`, or None if unset."""
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
        """Set a header sent with every subsequent request.

        Overwrites any existing value for *name*, including the SDK defaults.

        Args:
            name: Header name.
            value: Header value.
        """
        self.headers[name] = value

    def remove_header(self, name: str) -> None:
        """Stop sending a header, if it was set. Absent names are ignored.

        Args:
            name: Header name to drop.
        """
        self.headers.pop(name, None)

    def set_verbose_errors(self, verbose: bool) -> None:
        """Choose whether broadcasts ask the node to pre-check the transaction.

        When enabled (the default), :meth:`submit_transaction` broadcasts with
        ``check_transaction=true`` so the node reports why a transaction is
        invalid instead of silently dropping it — at the cost of extra node-side
        verification work.

        Args:
            verbose: True to request the pre-check, False to broadcast bare.
        """
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
        """Fetch the block at *height*.

        Args:
            height: Block height to read.

        Returns:
            The block as decoded JSON.

        Raises:
            AleoNetworkError: If no block exists at that height, or the node
                rejects the request.
        """
        return await self._get(f"/block/{height}", "getBlock")

    async def get_block_by_hash(self, block_hash: str) -> Any:
        """Fetch a block by its hash.

        Args:
            block_hash: Block hash (``ab1…``).

        Returns:
            The block as decoded JSON.

        Raises:
            AleoNetworkError: If the hash matches no block on this network.
        """
        return await self._get(f"/block/{block_hash}", "getBlockByHash")

    async def get_block_range(self, start: int, end: int) -> list[Any]:
        """Fetch a range of blocks in one request.

        Args:
            start: First height to fetch; must be non-negative.
            end: Last height of the range; must be at least *start*, and no more
                than 50 above it.

        Returns:
            The blocks in ascending height order.

        Raises:
            ValueError: If *start* is negative, exceeds *end*, or the span is
                wider than 50 blocks — checked locally, before any request.
            AleoNetworkError: If the node rejects the request.
        """
        validate_block_range(start, end)
        return await self._get(f"/blocks?start={start}&end={end}", "getBlockRange")

    async def get_latest_block(self) -> Any:
        """Fetch the newest block the node has.

        Returns:
            The latest block as decoded JSON.

        Raises:
            AleoNetworkError: If the node rejects the request.
        """
        return await self._get("/block/latest", "getLatestBlock")

    async def get_latest_height(self) -> int:
        """Fetch the current chain tip height.

        Cheaper than :meth:`get_latest_block` when you only need the number.

        Returns:
            The height of the newest block.

        Raises:
            AleoNetworkError: If the node rejects the request.
        """
        return int(await self._get("/block/height/latest", "getLatestHeight"))

    async def get_latest_block_hash(self) -> str:
        """Fetch the hash of the newest block.

        Returns:
            The latest block hash (``ab1…``).

        Raises:
            AleoNetworkError: If the node rejects the request.
        """
        return str(await self._get("/block/hash/latest", "getLatestBlockHash"))

    async def get_latest_committee(self) -> Any:
        """Fetch the current validator committee.

        Returns:
            The committee — members and their stake — as decoded JSON.

        Raises:
            AleoNetworkError: If the node rejects the request.
        """
        return await self._get("/committee/latest", "getLatestCommittee")

    async def get_committee_by_height(self, height: int) -> Any:
        """Fetch the validator committee as of *height*.

        Args:
            height: Block height whose committee you want.

        Returns:
            The committee at that height as decoded JSON.

        Raises:
            AleoNetworkError: If the node has pruned that height or rejects the
                request.
        """
        return await self._get(f"/committee/{height}", "getCommitteeByHeight")

    async def get_state_root(self) -> str:
        """Fetch the latest global state root.

        The state root pins the chain state an offline query is built against —
        see :class:`OfflineQuery`.

        Returns:
            The current state root (``sr1…``).

        Raises:
            AleoNetworkError: If the node rejects the request.
        """
        return str(await self._get("/stateRoot/latest", "getStateRoot"))

    async def get_state_paths(self, commitments: list[str]) -> list[Any]:
        """Fetch inclusion proofs for record commitments.

        State paths are what let a proof assert that a record was in the global
        state tree without revealing which one — needed to execute offline.

        Args:
            commitments: Record commitments to prove inclusion for; each is
                URL-escaped before it goes on the wire.

        Returns:
            One state path per commitment, in the order requested.

        Raises:
            AleoNetworkError: If any commitment is unknown to the node, or the
                node rejects the request.
        """
        csv = ",".join(quote(c, safe="") for c in commitments)
        return await self._get(f"/statePaths?commitments={csv}", "getStatePaths")

    # ── Program endpoints ─────────────────────────────────────────────────

    async def get_program(self, program_id: str, edition: int | None = None) -> str:
        """Fetch a program's Aleo instructions source.

        Args:
            program_id: Program to read, e.g. ``"credits.aleo"``.
            edition: Which amendment to read; omit for the newest one on chain.

        Returns:
            The program source text.

        Raises:
            AleoNetworkError: If the program (or that edition of it) is not
                deployed on this network.
        """
        if edition is not None:
            return await self._get(f"/program/{program_id}/{edition}", "getProgramVersion")
        return await self._get(f"/program/{program_id}", "getProgramVersion")

    async def get_latest_program_edition(self, program_id: str) -> int:
        """Fetch the newest edition number for a program.

        Pass the result to :meth:`get_program` to pin a read to the edition you
        checked, rather than racing a later amendment.

        Args:
            program_id: Program to read, e.g. ``"credits.aleo"``.

        Returns:
            The newest edition number on chain.

        Raises:
            AleoNetworkError: If the program is not deployed on this network.
        """
        raw = await self._get_raw(f"/program/{program_id}/latest_edition", "getLatestProgramEdition")
        return int(json.loads(raw))

    async def get_program_amendment_count(self, program_id: str) -> Any:
        """Fetch how many times a program has been amended.

        Args:
            program_id: Program to read, e.g. ``"credits.aleo"``.

        Returns:
            The amendment count as decoded JSON.

        Raises:
            AleoNetworkError: If the program is not deployed on this network.
        """
        raw = await self._get_raw(f"/program/{program_id}/amendment_count", "getProgramAmendmentCount")
        return json.loads(raw)

    async def get_program_object(self, program_id: str, edition: int | None = None) -> Any:
        """Fetch a program and parse it into a ``Program``.

        Use this over :meth:`get_program` when you want to inspect functions,
        mappings, or imports rather than hold the raw text. The result belongs to
        the extension module matching this client's network — mainnet and testnet
        types are not interchangeable.

        Args:
            program_id: Program to read, e.g. ``"credits.aleo"``.
            edition: Which amendment to read; omit for the newest one on chain.

        Returns:
            The parsed program.

        Raises:
            AleoNetworkError: If the program is not deployed on this network.
            ValueError: If the fetched source fails to parse.
        """
        Program = self._net().Program
        source = await self.get_program(program_id, edition)
        return Program.from_source(source)

    async def get_program_imports(
        self,
        program_id: str,
        imports: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Fetch a program's full import closure, sources included.

        Walks imports depth-first, so an import's own imports are resolved before
        it. Each distinct program is fetched once, but this still costs one
        request per program in the closure.

        Args:
            program_id: Program whose imports to resolve.
            imports: Already-known ``{program_id: source}`` entries to treat as
                fetched; mutated in place and returned. Pass a populated dict to
                reuse a closure across calls.

        Returns:
            Every transitive import as ``{program_id: source}``. The program
            itself is not included.

        Raises:
            AleoNetworkError: If the program or any of its imports is not
                deployed on this network.
        """
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
        """Fetch the names a program imports directly.

        Unlike :meth:`get_program_imports` this does not recurse and does not
        fetch the imported sources — one request, names only.

        Args:
            program_id: Program whose imports to list.

        Returns:
            The directly-imported program IDs, in declaration order.

        Raises:
            AleoNetworkError: If the program is not deployed on this network.
        """
        Program = self._net().Program
        source = await self.get_program(program_id)
        prog = Program.from_source(source)
        return [str(imp) for imp in prog.imports]

    async def get_program_mapping_plaintext(
        self, program_id: str, mapping_name: str, key: str
    ) -> Any:
        """Read a mapping entry and parse it into a ``Plaintext``.

        Use this over :meth:`get_program_mapping_value` when the value is a struct
        or record and you want to index into it instead of parsing the string
        yourself.

        Args:
            program_id: Program that owns the mapping.
            mapping_name: Mapping to read.
            key: Mapping key, written as the Aleo literal the mapping is keyed by.

        Returns:
            The parsed value, from the extension module matching this client's
            network.

        Raises:
            AleoNetworkError: If the program or mapping does not exist.
            ValueError: If the returned value fails to parse as plaintext.
        """
        Plaintext = self._net().Plaintext
        raw = await self._get_raw(
            f"/program/{program_id}/mapping/{mapping_name}/{key}",
            "getProgramMappingPlaintext",
        )
        import json as _json
        return Plaintext.from_string(_json.loads(raw))

    async def get_transaction_object(self, tx_id: str) -> Any:
        """Fetch a transaction and parse it into a ``Transaction``.

        Use this over :meth:`get_transaction` when you want to walk transitions,
        inputs, or outputs as typed objects.

        Args:
            tx_id: Transaction ID (``at1…``).

        Returns:
            The parsed transaction, from the extension module matching this
            client's network.

        Raises:
            AleoNetworkError: If the node does not know this transaction.
            ValueError: If the response fails to parse as a transaction.
        """
        Transaction = self._net().Transaction
        raw = await self._get_raw(f"/transaction/{tx_id}", "getTransactionObject")
        return Transaction.from_json(raw)

    async def get_program_mapping_names(self, program_id: str) -> list[str]:
        """Fetch the names of a program's mappings.

        Args:
            program_id: Program whose mappings to list.

        Returns:
            The mapping names declared by the program.

        Raises:
            AleoNetworkError: If the program is not deployed on this network.
        """
        return await self._get(f"/program/{program_id}/mappings", "getProgramMappingNames")

    async def get_program_mapping_value(
        self, program_id: str, mapping_name: str, key: str
    ) -> str:
        """Read one entry out of a program mapping.

        Args:
            program_id: Program that owns the mapping, e.g. ``"credits.aleo"``.
            mapping_name: Mapping to read, e.g. ``"account"``.
            key: Mapping key, written as the Aleo literal the mapping is keyed by
                (an address for ``credits.aleo/account``).

        Returns:
            The stored value as a string. An unset key reads back as the mapping's
            zero value rather than raising.

        Raises:
            AleoNetworkError: If the program or mapping does not exist, or the key
                is malformed for the mapping's key type.
        """
        return await self._get(
            f"/program/{program_id}/mapping/{mapping_name}/{key}",
            "getProgramMappingValue",
        )

    async def get_public_balance(self, address: str) -> int:
        """Read an address's public ``credits.aleo`` balance.

        Public balance only — credits held privately in records are invisible
        here, so a funded account can legitimately report 0.

        Args:
            address: The address to look up (``aleo1…``).

        Returns:
            The balance in microcredits, or 0 if the account has no public
            balance. Network failures also read back as 0 rather than raising,
            so do not use this to probe whether the node is reachable.
        """
        try:
            val = await self.get_program_mapping_value("credits.aleo", "account", address)
            return int(val) if val else 0
        except AleoNetworkError:
            return 0

    # ── Transaction endpoints ─────────────────────────────────────────────

    async def get_transaction(self, tx_id: str) -> Any:
        """Fetch a transaction by ID.

        Returns the transaction whether or not it has been confirmed; use
        :meth:`get_confirmed_transaction` when you need its on-chain outcome.

        Args:
            tx_id: Transaction ID (``at1…``).

        Returns:
            The transaction as decoded JSON.

        Raises:
            AleoNetworkError: If the node does not know this transaction.
        """
        return await self._get(f"/transaction/{tx_id}", "getTransaction")

    async def get_confirmed_transaction(self, tx_id: str) -> Any:
        """Fetch a transaction along with its confirmed outcome.

        Args:
            tx_id: Transaction ID (``at1…``).

        Returns:
            The confirmed transaction as decoded JSON, including the ``status``
            field that distinguishes an accepted transaction from a rejected one.

        Raises:
            AleoNetworkError: If the transaction is still unconfirmed or unknown
                to the node.
        """
        return await self._get(f"/transaction/confirmed/{tx_id}", "getConfirmedTransaction")

    async def get_transactions(self, block_height: int) -> list[Any]:
        """Fetch every transaction in one block.

        Args:
            block_height: Height of the block to read.

        Returns:
            The block's transactions as decoded JSON; empty if the block held
            none.

        Raises:
            AleoNetworkError: If no block exists at that height.
        """
        return await self._get(f"/block/{block_height}/transactions", "getTransactions")

    async def get_transactions_in_mempool(self) -> list[Any]:
        """Fetch the transactions this node is holding unconfirmed.

        Mempool contents are per-node and change constantly — a transaction
        missing here may still be in flight elsewhere.

        Returns:
            The node's pending transactions as decoded JSON.

        Raises:
            AleoNetworkError: If the node rejects the request — public nodes
                often decline to expose their mempool.
        """
        return await self._get("/memoryPool/transactions", "getTransactionsInMempool")

    async def get_transition_id(self, input_or_output_id: str) -> str:
        """Find which transition produced or consumed an input/output ID.

        Args:
            input_or_output_id: A transition input or output ID to trace.

        Returns:
            The enclosing transition ID (``au1…``).

        Raises:
            AleoNetworkError: If nothing on chain matches that ID.
        """
        return await self._get(
            f"/find/transitionID/{input_or_output_id}", "getTransitionId"
        )

    async def get_deployment_transaction_id_for_program(self, program_id: str) -> str:
        """Find the transaction that deployed a program.

        Args:
            program_id: Deployed program, e.g. ``"credits.aleo"``.

        Returns:
            The deployment transaction ID (``at1…``), with the node's surrounding
            JSON quotes stripped.

        Raises:
            AleoNetworkError: If the program is not deployed on this network.
        """
        raw = await self._get(
            f"/find/transactionID/deployment/{program_id}",
            "getDeploymentTransactionIDForProgram",
        )
        return strip_quotes(str(raw))

    async def get_deployment_transaction_for_program(self, program_id: str) -> Any:
        """Fetch the deployment transaction for a program. Costs two requests.

        Resolves the deployment ID, then fetches that transaction.

        Args:
            program_id: Deployed program, e.g. ``"credits.aleo"``.

        Returns:
            The deployment transaction as decoded JSON.

        Raises:
            AleoNetworkError: If the program is not deployed on this network, or
                the node cannot return its deployment transaction.
        """
        tx_id = await self.get_deployment_transaction_id_for_program(program_id)
        return await self.get_transaction(tx_id)

    # ── POST endpoints ────────────────────────────────────────────────────

    async def submit_transaction(self, transaction: Any) -> str:
        """Broadcast a proven transaction to the network.

        This spends the transaction's fee and is not reversible. Returning
        successfully means the node accepted the broadcast, NOT that the
        transaction was confirmed — pass the ID to
        :meth:`wait_for_transaction_confirmation` for that.

        When verbose errors are on (the default) the node pre-checks the
        transaction and explains rejections; see :meth:`set_verbose_errors`.

        Args:
            transaction: The proven transaction; stringified before broadcast, so
                either a ``Transaction`` or its serialized form works.

        Returns:
            The broadcast transaction ID (``at1…``).

        Raises:
            AleoNetworkError: If the node rejects the transaction — invalid proof,
                insufficient fee, or a stale state root.
        """
        tx_str = str(transaction)
        endpoint = (
            f"{self._host}/transaction/broadcast?check_transaction=true"
            if self._verbose_errors
            else f"{self._host}/transaction/broadcast"
        )
        resp = await self._post(endpoint, tx_str, "submitTransaction")
        return json.loads(resp.text)

    async def submit_solution(self, solution: str) -> str:
        """Broadcast a prover solution to the network.

        Args:
            solution: The serialized solution to submit.

        Returns:
            The accepted solution's ID.

        Raises:
            AleoNetworkError: If the node rejects the solution — stale epoch or
                insufficient proof target.
        """
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
        """Poll until a transaction is confirmed, yielding to the event loop.

        Sleeps with :func:`asyncio.sleep` between polls, so other tasks keep
        running while this one waits.

        Polls the confirmed-transaction endpoint every *check_interval* seconds.
        Transient failures — including the 404s expected while the transaction is
        still propagating — are swallowed and retried; only an outright rejection
        or a malformed ID stops the loop early.

        Args:
            tx_id: Transaction ID to watch (``at1…``).
            check_interval: Seconds to wait between polls.
            timeout: Seconds to keep polling before giving up. Exceeding it means
                the transaction never appeared, not that it failed — it may still
                confirm afterwards.

        Returns:
            The confirmed transaction as decoded JSON, once its status is
            ``accepted``.

        Raises:
            TimeoutError: If the transaction has not been confirmed within
                *timeout* seconds.
            AleoNetworkError: If the network rejected the transaction, or *tx_id*
                is malformed.
        """
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
        """Submit a proving request, returning {ok, data|error, status}. Never raises on HTTP errors."""
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
        """Submit a proving request, raising AleoProvingError on failure."""
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
