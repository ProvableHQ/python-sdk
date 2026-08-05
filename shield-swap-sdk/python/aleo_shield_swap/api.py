"""Typed client for the off-chain DEX API (amm-api).

Route paths and response shapes come from ``codegen/amm_api.openapi.json`` —
when they drift, rerun ``codegen/regen-openapi.sh`` and fix here.

The live API returns MORE fields than the documented schemas (e.g. pools
entries carry undocumented ``token0_info``/``token1_info``), so models are
built tolerantly: unknown keys are dropped instead of raising, and the pools
token info is surfaced through :class:`PoolEntry`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, Optional, TypeVar

import requests

from . import _api_models as models
from .errors import (
    AirdropRateLimitedError,
    DexApiError,
    NotAuthenticatedError,
    NotRedeemedError,
)


def _check(resp: Any) -> None:
    """Map DEX API failures to the lifecycle taxonomy; DexApiError otherwise.

    Takes the response object so the body is only decoded on failure.
    The 403 classification keys on the API's "invite" wording — if the
    message ever drifts, this degrades to a plain DexApiError(403), which
    every catcher of these subclasses already handles.
    """
    code = resp.status_code
    if 200 <= code < 300:
        return
    text = resp.text
    if code == 401:
        raise NotAuthenticatedError(text)
    if code == 403 and "invite" in text.lower():
        raise NotRedeemedError(text)
    if code == 429:
        raise AirdropRateLimitedError(text)
    raise DexApiError(code, text)

#: DEX API host per network.  The API is deployed per network and the two are
#: not interchangeable: pool keys and blinded identities are network-scoped, so
#: a testnet key means nothing to the mainnet indexer.
SHIELD_SWAP_API_URLS: dict[str, str] = {
    "mainnet": "https://api.swap.shield.fi",
    "testnet": "https://api.testnet.swap.shield.fi",
}


def api_url_for(network: str) -> str:
    """The DEX API base for *network*.

    ``ShieldSwap`` calls this with its bound client's network, so the API
    always matches the chain being read.  ``SHIELD_SWAP_API_URL`` overrides
    every network — set it to point at a local or staging deployment.

    Args:
        network: ``"mainnet"`` or ``"testnet"``.

    Returns:
        The base URL, without a trailing slash.

    Raises:
        ValueError: If no host is known for *network* and no override is set —
            better than silently querying the wrong chain's indexer.
    """
    override = os.environ.get("SHIELD_SWAP_API_URL")
    if override:
        return override.rstrip("/")
    try:
        return SHIELD_SWAP_API_URLS[network]
    except KeyError:
        raise ValueError(
            f"No DEX API host known for network {network!r} — expected one of "
            f"{sorted(SHIELD_SWAP_API_URLS)}, or set SHIELD_SWAP_API_URL."
        ) from None


#: Fallback for a standalone :class:`ApiClient` built without a network.  Points
#: at testnet deliberately: an accidental default must not reach mainnet.
DEFAULT_API_URL = api_url_for("testnet")
_TIMEOUT = 30.0

T = TypeVar("T")


def _build(cls: type[T], d: Any) -> T:
    """Build a generated model from a response dict, dropping unknown keys."""
    if not isinstance(d, dict):
        raise DexApiError(200, f"expected an object for {cls.__name__}, got {d!r}")
    names = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in d.items() if k in names})


@dataclass(frozen=True)
class PoolEntry:
    """One ``/pools`` entry: the documented pool state plus the undocumented
    per-token info (which carries the load-bearing ``amm_token_program`` /
    ``underlying_program`` pair).
    Delegates attribute access to the pool state, so ``entry.key`` works."""

    pool: models.PoolStateDoc
    token0_info: Optional[models.TokenDoc]
    token1_info: Optional[models.TokenDoc]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.pool, name)


class ApiClient:
    """Synchronous DEX REST client; every method returns generated models.

    Some endpoints (route quoting, OHLCV, balances) are auth-gated: call
    :meth:`authenticate` once with any Aleo account — the API authenticates
    by signature (challenge/verify), no funds required — or adopt a
    previously issued JWT via ``token=``/:meth:`set_token`.

    Auth alone is not enough for the gated endpoints: the account must also
    have redeemed an invite code (``POST /access/redeem``), otherwise they
    return 403 ``redeem an invite code to unlock access``. Check with
    ``GET /access/status``.
    """

    def __init__(self, base_url: str = DEFAULT_API_URL, session: Any | None = None,
                 token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._token = token
        self._csrf: str | None = None     # cookie-session CSRF (authenticate())

    def __repr__(self) -> str:
        return f"ApiClient({self.base_url!r})"

    @property
    def is_authenticated(self) -> bool:
        """True when a credential is loaded: an ``ss_…``/JWT bearer token or
        a cookie session from :meth:`authenticate`."""
        return bool(self._token or self._csrf)

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self._csrf:
            # A live cookie session outranks a bearer credential: it covers
            # every tier (ss_ tokens are data/trading-only) and the server
            # honors the Authorization header over cookies when both are
            # sent.  The access token rides as an httpOnly cookie on
            # self._session; requests echo the CSRF.
            headers["x-csrf-token"] = self._csrf
        elif self._token:
            headers["authorization"] = f"Bearer {self._token}"
        return headers

    def _expired_session(self, resp: Any) -> bool:
        """A 401 while riding a cookie session with a bearer in reserve:
        the (15-min) session expired — drop it and retry as bearer."""
        if resp.status_code == 401 and self._csrf and self._token:
            self._csrf = None
            return True
        return False

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._session.get(f"{self.base_url}{path}", params=params,
                                 headers=self._headers(), timeout=_TIMEOUT)
        if self._expired_session(resp):
            resp = self._session.get(f"{self.base_url}{path}", params=params,
                                     headers=self._headers(), timeout=_TIMEOUT)
        _check(resp)
        return resp.json()

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        resp = self._session.post(f"{self.base_url}{path}", json=body,
                                  headers=self._headers(), timeout=_TIMEOUT)
        if self._expired_session(resp):
            resp = self._session.post(f"{self.base_url}{path}", json=body,
                                      headers=self._headers(), timeout=_TIMEOUT)
        _check(resp)
        return resp.json()

    # ── Auth ───────────────────────────────────────────────────────────────

    def authenticate(self, address: str, sign: Any) -> str:
        """Challenge/verify handshake; establishes the session.

        The staging API issues the session as httpOnly cookies on this
        client's HTTP session plus a CSRF token (stored and echoed as
        ``X-CSRF-Token``); older deployments return a bearer JWT in the
        body — both are handled.  Returns the stored credential (CSRF token
        or JWT).  Sessions are short-lived — mint a durable ``ss_…`` token
        via :meth:`create_api_token` for anything long-running.

        *sign* is a callable taking the challenge message string and
        returning an Aleo signature literal (``sign1…``) — e.g.::

            pk = aleo.testnet.PrivateKey.from_string(key)
            api.authenticate(str(pk.address),
                             lambda msg: str(pk.sign(msg.encode())))
        """
        challenge = self._post("/auth/challenge", {"address": address})["data"]
        signature = sign(challenge["message"])
        body = {"address": address, "signature": str(signature)}
        # Staging binds the verify to its challenge; older deployments don't
        # return an id — send it only when the challenge carried one.
        if challenge.get("challenge_id"):
            body["challenge_id"] = challenge["challenge_id"]
        data = self._post("/auth/verify", body)["data"]
        if data.get("token"):             # legacy body-JWT deployments
            self._token = data["token"]
            return self._token
        self._csrf = data["csrf_token"]
        return self._csrf

    def set_token(self, token: str) -> None:
        """Adopt a previously issued JWT."""
        self._token = token

    # ── Lifecycle ──────────────────────────────────────────────────────────
    # Registration/onboarding endpoints change over time — regen the spec
    # (codegen/regen-openapi.sh) before touching these.

    def access_status(self) -> models.AccessStatusResponse:
        """Whether this authenticated account has redeemed an invite code."""
        return _build(models.AccessStatusResponse,
                      self._get("/access/status")["data"])

    def redeem_code(self, code: str) -> models.AccessRedeemResponse:
        """Redeem a pasted invite — always a REFERRAL code.

        User-shared invites are referral codes (``/referral/redeem``);
        that is the ONLY kind a person pastes.  Access codes are a separate
        programmatic tier — see :meth:`redeem_access_code` — never routed
        through here.

        Sessions moved to the ``/auth/*`` endpoints, so no token comes
        back — re-authenticate if needed (one is still adopted if the API
        resurrects the legacy body-JWT).
        """
        data = self._post("/referral/redeem", {"code": code})["data"]
        out = _build(models.AccessRedeemResponse, data)
        token = getattr(out, "token", None)
        if token:
            self._token = token
        return out

    def redeem_access_code(self, code: str) -> models.AccessRedeemResponse:
        """Redeem a programmatically minted access code
        (``POST /access/redeem``) — not for human-pasted invites, which are
        referral codes and go through :meth:`redeem_code`.

        Minting these is deliberately not exposed by this SDK; obtain a code
        out-of-band from an operator."""
        data = self._post("/access/redeem", {"code": code})["data"]
        return _build(models.AccessRedeemResponse, data)

    def request_airdrop(self, address: str) -> models.AirdropStartResult:
        """Start the test-token airdrop job for *address* (private records).

        One claim per address per 15 minutes — raises
        :class:`AirdropRateLimitedError` on 429.  Poll the returned
        ``job_id`` with :meth:`get_airdrop_job`.
        """
        return _build(models.AirdropStartResult,
                      self._post("/airdrop", {"address": address})["data"])

    def get_airdrop_job(self, job_id: str) -> models.AirdropJob:
        """Progress of an airdrop job — ``running`` until every transfer lands."""
        data = self._get(f"/airdrop/{job_id}")["data"]
        results = [_build(models.AirdropResult, r)
                   for r in (data.get("results") or [])]
        return _build(models.AirdropJob, {**data, "results": results})

    def create_api_token(self, name: str,
                         expires_in_days: "int | None" = None
                         ) -> models.ApiTokenCreatedResponse:
        """Mint a long-lived DEX API token (the secret is returned ONCE).

        JWTs from :meth:`authenticate` expire in 24h; persist the returned
        ``.token`` for durable access.  Tiering (verified live): ``ss_…``
        tokens work on data/trading endpoints; ``/access/*`` and token
        management still require a session JWT.
        """
        body: dict[str, Any] = {"name": name}
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        return _build(models.ApiTokenCreatedResponse,
                      self._post("/api-tokens", body)["data"])

    # ── Pools & tokens ─────────────────────────────────────────────────────

    def get_pools(self) -> list[PoolEntry]:
        """Every pool the DEX lists, each with its two tokens' metadata.

        An entry exposes the pool's own fields directly — ``entry.key`` is the
        ``pool_key`` that ``swap``, ``mint``, and ``collect`` take.  Its
        ``token0_info`` / ``token1_info`` carry that token's ``symbol`` and
        ``decimals``, but the API does not guarantee them — check for ``None``
        before reading.
        """
        entries = self._get("/pools")["data"]
        return [
            PoolEntry(
                pool=_build(models.PoolStateDoc, e),
                token0_info=_build(models.TokenDoc, e["token0_info"]) if e.get("token0_info") else None,
                token1_info=_build(models.TokenDoc, e["token1_info"]) if e.get("token1_info") else None,
            )
            for e in entries
        ]

    def get_tokens(self) -> list[models.TokenDoc]:
        """Every token the DEX lists, with its id, symbol, and decimals.

        ``decimals`` converts between the two amount conventions.
        The API returns canonical decimal amounts (``"1.5"``), if using this
        value to call on-chain methods — ``swap(amount_in=…)``, ``mint``,
        ``collect`` — conversion to raw base units is necessary.
        """
        return [_build(models.TokenDoc, t) for t in self._get("/tokens")["data"]]

    # ── Trading ────────────────────────────────────────────────────────────

    def get_route(self, *, token_in: str, token_out: str,
                  amount_in: Any = None) -> models.RouteResultDoc:
        """Best route between two tokens.  *amount_in* is a CANONICAL
        decimal amount (human units, e.g. ``1.5``) — not base units —
        and the returned ``estimated_amount_out`` is decimal too."""
        params: dict[str, Any] = {"token_in": token_in, "token_out": token_out}
        if amount_in is not None:
            params["amount_in"] = str(amount_in)
        return _build(models.RouteResultDoc, self._get("/route", params)["data"])

    def get_swap(self, swap_id: str) -> models.SwapDoc:
        """The indexer's record of one swap, by its id.

        Note: The API may lag slightly behind chain state, so a recently
        broadcast swap may not be visible immediately and can be retried if a
        caller has confirmed a swap on chain — raises :class:`DexApiError` (404)
        until it is.
        """
        return _build(models.SwapDoc, self._get(f"/swaps/{swap_id}")["data"])

    def get_ohlcv(self, pool_key: str, *, granularity: str,
                  from_ts: str, to_ts: str) -> list[models.OhlcvDoc]:
        """Candles for one pool over a time window.

        *granularity* is one of ``"1m"``, ``"5m"``, ``"15m"``, ``"30m"``,
        ``"1h"``, ``"6h"``, ``"12h"``, ``"1d"``.  *from_ts* and *to_ts* are unix
        seconds — *from_ts* inclusive, *to_ts* exclusive.  Anything else raises
        :class:`DexApiError`.
        """
        data = self._get(f"/pools/{pool_key}/ohlcv",
                         {"granularity": granularity, "from": from_ts, "to": to_ts})["data"]
        return [_build(models.OhlcvDoc, o) for o in data]

    # ── Balances ───────────────────────────────────────────────────────────

    def get_public_balances(self, user: str) -> list[models.TokenBalanceDoc]:
        """Public token balances for an address, as the API sees them.

        Public only — tokens held privately in records are invisible here, so this
        understates a shielded account. Use ``ShieldSwap.get_private_balances``
        to get private balances.
        """
        return [_build(models.TokenBalanceDoc, b)
                for b in self._get("/balances", {"user": user})["data"]]


class AsyncApiClient:
    """Async mirror of :class:`ApiClient` (httpx — the ``[async]`` extra)."""

    def __init__(self, base_url: str = DEFAULT_API_URL, client: Any | None = None,
                 token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        if client is None:
            try:
                import httpx
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise ImportError(
                    "AsyncApiClient requires httpx — install the async extra: "
                    "pip install 'shield-swap-sdk[async]'"
                ) from exc
            client = httpx.AsyncClient(timeout=_TIMEOUT)
        self._client = client
        self._token = token
        self._csrf: str | None = None     # cookie-session CSRF (authenticate())

    def __repr__(self) -> str:
        return f"AsyncApiClient({self.base_url!r})"

    @property
    def is_authenticated(self) -> bool:
        """True once a credential is held — a cookie-session CSRF token or a JWT.

        Reflects only that a credential was stored, not that it is still valid;
        an expired session shows True here and fails on the next call.
        """
        return bool(self._token or self._csrf)

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self._csrf:
            # Cookie session outranks bearer — see ApiClient._headers.
            headers["x-csrf-token"] = self._csrf
        elif self._token:
            headers["authorization"] = f"Bearer {self._token}"
        return headers

    def _expired_session(self, resp: Any) -> bool:
        if resp.status_code == 401 and self._csrf and self._token:
            self._csrf = None
            return True
        return False

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = await self._client.get(f"{self.base_url}{path}", params=params,
                                      headers=self._headers())
        if self._expired_session(resp):
            resp = await self._client.get(f"{self.base_url}{path}", params=params,
                                          headers=self._headers())
        _check(resp)
        return resp.json()

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        resp = await self._client.post(f"{self.base_url}{path}", json=body,
                                       headers=self._headers())
        if self._expired_session(resp):
            resp = await self._client.post(f"{self.base_url}{path}", json=body,
                                           headers=self._headers())
        _check(resp)
        return resp.json()

    async def authenticate(self, address: str, sign: Any) -> str:
        """Async challenge/verify handshake; stores and returns the JWT."""
        challenge = (await self._post("/auth/challenge", {"address": address}))["data"]
        signature = sign(challenge["message"])
        body = {"address": address, "signature": str(signature)}
        if challenge.get("challenge_id"):
            body["challenge_id"] = challenge["challenge_id"]
        data = (await self._post("/auth/verify", body))["data"]
        if data.get("token"):             # legacy body-JWT deployments
            self._token = data["token"]
            return self._token
        self._csrf = data["csrf_token"]
        return self._csrf

    def set_token(self, token: str) -> None:
        """Adopt a previously issued JWT — see :meth:`ApiClient.set_token`."""
        self._token = token

    # ── Lifecycle (async mirror of ApiClient) ──────────────────────────────

    async def access_status(self) -> models.AccessStatusResponse:
        """Whether this authenticated account has redeemed an invite code."""
        return _build(models.AccessStatusResponse,
                      (await self._get("/access/status"))["data"])

    async def redeem_code(self, code: str) -> models.AccessRedeemResponse:
        """Redeem a pasted (referral) invite — see :meth:`ApiClient.redeem_code`."""
        data = (await self._post("/referral/redeem", {"code": code}))["data"]
        out = _build(models.AccessRedeemResponse, data)
        token = getattr(out, "token", None)
        if token:
            self._token = token
        return out

    async def redeem_access_code(self, code: str) -> models.AccessRedeemResponse:
        """Redeem an access code — see :meth:`ApiClient.redeem_access_code`."""
        data = (await self._post("/access/redeem", {"code": code}))["data"]
        return _build(models.AccessRedeemResponse, data)

    async def request_airdrop(self, address: str) -> models.AirdropStartResult:
        """Start the airdrop job for *address* — see :meth:`ApiClient.request_airdrop`."""
        return _build(models.AirdropStartResult,
                      (await self._post("/airdrop", {"address": address}))["data"])

    async def get_airdrop_job(self, job_id: str) -> models.AirdropJob:
        """Progress of an airdrop job — ``running`` until every transfer lands."""
        data = (await self._get(f"/airdrop/{job_id}"))["data"]
        results = [_build(models.AirdropResult, r)
                   for r in (data.get("results") or [])]
        return _build(models.AirdropJob, {**data, "results": results})

    async def create_api_token(self, name: str,
                               expires_in_days: "int | None" = None
                               ) -> models.ApiTokenCreatedResponse:
        """Mint a long-lived DEX API token — see :meth:`ApiClient.create_api_token`."""
        body: dict[str, Any] = {"name": name}
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        return _build(models.ApiTokenCreatedResponse,
                      (await self._post("/api-tokens", body))["data"])

    async def get_pools(self) -> list[PoolEntry]:
        """Every pool with its tokens' metadata — see :meth:`ApiClient.get_pools`."""
        entries = (await self._get("/pools"))["data"]
        return [
            PoolEntry(
                pool=_build(models.PoolStateDoc, e),
                token0_info=_build(models.TokenDoc, e["token0_info"]) if e.get("token0_info") else None,
                token1_info=_build(models.TokenDoc, e["token1_info"]) if e.get("token1_info") else None,
            )
            for e in entries
        ]

    async def get_tokens(self) -> list[models.TokenDoc]:
        """Every listed token — see :meth:`ApiClient.get_tokens`."""
        return [_build(models.TokenDoc, t) for t in (await self._get("/tokens"))["data"]]

    async def get_route(self, *, token_in: str, token_out: str,
                        amount_in: int | None = None) -> models.RouteResultDoc:
        """Best route between two tokens — see :meth:`ApiClient.get_route`.

        As on the sync client, *amount_in* is stringified onto the query and the
        API reads it as a canonical decimal amount, not base units.
        """
        params: dict[str, Any] = {"token_in": token_in, "token_out": token_out}
        if amount_in is not None:
            params["amount_in"] = str(amount_in)
        return _build(models.RouteResultDoc, (await self._get("/route", params))["data"])

    async def get_swap(self, swap_id: str) -> models.SwapDoc:
        """One swap by id — see :meth:`ApiClient.get_swap`."""
        return _build(models.SwapDoc, (await self._get(f"/swaps/{swap_id}"))["data"])

    async def get_ohlcv(self, pool_key: str, *, granularity: str,
                        from_ts: str, to_ts: str) -> list[models.OhlcvDoc]:
        """Candles for one pool — see :meth:`ApiClient.get_ohlcv`."""
        data = (await self._get(f"/pools/{pool_key}/ohlcv",
                                {"granularity": granularity, "from": from_ts,
                                 "to": to_ts}))["data"]
        return [_build(models.OhlcvDoc, o) for o in data]

    async def get_public_balances(self, user: str) -> list[models.TokenBalanceDoc]:
        """Public balances for *user* — see :meth:`ApiClient.get_public_balances`."""
        return [_build(models.TokenBalanceDoc, b)
                for b in (await self._get("/balances", {"user": user}))["data"]]
