"""Typed client for the off-chain DEX API (amm-api).

Route paths and response shapes come from ``codegen/amm_api.openapi.json`` —
when they drift, rerun ``codegen/regen-openapi.sh`` and fix here.

The live API returns MORE fields than the documented schemas (e.g. pools
entries carry undocumented ``token0_info``/``token1_info``), so models are
built tolerantly: unknown keys are dropped instead of raising, and the pools
token info is surfaced through :class:`PoolEntry`.
"""
from __future__ import annotations

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

DEFAULT_API_URL = "https://amm-api.dev.provable.com"
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
    per-token info (which carries the load-bearing ``wrapper_program``).
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

    def __repr__(self) -> str:
        return f"ApiClient({self.base_url!r})"

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self._token:
            headers["authorization"] = f"Bearer {self._token}"
        return headers

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._session.get(f"{self.base_url}{path}", params=params,
                                 headers=self._headers(), timeout=_TIMEOUT)
        _check(resp)
        return resp.json()

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        resp = self._session.post(f"{self.base_url}{path}", json=body,
                                  headers=self._headers(), timeout=_TIMEOUT)
        _check(resp)
        return resp.json()

    # ── Auth ───────────────────────────────────────────────────────────────

    def authenticate(self, address: str, sign: Any) -> str:
        """Challenge/verify handshake; stores and returns the JWT.

        *sign* is a callable taking the challenge message string and
        returning an Aleo signature literal (``sign1…``) — e.g.::

            pk = aleo.testnet.PrivateKey.from_string(key)
            api.authenticate(str(pk.address),
                             lambda msg: str(pk.sign(msg.encode())))
        """
        challenge = self._post("/auth/challenge", {"address": address})
        signature = sign(challenge["data"]["message"])
        verified = self._post("/auth/verify",
                              {"address": address, "signature": str(signature)})
        self._token = verified["data"]["token"]
        return self._token

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
        """Redeem an invite code; adopts the fresh token the API returns."""
        out = _build(models.AccessRedeemResponse,
                     self._post("/access/redeem", {"code": code})["data"])
        if out.token:
            self._token = out.token
        return out

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
        return [_build(models.TokenDoc, t) for t in self._get("/tokens")["data"]]

    # ── Trading ────────────────────────────────────────────────────────────

    def get_route(self, *, token_in: str, token_out: str,
                  amount_in: int | None = None) -> models.RouteResultDoc:
        params: dict[str, Any] = {"token_in": token_in, "token_out": token_out}
        if amount_in is not None:
            params["amount_in"] = str(amount_in)
        return _build(models.RouteResultDoc, self._get("/route", params)["data"])

    def get_swap(self, swap_id: str) -> models.SwapDoc:
        return _build(models.SwapDoc, self._get(f"/swaps/{swap_id}")["data"])

    def get_ohlcv(self, pool_key: str, *, granularity: str,
                  from_ts: str, to_ts: str) -> list[models.OhlcvDoc]:
        data = self._get(f"/pools/{pool_key}/ohlcv",
                         {"granularity": granularity, "from": from_ts, "to": to_ts})["data"]
        return [_build(models.OhlcvDoc, o) for o in data]

    # ── Balances ───────────────────────────────────────────────────────────

    def get_public_balances(self, user: str) -> list[models.TokenBalanceDoc]:
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

    def __repr__(self) -> str:
        return f"AsyncApiClient({self.base_url!r})"

    def _headers(self) -> dict[str, str]:
        headers = {"accept": "application/json"}
        if self._token:
            headers["authorization"] = f"Bearer {self._token}"
        return headers

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = await self._client.get(f"{self.base_url}{path}", params=params,
                                      headers=self._headers())
        _check(resp)
        return resp.json()

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        resp = await self._client.post(f"{self.base_url}{path}", json=body,
                                       headers=self._headers())
        _check(resp)
        return resp.json()

    async def authenticate(self, address: str, sign: Any) -> str:
        """Async challenge/verify handshake; stores and returns the JWT."""
        challenge = await self._post("/auth/challenge", {"address": address})
        signature = sign(challenge["data"]["message"])
        verified = await self._post("/auth/verify",
                                    {"address": address, "signature": str(signature)})
        self._token = verified["data"]["token"]
        return self._token

    def set_token(self, token: str) -> None:
        self._token = token

    # ── Lifecycle (async mirror of ApiClient) ──────────────────────────────

    async def access_status(self) -> models.AccessStatusResponse:
        """Whether this authenticated account has redeemed an invite code."""
        return _build(models.AccessStatusResponse,
                      (await self._get("/access/status"))["data"])

    async def redeem_code(self, code: str) -> models.AccessRedeemResponse:
        """Redeem an invite code; adopts the fresh token the API returns."""
        out = _build(models.AccessRedeemResponse,
                     (await self._post("/access/redeem", {"code": code}))["data"])
        if out.token:
            self._token = out.token
        return out

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
        return [_build(models.TokenDoc, t) for t in (await self._get("/tokens"))["data"]]

    async def get_route(self, *, token_in: str, token_out: str,
                        amount_in: int | None = None) -> models.RouteResultDoc:
        params: dict[str, Any] = {"token_in": token_in, "token_out": token_out}
        if amount_in is not None:
            params["amount_in"] = str(amount_in)
        return _build(models.RouteResultDoc, (await self._get("/route", params))["data"])

    async def get_swap(self, swap_id: str) -> models.SwapDoc:
        return _build(models.SwapDoc, (await self._get(f"/swaps/{swap_id}"))["data"])

    async def get_ohlcv(self, pool_key: str, *, granularity: str,
                        from_ts: str, to_ts: str) -> list[models.OhlcvDoc]:
        data = (await self._get(f"/pools/{pool_key}/ohlcv",
                                {"granularity": granularity, "from": from_ts,
                                 "to": to_ts}))["data"]
        return [_build(models.OhlcvDoc, o) for o in data]

    async def get_public_balances(self, user: str) -> list[models.TokenBalanceDoc]:
        return [_build(models.TokenBalanceDoc, b)
                for b in (await self._get("/balances", {"user": user}))["data"]]
