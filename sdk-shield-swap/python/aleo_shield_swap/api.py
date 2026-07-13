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
from .errors import DexApiError

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
        if not 200 <= resp.status_code < 300:
            raise DexApiError(resp.status_code, resp.text)
        return resp.json()

    def _post(self, path: str, body: dict[str, Any]) -> Any:
        resp = self._session.post(f"{self.base_url}{path}", json=body,
                                  headers=self._headers(), timeout=_TIMEOUT)
        if not 200 <= resp.status_code < 300:
            raise DexApiError(resp.status_code, resp.text)
        return resp.json()

    # ── Auth ───────────────────────────────────────────────────────────────

    def authenticate(self, address: str, sign: Any) -> str:
        """Challenge/verify handshake; stores and returns the JWT.

        *sign* is a callable taking the challenge message string and
        returning an Aleo signature literal (``sign1…``) — e.g.::

            api.authenticate(str(acct.address),
                             lambda msg: str(aleo.account.sign(msg.encode(), acct)))
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
                    "pip install 'aleo-shield-swap[async]'"
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
        if not 200 <= resp.status_code < 300:
            raise DexApiError(resp.status_code, resp.text)
        return resp.json()

    async def _post(self, path: str, body: dict[str, Any]) -> Any:
        resp = await self._client.post(f"{self.base_url}{path}", json=body,
                                       headers=self._headers())
        if not 200 <= resp.status_code < 300:
            raise DexApiError(resp.status_code, resp.text)
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
