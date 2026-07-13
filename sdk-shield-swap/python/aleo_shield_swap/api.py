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
    """Synchronous DEX REST client; every method returns generated models."""

    def __init__(self, base_url: str = DEFAULT_API_URL, session: Any | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = session or requests.Session()

    def __repr__(self) -> str:
        return f"ApiClient({self.base_url!r})"

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=_TIMEOUT)
        if not 200 <= resp.status_code < 300:
            raise DexApiError(resp.status_code, resp.text)
        return resp.json()

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
                  from_ts: str, to_ts: str) -> Any:
        return self._get(f"/pools/{pool_key}/ohlcv",
                         {"granularity": granularity, "from": from_ts, "to": to_ts})["data"]

    # ── Balances ───────────────────────────────────────────────────────────

    def get_public_balances(self, user: str) -> list[models.TokenBalanceDoc]:
        return [_build(models.TokenBalanceDoc, b)
                for b in self._get("/balances", {"user": user})["data"]]
