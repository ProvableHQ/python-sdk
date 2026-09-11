"""Typed client for the off-chain DEX API (amm-api).

Route paths and response shapes come from ``codegen/amm_api.openapi.json`` —
when they drift, rerun ``codegen/regen-openapi.sh`` and fix here.

The live API returns MORE fields than the documented schemas (e.g. pools
entries carry undocumented ``token0_info``/``token1_info``), so models are
built tolerantly: unknown keys are dropped instead of raising, and the pools
token info is surfaced through :class:`PoolEntry`.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import types
import typing
from dataclasses import dataclass, fields
from typing import Any, Optional, TypeVar

import requests

from . import _api_models as models
from .errors import (
    AirdropRateLimitedError,
    DexApiError,
    NotAuthenticatedError,
)


def _check(resp: Any) -> None:
    """Map DEX API failures to the lifecycle taxonomy; DexApiError otherwise.

    Takes the response object so the body is only decoded on failure.
    """
    code = resp.status_code
    if 200 <= code < 300:
        return
    text = resp.text
    if code == 401:
        raise NotAuthenticatedError(text)
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


@functools.lru_cache(maxsize=None)
def _hints(cls: type) -> dict[str, Any]:
    """Resolved field annotations of a generated model (they are strings under
    ``from __future__ import annotations``)."""
    return typing.get_type_hints(cls, globalns=dict(vars(models)))


def _coerce(hint: Any, value: Any) -> Any:
    """Build nested generated models where the annotation names one."""
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        for arg in typing.get_args(hint):
            if arg is not type(None) and dataclasses.is_dataclass(arg) \
                    and isinstance(value, dict):
                return _build(arg, value)  # type: ignore[arg-type]
        return value
    if origin is list and isinstance(value, list):
        args = typing.get_args(hint)
        if args and dataclasses.is_dataclass(args[0]):
            return [_build(args[0], v) if isinstance(v, dict) else v  # type: ignore[arg-type]
                    for v in value]
        return value
    if dataclasses.is_dataclass(hint) and isinstance(value, dict):
        return _build(hint, value)  # type: ignore[arg-type]
    return value


def _session_binding(session: models.SessionPayload) -> dict[str, str]:
    """Headers that bind a session-ending request to one session — the API
    refuses ``/auth/logout`` without them whenever a refresh cookie rides."""
    headers = {"x-shield-wallet-address": str(session.address)}
    if session.session_id:
        headers["x-shield-session-id"] = str(session.session_id)
    return headers


def _build(cls: type[T], d: Any) -> T:
    """Build a generated model from a response dict, dropping unknown keys.

    Nested models (a ``TokenDoc`` inside a pool, the hops of a route, the
    boundary ticks of a rebalance state) are built recursively from the
    field annotations, so ``route.hops[0].pool_key`` is attribute access all
    the way down rather than a dict at the second level.
    """
    if not isinstance(d, dict):
        raise DexApiError(200, f"expected an object for {cls.__name__}, got {d!r}")
    hints = _hints(cls)
    model_fields = fields(cls)  # type: ignore[arg-type]
    names = {f.name for f in model_fields}
    kwargs = {k: _coerce(hints.get(k), v) for k, v in d.items() if k in names}
    # A field the API stopped sending reads as None rather than failing the
    # whole response: the spec drifts faster than releases, and one dropped
    # column must not take every pool read down with it.
    for f in model_fields:
        if (f.name not in kwargs and f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING):
            kwargs[f.name] = None
    return cls(**kwargs)


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

    Authentication alone grants access; there is no invite gate.  A
    **referral code** is optional attribution — redeem one with
    :meth:`redeem_code` to record who referred this account — and every
    authenticated account owns a code of its own to share
    (:meth:`my_referral_code`).
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

    def _post(self, path: str, body: dict[str, Any],
              headers: dict[str, str] | None = None) -> Any:
        resp = self._session.post(f"{self.base_url}{path}", json=body,
                                  headers={**self._headers(), **(headers or {})},
                                  timeout=_TIMEOUT)
        if self._expired_session(resp):
            resp = self._session.post(f"{self.base_url}{path}", json=body,
                                      headers={**self._headers(), **(headers or {})},
                                      timeout=_TIMEOUT)
        _check(resp)
        return resp.json()

    def _put(self, path: str, body: dict[str, Any]) -> Any:
        resp = self._session.put(f"{self.base_url}{path}", json=body,
                                 headers=self._headers(), timeout=_TIMEOUT)
        if self._expired_session(resp):
            resp = self._session.put(f"{self.base_url}{path}", json=body,
                                     headers=self._headers(), timeout=_TIMEOUT)
        _check(resp)
        return resp.json()

    def _delete(self, path: str) -> Any:
        resp = self._session.delete(f"{self.base_url}{path}",
                                    headers=self._headers(), timeout=_TIMEOUT)
        if self._expired_session(resp):
            resp = self._session.delete(f"{self.base_url}{path}",
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
        csrf = str(data["csrf_token"])
        self._csrf = csrf
        return csrf

    def set_token(self, token: str) -> None:
        """Adopt a previously issued JWT."""
        self._token = token

    # ── Lifecycle ──────────────────────────────────────────────────────────
    # Registration/onboarding endpoints change over time — regen the spec
    # (codegen/regen-openapi.sh) before touching these.

    # ── Session management (cookie sessions from authenticate()) ───────────

    def _drop_session(self) -> None:
        """Forget the cookie session locally after a logout/revoke: the CSRF
        token and the httpOnly cookies.  A bearer token in reserve stays."""
        self._csrf = None
        cookies = getattr(self._session, "cookies", None)
        if cookies is not None:
            cookies.clear()

    def get_session(self) -> models.SessionPayload:
        """The current cookie session: address, expiry, session id, and its
        CSRF token.  401 (:class:`NotAuthenticatedError`) once it has expired."""
        return _build(models.SessionPayload, self._get("/auth/session")["data"])

    def refresh_session(self) -> models.SessionPayload:
        """Extend the current cookie session and adopt its (possibly rotated)
        CSRF token.  Sessions are short-lived; call this from long-running
        processes, or mint an ``ss_`` token instead (:meth:`create_api_token`)."""
        data = self._post("/auth/refresh", {})["data"]
        session = _build(models.SessionPayload, data)
        if session.csrf_token:
            self._csrf = str(session.csrf_token)
        return session

    def list_sessions(self) -> list[models.ActiveSessionPayload]:
        """Every live cookie session for this account (``current`` marks this
        one), with start/refresh/expiry times and user agent."""
        return [_build(models.ActiveSessionPayload, s)
                for s in self._get("/auth/sessions")["data"]]

    def revoke_session(self, session_id: str) -> models.RevokeSessionPayload:
        """End one session by id (from :meth:`list_sessions`).  Revoking the
        current one also drops it locally."""
        out = _build(models.RevokeSessionPayload,
                     self._post(f"/auth/sessions/{session_id}/revoke", {})["data"])
        if out.current:
            self._drop_session()
        return out

    def logout(self, session: Optional[models.SessionPayload] = None
               ) -> models.LogoutResponse:
        """End the current cookie session and forget it locally.  A bearer
        token loaded via ``token=`` is untouched and keeps working.

        The server requires the request to be *bound* to the session it ends
        (``X-Shield-Session-Id`` / ``X-Shield-Wallet-Address``), so this
        reads :meth:`get_session` first unless *session* is passed."""
        session = session or self.get_session()
        out = _build(models.LogoutResponse,
                     self._post("/auth/logout", {}, _session_binding(session))["data"])
        self._drop_session()
        return out

    def logout_all(self) -> models.LogoutAllResponse:
        """End every cookie session for this account (bumps the account's
        ``session_version``) and forget the local one."""
        out = _build(models.LogoutAllResponse, self._post("/auth/logout-all", {})["data"])
        self._drop_session()
        return out

    def get_ws_ticket(self) -> models.AuthTokenPayload:
        """A short-lived JWT for the API's websocket feed (the browser client's
        live updates).  Needs a session."""
        return _build(models.AuthTokenPayload, self._get("/auth/ws-ticket")["data"])

    def referral_status(self) -> models.ReferralStatusResponse:
        """This account's referral picture: ``referred_by`` (the referrer's
        address once a code was redeemed, else None), ``my_code`` (the code
        this account shares), and ``has_access``.  Network read.

        ``has_access`` is always true for an authenticated account — access
        is granted by authentication alone, no code required — and the call
        raises :class:`NotAuthenticatedError` when the session is missing or
        expired, which makes this the session liveness probe (the dedicated
        ``/access/status`` route was retired in 2026-09)."""
        return _build(models.ReferralStatusResponse,
                      self._get("/referral/status")["data"])

    def my_referral_code(self) -> Optional[str]:
        """The referral code this account hands to others.

        The API issues the code on the first request, so this normally
        returns a value; None means issuance is disabled for the account.
        Network read.
        """
        data = self._get("/referral/my-code")["data"]
        return _build(models.ReferralMyCodeResponse, data).code

    def redeem_code(self, code: str) -> models.ReferralRedeemResponse:
        """Redeem a referral code (``POST /referral/redeem``) — optional.

        Access does not depend on this: authentication alone unlocks every
        endpoint.  Redeeming records who referred the account, once: a
        repeat returns ``status="already_redeemed"`` without changing
        anything, while an unknown code or the account's own code is a 400
        (:class:`DexApiError`).

        Sessions live on the ``/auth/*`` endpoints, so no token comes
        back (one is still adopted if the API resurrects the legacy
        body-JWT).
        """
        data = self._post("/referral/redeem", {"code": code})["data"]
        out = _build(models.ReferralRedeemResponse, data)
        token = data.get("token")
        if token:
            self._token = token
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

    def list_api_tokens(self) -> list[models.ApiTokenRow]:
        """The account's durable API tokens — prefixes and metadata only, the
        secrets are never returned again.  Needs a session (not an ``ss_`` token)."""
        data = self._get("/api-tokens")["data"]
        return _build(models.ApiTokenListResponse, data).tokens

    def revoke_api_token(self, token_id: str) -> models.ApiTokenRevokeResponse:
        """Revoke one durable API token by its id (from :meth:`list_api_tokens`).
        Needs a session; takes effect immediately for every holder of the secret."""
        return _build(models.ApiTokenRevokeResponse,
                      self._delete(f"/api-tokens/{token_id}")["data"])

    # ── Referral reporting ─────────────────────────────────────────────────

    def referral_settings(self) -> models.ReferralSettingsResponse:
        """This account's code issuance settings: ``codes_per_user`` (how many
        codes it may issue) under the deployment's ``codes_per_user_limit``,
        and an optional ``max_users`` cap on redemptions.  Needs a session."""
        return _build(models.ReferralSettingsResponse,
                      self._get("/referral/settings")["data"])

    def update_referral_settings(self, *, codes_per_user: int,
                                 max_users: Optional[int] = None
                                 ) -> models.ReferralSettingsResponse:
        """Change the issuance settings (bounded by ``codes_per_user_limit``);
        returns the settings as stored.  403 when the account may not."""
        body: dict[str, Any] = {"codes_per_user": codes_per_user}
        if max_users is not None:
            body["max_users"] = max_users
        return _build(models.ReferralSettingsResponse,
                      self._put("/referral/settings", body)["data"])

    def list_referral_codes(self, *, limit: Optional[int] = None,
                            offset: Optional[int] = None) -> models.ReferralListResponse:
        """The codes this account has issued, with redemption details, plus
        the ``total``/``redeemed``/``available``/``redemptions`` tallies.
        Page with *limit*/*offset*.  Needs a session."""
        params = {k: v for k, v in (("limit", limit), ("offset", offset)) if v is not None}
        return _build(models.ReferralListResponse,
                      self._get("/referral/codes", params or None)["data"])

    def generate_referral_codes(self, count: int = 1) -> list[str]:
        """Issue *count* new referral codes for this account to hand out.
        400 once the account's ``codes_per_user`` allowance is used up."""
        return list(_build(models.ReferralGenerateResponse,
                           self._post("/referral/generate", {"count": count})["data"]).codes)

    def report_referral_activity(self, *, action: str, tx_id: str,
                                 metadata: Any = None
                                 ) -> models.ReferralActivityResponse:
        """Attribute an on-chain action (``"create_pool"``) to this account's
        referral link.  Best-effort analytics — nothing on chain depends on it."""
        body: dict[str, Any] = {"action": action, "tx_id": tx_id}
        if metadata is not None:
            body["metadata"] = metadata
        return _build(models.ReferralActivityResponse,
                      self._post("/referral/activity", body)["data"])

    def report_referral_swap_claim(self, *, code: str, blinded_address: str
                                   ) -> models.ReferralSwapClaimResponse:
        """Link one private swap (by its blinded address) to a referral code so
        the referrer is credited without revealing the trader.  Best-effort."""
        return _build(models.ReferralSwapClaimResponse,
                      self._post("/referral/swap-claims",
                                 {"code": code, "blinded_address": blinded_address})["data"])

    def report_referral_address_batch(self, *, code: str, blinded_addresses: list[str]
                                      ) -> models.ReferralAddressBatchResponse:
        """Batch form of :meth:`report_referral_swap_claim` — returns how many
        were recorded, already known, or claimed by another code."""
        return _build(models.ReferralAddressBatchResponse,
                      self._post("/referral/address-batches",
                                 {"code": code, "blinded_addresses": blinded_addresses})["data"])

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

    def get_pool(self, pool_key: str) -> models.PoolWithStatsDoc:
        """One pool with its token metadata, reserves, display orientation, and
        rolling stats.  Network read; 404 for an unknown key."""
        return _build(models.PoolWithStatsDoc, self._get(f"/pools/{pool_key}")["data"])

    def get_pool_stats(self, pool_key: str) -> models.PoolStats24hDoc:
        """Rolling 24h/7d analytics for a pool: price and change, highs/lows,
        volume, LP fees (raw token0 units), and reserves.  Auth-gated."""
        return _build(models.PoolStats24hDoc,
                      self._get(f"/pools/{pool_key}/stats")["data"])

    def get_pool_stats_batch(self, pool_keys: list[str]
                             ) -> dict[str, models.PoolStats24hDoc]:
        """:meth:`get_pool_stats` for many pools in one request, keyed by pool
        key.  Pools the indexer has no stats for are simply absent.  Public."""
        if not pool_keys:
            return {}
        data = self._get("/pools/stats", {"keys": ",".join(pool_keys)})["data"]
        return {k: _build(models.PoolStats24hDoc, v) for k, v in data.items()}

    def get_liquidity_distribution(self, pool_key: str) -> list[models.TickLiquidityDoc]:
        """The pool's depth: ``liquidity_net`` at every initialized tick, in
        tick order (the chart behind a liquidity-distribution view).  Public;
        404 (:class:`DexApiError`) for an unknown pool."""
        return [_build(models.TickLiquidityDoc, t)
                for t in self._get(f"/pools/{pool_key}/liquidity-distribution")["data"]]

    # ── Compliance (public; check before spending on a write) ──────────────

    def get_compliance(self) -> models.GlobalConfigStatus:
        """Deployment-wide switches: ``global_paused`` (every write halts)
        and ``pool_creation_is_open`` (whether :meth:`ShieldSwap.create_pool`
        is permitted for non-operators)."""
        return _build(models.GlobalConfigStatus, self._get("/compliance")["data"])

    def get_token_compliance(self, token_id: str) -> models.TokenComplianceStatus:
        """Whether a token is ``allowed`` on the DEX and whether trading in it
        is currently ``paused``."""
        return _build(models.TokenComplianceStatus,
                      self._get(f"/compliance/tokens/{token_id}")["data"])

    def get_pair_compliance(self, token0: str, token1: str) -> models.PairComplianceStatus:
        """Whether trading between two tokens is currently ``paused``."""
        return _build(models.PairComplianceStatus,
                      self._get(f"/compliance/pairs/{token0}/{token1}")["data"])

    def get_pool_trades(self, pool_key: str, *, limit: Optional[int] = None,
                        offset: Optional[int] = None,
                        trade_type: Optional[str] = None) -> list[models.PoolTradeDoc]:
        """Recent fills in a pool, newest first, with per-leg fee split
        (``fee0`` is gross and includes ``protocolFee0``), post-trade price,
        liquidity, and tick.  Auth-gated; page with *limit*/*offset*."""
        params = {k: v for k, v in (("limit", limit), ("offset", offset),
                                    ("trade_type", trade_type)) if v is not None}
        data = self._get(f"/pools/{pool_key}/trades", params or None)["data"]
        return [_build(models.PoolTradeDoc, t) for t in data]

    def get_initialized_ticks(self, pool_key: str) -> list[int]:
        """The pool's initialized ticks, ascending — the indexer's copy of the
        on-chain tick list, usable for insert hints when a node walk is too
        slow.  Auth-gated."""
        return [int(t) for t in self._get(f"/pools/{pool_key}/initialized-ticks")["data"]]

    def get_fee_tiers(self) -> list[models.FeeTierDoc]:
        """Registered fee tiers (``fee_tier`` in hundredths of a bip) and the
        tick spacing each is bound to, or None when unbound.  Auth-gated."""
        return [_build(models.FeeTierDoc, t) for t in self._get("/fee-tiers")["data"]]

    def get_protocol_state(self, *, minimum_revision: Optional[int] = None
                           ) -> models.ProtocolStateResponse:
        """The indexer's view of protocol configuration and its own freshness.

        ``freshness.ready_for_quote`` says whether quotes reflect the chain
        head; ``revision`` increments on every config change and is echoed by
        ``/route`` as ``protocol_revision`` — pass *minimum_revision* to wait
        for the indexer to reach one.  Returned unwrapped (no ``data``).
        """
        params = {"minimum_revision": minimum_revision} if minimum_revision is not None else None
        return _build(models.ProtocolStateResponse, self._get("/protocol/state", params))

    # ── Account views ──────────────────────────────────────────────────────
    # Swap history/detail and per-token position detail are chain reads now
    # (``ShieldSwap.get_swap_output`` / ``get_swap_execution`` /
    # ``get_position``) — the API retired those routes in 2026-09.

    def get_positions(self, *, limit: Optional[int] = None,
                      offset: Optional[int] = None) -> list[models.PositionDoc]:
        """The authenticated account's positions as the indexer sees them —
        public amounts and owed balances, no record identity.  For the
        private side use ``ShieldSwap.get_owned_positions``."""
        params = {k: v for k, v in (("limit", limit), ("offset", offset)) if v is not None}
        return [_build(models.PositionDoc, p)
                for p in self._get("/positions", params or None)["data"]]

    def get_unclaimed(self) -> models.UnclaimedPayloadDoc:
        """Everything the authenticated account can still collect, as the
        indexer sees it: swaps with finalized-but-unclaimed output and
        positions with owed balances.  A cross-check for a local journal —
        the chain, not this, gates the claim amounts."""
        return _build(models.UnclaimedPayloadDoc, self._get("/unclaimed")["data"])

    # ── Trading ────────────────────────────────────────────────────────────

    def get_route(self, *, token_in: str, token_out: str,
                  amount_in: Any = None,
                  pool_key: Optional[str] = None) -> models.RouteResultDoc:
        """Best route between two tokens.  *amount_in* is a CANONICAL
        decimal amount (human units, e.g. ``1.5``) — not base units —
        and the returned ``estimated_amount_out`` is decimal too.  *pool_key*
        pins the quote to one pool instead of the router's best path."""
        params: dict[str, Any] = {"token_in": token_in, "token_out": token_out}
        if amount_in is not None:
            params["amount_in"] = str(amount_in)
        if pool_key is not None:
            params["pool_key"] = pool_key
        return _build(models.RouteResultDoc, self._get("/route", params)["data"])

    def get_route_topology(self) -> models.RouteTopologyDoc:
        """The routable token graph: every (token0, token1) edge with an
        enabled pool and the router's ``max_hops``.  Lets a client enumerate
        reachable pairs without probing ``/route`` per pair."""
        return _build(models.RouteTopologyDoc, self._get("/route/topology")["data"])

    def get_rebalance_state(self, pool_key: str, *, tick_lower: int, tick_upper: int,
                            old_liquidity: int, mint_tick_lower: int,
                            mint_tick_upper: int) -> models.RebalanceState:
        """The indexer's snapshot for planning a rebalance: live price and
        fee accumulators, the CURRENT range's boundary ticks, and insert hints
        for the successor range computed as if the old position were already
        closed.  Amounts are decimal strings; ``observed_block`` says how
        fresh.  The SDK's own planner reads the chain instead — this is for
        cross-checking or for callers without node access."""
        params = {"tick_lower": tick_lower, "tick_upper": tick_upper,
                  "old_liquidity": str(old_liquidity),
                  "mint_tick_lower": mint_tick_lower, "mint_tick_upper": mint_tick_upper}
        return _build(models.RebalanceState,
                      self._get(f"/pools/{pool_key}/rebalance-state", params)["data"])

    def get_ohlcv(self, pool_key: str, *, granularity: str,
                  from_ts: int, to_ts: int) -> list[models.OhlcvDoc]:
        """Candles for one pool over a time window.

        *granularity* is one of ``"1m"``, ``"5m"``, ``"15m"``, ``"30m"``,
        ``"1h"``, ``"6h"``, ``"12h"``, ``"1d"``.  *from_ts* and *to_ts* are unix
        seconds (the API's ``int64``) — *from_ts* inclusive, *to_ts* exclusive.
        A timestamp string rather than an integer is rejected with
        :class:`DexApiError` 400.
        """
        data = self._get(f"/pools/{pool_key}/ohlcv",
                         {"granularity": granularity, "from": from_ts, "to": to_ts})["data"]
        return [_build(models.OhlcvDoc, o) for o in data]

    # Balances are chain reads: ``ShieldSwap.get_public_balances`` (each token
    # program's ``balances`` mapping) and ``get_private_balances`` (records).
    # The API's ``/balances`` route was retired in 2026-09.


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

    async def _post(self, path: str, body: dict[str, Any],
                    headers: dict[str, str] | None = None) -> Any:
        resp = await self._client.post(f"{self.base_url}{path}", json=body,
                                       headers={**self._headers(), **(headers or {})})
        if self._expired_session(resp):
            resp = await self._client.post(f"{self.base_url}{path}", json=body,
                                           headers={**self._headers(), **(headers or {})})
        _check(resp)
        return resp.json()

    async def _put(self, path: str, body: dict[str, Any]) -> Any:
        resp = await self._client.put(f"{self.base_url}{path}", json=body, headers=self._headers())
        if self._expired_session(resp):
            resp = await self._client.put(f"{self.base_url}{path}", json=body,
                                          headers=self._headers())
        _check(resp)
        return resp.json()

    async def _delete(self, path: str) -> Any:
        resp = await self._client.delete(f"{self.base_url}{path}", headers=self._headers())
        if self._expired_session(resp):
            resp = await self._client.delete(f"{self.base_url}{path}", headers=self._headers())
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
        csrf = str(data["csrf_token"])
        self._csrf = csrf
        return csrf

    def set_token(self, token: str) -> None:
        """Adopt a previously issued JWT — see :meth:`ApiClient.set_token`."""
        self._token = token

    # ── Lifecycle (async mirror of ApiClient) ──────────────────────────────

    # ── Session management (async mirrors) ─────────────────────────────────

    def _drop_session(self) -> None:
        self._csrf = None
        cookies = getattr(self._client, "cookies", None)
        if cookies is not None:
            cookies.clear()

    async def get_session(self) -> models.SessionPayload:
        """Current cookie session — see :meth:`ApiClient.get_session`."""
        return _build(models.SessionPayload, (await self._get("/auth/session"))["data"])

    async def refresh_session(self) -> models.SessionPayload:
        """Extend the session — see :meth:`ApiClient.refresh_session`."""
        session = _build(models.SessionPayload, (await self._post("/auth/refresh", {}))["data"])
        if session.csrf_token:
            self._csrf = str(session.csrf_token)
        return session

    async def list_sessions(self) -> list[models.ActiveSessionPayload]:
        """Live sessions — see :meth:`ApiClient.list_sessions`."""
        return [_build(models.ActiveSessionPayload, s)
                for s in (await self._get("/auth/sessions"))["data"]]

    async def revoke_session(self, session_id: str) -> models.RevokeSessionPayload:
        """End one session — see :meth:`ApiClient.revoke_session`."""
        out = _build(models.RevokeSessionPayload,
                     (await self._post(f"/auth/sessions/{session_id}/revoke", {}))["data"])
        if out.current:
            self._drop_session()
        return out

    async def logout(self, session: Optional[models.SessionPayload] = None
                     ) -> models.LogoutResponse:
        """End the current session — see :meth:`ApiClient.logout`."""
        session = session or await self.get_session()
        data = (await self._post("/auth/logout", {}, _session_binding(session)))["data"]
        out = _build(models.LogoutResponse, data)
        self._drop_session()
        return out

    async def logout_all(self) -> models.LogoutAllResponse:
        """End every session — see :meth:`ApiClient.logout_all`."""
        out = _build(models.LogoutAllResponse, (await self._post("/auth/logout-all", {}))["data"])
        self._drop_session()
        return out

    async def get_ws_ticket(self) -> models.AuthTokenPayload:
        """Websocket ticket — see :meth:`ApiClient.get_ws_ticket`."""
        return _build(models.AuthTokenPayload, (await self._get("/auth/ws-ticket"))["data"])

    async def referral_status(self) -> models.ReferralStatusResponse:
        """Referral picture — see :meth:`ApiClient.referral_status`."""
        return _build(models.ReferralStatusResponse,
                      (await self._get("/referral/status"))["data"])

    async def my_referral_code(self) -> Optional[str]:
        """This account's shareable code — see :meth:`ApiClient.my_referral_code`."""
        data = (await self._get("/referral/my-code"))["data"]
        return _build(models.ReferralMyCodeResponse, data).code

    async def redeem_code(self, code: str) -> models.ReferralRedeemResponse:
        """Redeem an optional referral code — see :meth:`ApiClient.redeem_code`."""
        data = (await self._post("/referral/redeem", {"code": code}))["data"]
        out = _build(models.ReferralRedeemResponse, data)
        token = data.get("token")
        if token:
            self._token = token
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
                        amount_in: Any = None,
                        pool_key: Optional[str] = None) -> models.RouteResultDoc:
        """Best route between two tokens — see :meth:`ApiClient.get_route`.

        As on the sync client, *amount_in* is stringified onto the query and the
        API reads it as a canonical decimal amount, not base units.
        """
        params: dict[str, Any] = {"token_in": token_in, "token_out": token_out}
        if amount_in is not None:
            params["amount_in"] = str(amount_in)
        if pool_key is not None:
            params["pool_key"] = pool_key
        return _build(models.RouteResultDoc, (await self._get("/route", params))["data"])

    async def get_ohlcv(self, pool_key: str, *, granularity: str,
                        from_ts: int, to_ts: int) -> list[models.OhlcvDoc]:
        """Candles for one pool — see :meth:`ApiClient.get_ohlcv`."""
        data = (await self._get(f"/pools/{pool_key}/ohlcv",
                                {"granularity": granularity, "from": from_ts,
                                 "to": to_ts}))["data"]
        return [_build(models.OhlcvDoc, o) for o in data]

    # ── 2026-09 API surface (async mirrors) ────────────────────────────────

    async def list_api_tokens(self) -> list[models.ApiTokenRow]:
        """Durable token metadata — see :meth:`ApiClient.list_api_tokens`."""
        data = (await self._get("/api-tokens"))["data"]
        return _build(models.ApiTokenListResponse, data).tokens

    async def revoke_api_token(self, token_id: str) -> models.ApiTokenRevokeResponse:
        """Revoke a durable token — see :meth:`ApiClient.revoke_api_token`."""
        return _build(models.ApiTokenRevokeResponse,
                      (await self._delete(f"/api-tokens/{token_id}"))["data"])

    async def referral_settings(self) -> models.ReferralSettingsResponse:
        """Issuance settings — see :meth:`ApiClient.referral_settings`."""
        return _build(models.ReferralSettingsResponse,
                      (await self._get("/referral/settings"))["data"])

    async def update_referral_settings(self, *, codes_per_user: int,
                                       max_users: Optional[int] = None
                                       ) -> models.ReferralSettingsResponse:
        """Change issuance settings — see :meth:`ApiClient.update_referral_settings`."""
        body: dict[str, Any] = {"codes_per_user": codes_per_user}
        if max_users is not None:
            body["max_users"] = max_users
        return _build(models.ReferralSettingsResponse,
                      (await self._put("/referral/settings", body))["data"])

    async def list_referral_codes(self, *, limit: Optional[int] = None,
                                  offset: Optional[int] = None) -> models.ReferralListResponse:
        """Issued codes — see :meth:`ApiClient.list_referral_codes`."""
        params = {k: v for k, v in (("limit", limit), ("offset", offset)) if v is not None}
        return _build(models.ReferralListResponse,
                      (await self._get("/referral/codes", params or None))["data"])

    async def generate_referral_codes(self, count: int = 1) -> list[str]:
        """Issue new codes — see :meth:`ApiClient.generate_referral_codes`."""
        data = (await self._post("/referral/generate", {"count": count}))["data"]
        return list(_build(models.ReferralGenerateResponse, data).codes)

    async def report_referral_activity(self, *, action: str, tx_id: str,
                                       metadata: Any = None
                                       ) -> models.ReferralActivityResponse:
        """Attribute an action — see :meth:`ApiClient.report_referral_activity`."""
        body: dict[str, Any] = {"action": action, "tx_id": tx_id}
        if metadata is not None:
            body["metadata"] = metadata
        return _build(models.ReferralActivityResponse,
                      (await self._post("/referral/activity", body))["data"])

    async def report_referral_swap_claim(self, *, code: str, blinded_address: str
                                         ) -> models.ReferralSwapClaimResponse:
        """Link a swap to a code — see :meth:`ApiClient.report_referral_swap_claim`."""
        return _build(models.ReferralSwapClaimResponse,
                      (await self._post("/referral/swap-claims",
                                        {"code": code, "blinded_address": blinded_address}))["data"])

    async def report_referral_address_batch(self, *, code: str, blinded_addresses: list[str]
                                            ) -> models.ReferralAddressBatchResponse:
        """Batch swap attribution — see :meth:`ApiClient.report_referral_address_batch`."""
        return _build(models.ReferralAddressBatchResponse,
                      (await self._post("/referral/address-batches",
                                        {"code": code, "blinded_addresses": blinded_addresses}))["data"])

    async def get_pool(self, pool_key: str) -> models.PoolWithStatsDoc:
        """One pool with stats — see :meth:`ApiClient.get_pool`."""
        return _build(models.PoolWithStatsDoc, (await self._get(f"/pools/{pool_key}"))["data"])

    async def get_pool_stats(self, pool_key: str) -> models.PoolStats24hDoc:
        """Rolling pool analytics — see :meth:`ApiClient.get_pool_stats`."""
        return _build(models.PoolStats24hDoc,
                      (await self._get(f"/pools/{pool_key}/stats"))["data"])

    async def get_pool_stats_batch(self, pool_keys: list[str]
                                   ) -> dict[str, models.PoolStats24hDoc]:
        """Stats for many pools — see :meth:`ApiClient.get_pool_stats_batch`."""
        if not pool_keys:
            return {}
        data = (await self._get("/pools/stats", {"keys": ",".join(pool_keys)}))["data"]
        return {k: _build(models.PoolStats24hDoc, v) for k, v in data.items()}

    async def get_liquidity_distribution(self, pool_key: str) -> list[models.TickLiquidityDoc]:
        """Pool depth per tick — see :meth:`ApiClient.get_liquidity_distribution`."""
        data = (await self._get(f"/pools/{pool_key}/liquidity-distribution"))["data"]
        return [_build(models.TickLiquidityDoc, t) for t in data]

    async def get_compliance(self) -> models.GlobalConfigStatus:
        """Deployment switches — see :meth:`ApiClient.get_compliance`."""
        return _build(models.GlobalConfigStatus, (await self._get("/compliance"))["data"])

    async def get_token_compliance(self, token_id: str) -> models.TokenComplianceStatus:
        """Token allow/pause state — see :meth:`ApiClient.get_token_compliance`."""
        return _build(models.TokenComplianceStatus,
                      (await self._get(f"/compliance/tokens/{token_id}"))["data"])

    async def get_pair_compliance(self, token0: str, token1: str) -> models.PairComplianceStatus:
        """Pair pause state — see :meth:`ApiClient.get_pair_compliance`."""
        return _build(models.PairComplianceStatus,
                      (await self._get(f"/compliance/pairs/{token0}/{token1}"))["data"])

    async def get_pool_trades(self, pool_key: str, *, limit: Optional[int] = None,
                              offset: Optional[int] = None,
                              trade_type: Optional[str] = None) -> list[models.PoolTradeDoc]:
        """Recent fills — see :meth:`ApiClient.get_pool_trades`."""
        params = {k: v for k, v in (("limit", limit), ("offset", offset),
                                    ("trade_type", trade_type)) if v is not None}
        data = (await self._get(f"/pools/{pool_key}/trades", params or None))["data"]
        return [_build(models.PoolTradeDoc, t) for t in data]

    async def get_initialized_ticks(self, pool_key: str) -> list[int]:
        """Initialized ticks — see :meth:`ApiClient.get_initialized_ticks`."""
        data = (await self._get(f"/pools/{pool_key}/initialized-ticks"))["data"]
        return [int(t) for t in data]

    async def get_fee_tiers(self) -> list[models.FeeTierDoc]:
        """Fee tiers — see :meth:`ApiClient.get_fee_tiers`."""
        return [_build(models.FeeTierDoc, t) for t in (await self._get("/fee-tiers"))["data"]]

    async def get_protocol_state(self, *, minimum_revision: Optional[int] = None
                                 ) -> models.ProtocolStateResponse:
        """Protocol config + indexer freshness — see :meth:`ApiClient.get_protocol_state`."""
        params = {"minimum_revision": minimum_revision} if minimum_revision is not None else None
        return _build(models.ProtocolStateResponse, await self._get("/protocol/state", params))

    async def get_positions(self, *, limit: Optional[int] = None,
                            offset: Optional[int] = None) -> list[models.PositionDoc]:
        """The session's indexed positions — see :meth:`ApiClient.get_positions`."""
        params = {k: v for k, v in (("limit", limit), ("offset", offset)) if v is not None}
        return [_build(models.PositionDoc, p)
                for p in (await self._get("/positions", params or None))["data"]]

    async def get_unclaimed(self) -> models.UnclaimedPayloadDoc:
        """Collectable swaps and owed positions — see :meth:`ApiClient.get_unclaimed`."""
        return _build(models.UnclaimedPayloadDoc, (await self._get("/unclaimed"))["data"])

    async def get_route_topology(self) -> models.RouteTopologyDoc:
        """Routable token graph — see :meth:`ApiClient.get_route_topology`."""
        return _build(models.RouteTopologyDoc, (await self._get("/route/topology"))["data"])

    async def get_rebalance_state(self, pool_key: str, *, tick_lower: int, tick_upper: int,
                                  old_liquidity: int, mint_tick_lower: int,
                                  mint_tick_upper: int) -> models.RebalanceState:
        """Indexer rebalance snapshot — see :meth:`ApiClient.get_rebalance_state`."""
        params = {"tick_lower": tick_lower, "tick_upper": tick_upper,
                  "old_liquidity": str(old_liquidity),
                  "mint_tick_lower": mint_tick_lower, "mint_tick_upper": mint_tick_upper}
        return _build(models.RebalanceState,
                      (await self._get(f"/pools/{pool_key}/rebalance-state", params))["data"])
