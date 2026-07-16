"""Stage-list-driven onboarding — the ONLY definition of the registration flow.

Registration steps change over time.  Each stage is a self-describing
(name, is_done, run) triple; adding/removing/reordering a step is an edit
to ``REGISTRATION_STAGES`` and nothing else — reports, journals, docs, and
tools all derive from the list.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .errors import (
    AirdropPendingError,
    AirdropRateLimitedError,
    CredentialsMissingError,
    NotAuthenticatedError,
    NotRedeemedError,
)
from .journal import Journal
from .types import OnboardReport, StageOutcome


class _Ctx:
    """Mutable state threaded through the stages of one onboard() run."""

    def __init__(self, dex: Any, profile: Any, invite_code: Optional[str],
                 poll_seconds: float, timeout_seconds: float) -> None:
        self.dex = dex
        self.profile = profile
        self.invite_code = invite_code
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.journal = Journal(profile.journal_path)
        self._wrappers: Optional[list[str]] = None

    def wrapper_programs(self) -> list[str]:
        """Airdroppable token programs, from the live token registry."""
        if self._wrappers is None:
            self._wrappers = [t.wrapper_program for t in self.dex.api.get_tokens()
                              if t.wrapper_program]
        return self._wrappers

    def funded(self) -> bool:
        balances = self.dex.get_private_balances(self.wrapper_programs())
        return any(v > 0 for v in balances.values())


@dataclass
class Stage:
    name: str
    is_done: Callable[[_Ctx], bool]
    run: Callable[[_Ctx], str]          # returns a one-line detail


# ── Stages ────────────────────────────────────────────────────────────────────

def _auth_done(ctx: _Ctx) -> bool:
    if getattr(ctx.dex.api, "_token", None) is None:
        return False
    try:                                  # a stored-but-expired JWT is not auth
        ctx.dex.api.access_status()
        return True
    except NotAuthenticatedError:
        return False


def _auth_run(ctx: _Ctx) -> str:
    import aleo
    net = getattr(aleo, ctx.profile.network)
    pk = net.PrivateKey.from_string(ctx.profile.private_key)
    jwt = ctx.dex.api.authenticate(ctx.profile.address,
                                   lambda msg: str(pk.sign(msg.encode())))
    ctx.profile.save_credentials(jwt=jwt)
    return "authenticated (24h JWT)"


def _redeem_done(ctx: _Ctx) -> bool:
    return bool(ctx.dex.api.access_status().has_access)


def _redeem_run(ctx: _Ctx) -> str:
    if not ctx.invite_code:
        raise NotRedeemedError()
    out = ctx.dex.api.redeem_code(ctx.invite_code)
    ctx.profile.save_credentials(jwt=getattr(out, "token", None))
    return f"invite redeemed ({out.status})"


def provision_provable_credentials(endpoint: str, username: str) -> tuple[str, str]:
    """Create a Provable API consumer + key (``POST /consumers``, keyless).

    Returns ``(api_key, consumer_id)`` — the pair the scanner and delegated
    proving authenticate with.
    """
    import requests

    resp = requests.post(f"{endpoint.rstrip('/')}/consumers",
                         json={"username": username}, timeout=30.0)
    if not 200 <= resp.status_code < 300:
        raise CredentialsMissingError(
            f"POST /consumers -> {resp.status_code}: {resp.text[:120]}")
    data = resp.json()
    return data["key"], data["consumer"]["id"]


def _creds_done(ctx: _Ctx) -> bool:
    c = ctx.profile.credentials
    return bool(c.get("dps_api_key") and c.get("dps_consumer_id")
                and c.get("dex_api_token"))


def _creds_run(ctx: _Ctx) -> str:
    """Register BOTH credential systems, unless already stored.

    Provable API (scanner + delegated proving): imported from
    ``ALEO_E2E_API_KEY``/``ALEO_E2E_CONSUMER_ID`` when set, otherwise
    provisioned via ``POST /consumers``.  Shield-swap API: a durable
    ``ss_…`` token minted via ``POST /api-tokens`` (the 24h session JWT
    stays for the ``/access/*`` tier).
    """
    details: list[str] = []
    creds = ctx.profile.credentials
    if not (creds.get("dps_api_key") and creds.get("dps_consumer_id")):
        key = os.environ.get("ALEO_E2E_API_KEY")
        cid = os.environ.get("ALEO_E2E_CONSUMER_ID")
        if key and cid:
            details.append("Provable credentials imported from env")
        else:
            key, cid = provision_provable_credentials(
                ctx.profile.endpoint, f"shield-swap-{ctx.profile.address}")
            details.append("Provable consumer + API key provisioned")
        ctx.profile.save_credentials(dps_api_key=key, dps_consumer_id=cid)
    if not ctx.profile.credentials.get("dex_api_token"):
        tok = ctx.dex.api.create_api_token(
            f"shield-swap-profile-{ctx.profile.address[:16]}")
        ctx.profile.save_credentials(dex_api_token=tok.token)
        details.append("durable DEX API token minted")
    refresh = getattr(ctx.dex, "_refresh_credentials", None)
    if refresh is not None:
        refresh()                         # live facade picks up the new key
    return "; ".join(details) or "already stored"


def _airdrop_done(ctx: _Ctx) -> bool:
    return ctx.funded()


def _airdrop_run(ctx: _Ctx) -> str:
    try:
        start = ctx.dex.api.request_airdrop(ctx.profile.address)
    except AirdropRateLimitedError:
        return "rate-limited (claimed <15min ago) — waiting on records"
    deadline = time.monotonic() + ctx.timeout_seconds
    while True:
        job = ctx.dex.api.get_airdrop_job(start.job_id)
        if job.status == "complete":
            return f"airdrop complete ({job.total} tokens)"
        if time.monotonic() >= deadline:
            raise AirdropPendingError(start.job_id)
        time.sleep(ctx.poll_seconds)


def _funded_done(ctx: _Ctx) -> bool:
    return ctx.funded()


def _funded_run(ctx: _Ctx) -> str:
    deadline = time.monotonic() + ctx.timeout_seconds
    while True:
        if ctx.funded():
            return "private records scanned and spendable"
        if time.monotonic() >= deadline:
            raise AirdropPendingError()
        time.sleep(ctx.poll_seconds)


REGISTRATION_STAGES: list[Stage] = [
    Stage("authenticate", _auth_done, _auth_run),
    Stage("redeem", _redeem_done, _redeem_run),
    Stage("credentials", _creds_done, _creds_run),
    Stage("airdrop", _airdrop_done, _airdrop_run),
    Stage("funded", _funded_done, _funded_run),
]


def run_onboard(dex: Any, profile: Any, invite_code: Optional[str] = None,
                poll_seconds: float = 5.0,
                timeout_seconds: float = 600.0) -> OnboardReport:
    """Run every not-yet-done registration stage, in order, and report.

    Idempotent: already-satisfied stages are skipped, so calling this on a
    registered, funded account is a no-op that says so.
    """
    ctx = _Ctx(dex, profile, invite_code, poll_seconds, timeout_seconds)
    outcomes: list[StageOutcome] = []
    for stage in REGISTRATION_STAGES:
        if stage.is_done(ctx):
            outcome = StageOutcome(stage.name, "skipped", "already satisfied")
        else:
            outcome = StageOutcome(stage.name, "ran", stage.run(ctx))
        ctx.journal.record_stage(outcome.name, outcome.action, outcome.detail)
        outcomes.append(outcome)
    return OnboardReport(profile.address, outcomes, funded=ctx.funded())
