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
    return getattr(ctx.dex.api, "_token", None) is not None


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


def _creds_done(ctx: _Ctx) -> bool:
    c = ctx.profile.credentials
    return bool(c.get("dps_api_key") and c.get("dps_consumer_id"))


def _creds_run(ctx: _Ctx) -> str:
    # No provisioning endpoint exists yet (spec blocker) — import from env.
    # When the API grows one, this function body is the only change.
    key = os.environ.get("ALEO_E2E_API_KEY")
    cid = os.environ.get("ALEO_E2E_CONSUMER_ID")
    if not (key and cid):
        raise CredentialsMissingError()
    ctx.profile.save_credentials(dps_api_key=key, dps_consumer_id=cid)
    refresh = getattr(ctx.dex, "_refresh_credentials", None)
    if refresh is not None:
        refresh()                         # live facade picks up the new key
    return "delegated-proving credentials imported from env"


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
