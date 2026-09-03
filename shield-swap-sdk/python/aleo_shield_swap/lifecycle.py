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
    NotFundedError,
    NotAuthenticatedError,
)
from .journal import Journal
from .types import OnboardReport, StageOutcome


class _Ctx:
    """Mutable state threaded through the stages of one onboard() run."""

    def __init__(self, dex: Any, profile: Any, referral_code: Optional[str],
                 poll_seconds: float, timeout_seconds: float) -> None:
        self.dex = dex
        self.profile = profile
        self.referral_code = referral_code
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.journal = Journal(profile.journal_path)
        self._wrappers: Optional[list[str]] = None

    def wrapper_programs(self) -> list[str]:
        """Record-funding token programs (underlying for wrapped assets),
        from the live token registry — where airdropped records land."""
        if self._wrappers is None:
            self._wrappers = [t.underlying_program or t.amm_token_program
                              for t in self.dex.api.get_tokens()
                              if t.underlying_program or t.amm_token_program]
        return self._wrappers

    def funded(self) -> bool:
        """True once any wrapper-underlying program holds a private balance.

        Reads private balances, so it needs the account's view key and costs a
        record scan per wrapper program. Any non-zero balance counts as funded.
        """
        balances = self.dex.get_private_balances(self.wrapper_programs())
        return any(v > 0 for v in balances.values())


@dataclass
class Stage:
    """One resumable step of registration: how to tell it is done, and how to do it.

    ``is_done`` is checked before ``run``, so a stage already satisfied is skipped
    — that is what makes onboarding idempotent across sessions. Both receive the
    shared context; ``run`` returns a one-line detail for the journal.
    """

    name: str
    is_done: Callable[[_Ctx], bool]
    run: Callable[[_Ctx], str]          # returns a one-line detail


# ── Stages ────────────────────────────────────────────────────────────────────

def _auth_done(ctx: _Ctx) -> bool:
    # A credential may be a bearer token OR a cookie session (staging).
    authed = getattr(ctx.dex.api, "is_authenticated",
                     getattr(ctx.dex.api, "_token", None) is not None)
    if not authed:
        return False
    try:                                  # a stored-but-expired credential is not auth
        ctx.dex.api.access_status()
        return True
    except NotAuthenticatedError:
        return False


def _auth_run(ctx: _Ctx) -> str:
    import aleo
    net = getattr(aleo, ctx.profile.network)
    pk = net.PrivateKey.from_string(ctx.profile.private_key)
    ctx.dex.api.authenticate(ctx.profile.address,
                             lambda msg: str(pk.sign(msg.encode())))
    # Only a body-JWT (legacy deployments) is worth persisting — a cookie
    # session's CSRF token is useless in a new process and would shadow the
    # durable ss_ token if saved as "jwt".
    jwt = getattr(ctx.dex.api, "_token", None)
    if jwt:
        ctx.profile.save_credentials(jwt=jwt)
        return "authenticated (JWT)"
    return "authenticated (cookie session)"


def _referral_done(ctx: _Ctx) -> bool:
    # Optional attribution, never a gate: nothing to do without a code, and
    # nothing to do once the account already has a referrer (the API records
    # exactly one).  Access itself comes from authentication alone.
    if not ctx.referral_code:
        return True
    return bool(ctx.dex.api.referral_status().referred_by)


def _referral_run(ctx: _Ctx) -> str:
    out = ctx.dex.api.redeem_code(ctx.referral_code)
    token = getattr(out, "token", None)
    if token:                             # legacy deployments only
        ctx.profile.save_credentials(jwt=token)
    return f"referral code redeemed ({out.status})"


def provision_provable_credentials(endpoint: str, username: str) -> tuple[str, str]:
    """Create a Provable API consumer + key (``POST /consumers``, keyless).

    Returns ``(api_key, consumer_id)`` — the pair the scanner and delegated
    proving authenticate with.  Usernames are labels with a UNIQUE
    constraint server-side; on a 409 collision, retry once with a random
    suffix (the key/id pair is what matters, not the name).
    """
    import secrets

    import requests

    for name in (username, f"{username}-{secrets.token_hex(4)}"):
        resp = requests.post(f"{endpoint.rstrip('/')}/consumers",
                             json={"username": name}, timeout=30.0)
        if 200 <= resp.status_code < 300:
            data = resp.json()
            return data["key"], data["consumer"]["id"]
        if resp.status_code != 409:
            break
    raise CredentialsMissingError(
        f"POST /consumers -> {resp.status_code}: {resp.text[:120]}")


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
    # /airdrop and /airdrop/{job_id} exist on testnet only — the mainnet API
    # publishes neither, so requesting one there 404s. Say so instead, because
    # the remedy is the caller funding the account, not a retry.
    if ctx.profile.network != "testnet":
        raise NotFundedError(
            f"no faucet on {ctx.profile.network}: the airdrop endpoints are "
            f"testnet-only. Fund {ctx.profile.address} with the tokens you "
            "intend to trade, then re-run onboard()."
        )
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
    Stage("referral", _referral_done, _referral_run),
    Stage("credentials", _creds_done, _creds_run),
    Stage("airdrop", _airdrop_done, _airdrop_run),
    Stage("funded", _funded_done, _funded_run),
]


def run_onboard(dex: Any, profile: Any, referral_code: Optional[str] = None,
                poll_seconds: float = 5.0,
                timeout_seconds: float = 600.0) -> OnboardReport:
    """Run every not-yet-done registration stage, in order, and report.

    Idempotent: already-satisfied stages are skipped, so calling this on a
    registered, funded account is a no-op that says so.  *referral_code* is
    optional attribution — access never depends on it.
    """
    ctx = _Ctx(dex, profile, referral_code, poll_seconds, timeout_seconds)
    outcomes: list[StageOutcome] = []
    for stage in REGISTRATION_STAGES:
        if stage.is_done(ctx):
            outcome = StageOutcome(stage.name, "skipped", "already satisfied")
        else:
            outcome = StageOutcome(stage.name, "ran", stage.run(ctx))
        ctx.journal.record_stage(outcome.name, outcome.action, outcome.detail)
        outcomes.append(outcome)
    return OnboardReport(profile.address, outcomes, funded=ctx.funded())
