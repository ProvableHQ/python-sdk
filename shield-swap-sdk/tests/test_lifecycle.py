import pytest

from aleo_shield_swap.errors import (AirdropRateLimitedError,
                                     CredentialsMissingError)
from aleo_shield_swap.journal import Journal
from aleo_shield_swap.lifecycle import REGISTRATION_STAGES, run_onboard
from aleo_shield_swap.profile import Profile


class _Tok:
    def __init__(self, program):
        self.amm_token_program = program
        self.underlying_program = program


class _StubApi:
    # Access is granted by authentication alone (has_access is True for any
    # authenticated account); a referral code is optional attribution.
    def __init__(self, referred_by=None):
        self._token = None
        self.referred_by = referred_by
        self.redeemed_with = None
        self.airdrops = 0

    def authenticate(self, address, sign):
        self._token = "jwt"
        return "jwt"

    def referral_status(self):
        return type("RS", (), {"has_access": self._token is not None,
                               "referred_by": self.referred_by,
                               "my_code": "MYCODE", "code": None})()

    def redeem_code(self, code):
        self.redeemed_with = code
        self.referred_by = "aleo1referrer"
        return type("R", (), {"code": code, "status": "redeemed",
                              "token": "jwt2"})()

    def request_airdrop(self, address):
        self.airdrops += 1
        return type("A", (), {"job_id": "j1", "status": "running"})()

    def get_airdrop_job(self, job_id):
        return type("J", (), {"status": "complete", "total": 3, "results": []})()

    def get_tokens(self):
        return [_Tok("waleo.aleo"), _Tok("wusdcx.aleo"), _Tok("weth.aleo")]

    def create_api_token(self, name, expires_in_days=None):
        return type("T", (), {"token": f"ss_minted_{name[:12]}"})()

    # Durable-token bookkeeping the credentials stage reclaims from.
    tokens: list = []
    revoked: list = []

    def list_api_tokens(self):
        return list(self.tokens)

    def revoke_api_token(self, token_id):
        self.revoked.append(token_id)
        for row in self.tokens:
            if row.id == token_id:
                row.revoked_at = "now"
        return type("R", (), {"id": token_id, "revoked": True})()


class _StubDex:
    def __init__(self, api, balances, funded_from_start=False):
        self.api = api
        self._balances = balances
        self._funded_from_start = funded_from_start

    def get_private_balances(self, programs, account=None):
        if not (self._funded_from_start or self.api.airdrops):
            return {p: 0 for p in programs}       # records land with the airdrop
        return {p: self._balances.get(p, 0) for p in programs}


@pytest.fixture
def profile(tmp_path):
    # Real keygen: the authenticate stage parses the key with the native
    # PrivateKey type, so a fake string won't do.
    return Profile.load_or_create(tmp_path / "home")


@pytest.fixture
def dps_env(monkeypatch):
    monkeypatch.setenv("ALEO_E2E_API_KEY", "k")
    monkeypatch.setenv("ALEO_E2E_CONSUMER_ID", "c")


def test_fresh_account_runs_every_stage(profile, dps_env):
    api = _StubApi()
    dex = _StubDex(api, {"waleo.aleo": 7})           # funded once airdrop "lands"
    report = run_onboard(dex, profile, referral_code="CODE", poll_seconds=0)
    assert [o.name for o in report.outcomes] == [s.name for s in REGISTRATION_STAGES]
    # funded is a verification stage: once the airdrop lands it is already
    # satisfied, so it reports "skipped" rather than polling.
    assert [o.action for o in report.outcomes] == ["ran"] * 4 + ["skipped"]
    assert api.redeemed_with == "CODE" and api.airdrops == 1
    assert report.funded is True
    assert profile.credentials["dps_api_key"] == "k"
    assert profile.credentials["dex_api_token"].startswith("ss_minted_")
    assert profile.credentials["jwt"] == "jwt2"       # redeem's fresh token wins


def test_registered_funded_account_is_noop(profile, dps_env):
    profile.save_credentials(jwt="oldjwt", dps_api_key="k", dps_consumer_id="c",
                             dex_api_token="ss_stored")
    api = _StubApi()
    api._token = "oldjwt"
    dex = _StubDex(api, {"waleo.aleo": 7}, funded_from_start=True)
    report = run_onboard(dex, profile)
    assert all(o.action == "skipped" for o in report.outcomes)
    assert api.airdrops == 0


def test_onboard_without_referral_code_completes(profile, dps_env):
    """Access is granted by authentication alone — no code is required.

    The referral stage is optional attribution: with nothing to redeem it is
    skipped and every later stage still runs.
    """
    api = _StubApi()
    dex = _StubDex(api, {"waleo.aleo": 7})
    report = run_onboard(dex, profile, poll_seconds=0)    # no referral_code
    by_name = {o.name: o for o in report.outcomes}
    assert "redeem" not in by_name                        # old gate is gone
    assert by_name["referral"].action == "skipped"
    assert api.redeemed_with is None
    assert by_name["credentials"].action == "ran"
    assert by_name["airdrop"].action == "ran"
    assert report.funded is True


def test_referral_code_redeemed_only_once(profile, dps_env):
    """A supplied code is redeemed; an already-referred account never
    redeems again (attribution is one-time on the server)."""
    api = _StubApi(referred_by="aleo1someone")
    api._token = "jwt"
    dex = _StubDex(api, {"waleo.aleo": 7}, funded_from_start=True)
    report = run_onboard(dex, profile, referral_code="LATECODE", poll_seconds=0)
    referral = next(o for o in report.outcomes if o.name == "referral")
    assert referral.action == "skipped"
    assert api.redeemed_with is None


def test_provisioning_failure_raises_instructively(profile, monkeypatch):
    monkeypatch.delenv("ALEO_E2E_API_KEY", raising=False)
    monkeypatch.delenv("ALEO_E2E_CONSUMER_ID", raising=False)
    monkeypatch.setattr("aleo_shield_swap.lifecycle.provision_provable_credentials",
                        lambda endpoint, username: (_ for _ in ()).throw(
                            CredentialsMissingError("POST /consumers -> 500")))
    api = _StubApi()
    api._token = "jwt"
    dex = _StubDex(api, {"waleo.aleo": 7})
    with pytest.raises(CredentialsMissingError, match="consumers"):
        run_onboard(dex, profile)


def test_credentials_auto_provision_both_systems(profile, monkeypatch):
    monkeypatch.delenv("ALEO_E2E_API_KEY", raising=False)
    monkeypatch.delenv("ALEO_E2E_CONSUMER_ID", raising=False)
    monkeypatch.setattr("aleo_shield_swap.lifecycle.provision_provable_credentials",
                        lambda endpoint, username: ("pk-auto", "cid-auto"))
    api = _StubApi()
    api._token = "jwt"
    dex = _StubDex(api, {"waleo.aleo": 7}, funded_from_start=True)
    report = run_onboard(dex, profile)
    creds_stage = next(o for o in report.outcomes if o.name == "credentials")
    assert creds_stage.action == "ran"
    assert "provisioned" in creds_stage.detail and "minted" in creds_stage.detail
    assert profile.credentials["dps_api_key"] == "pk-auto"
    assert profile.credentials["dps_consumer_id"] == "cid-auto"
    assert profile.credentials["dex_api_token"].startswith("ss_minted_")


def test_rate_limited_airdrop_with_funds_is_tolerated(profile, dps_env):
    api = _StubApi()
    api._token = "jwt"

    def limited(address):
        raise AirdropRateLimitedError()

    api.request_airdrop = limited
    dex = _StubDex(api, {"waleo.aleo": 7}, funded_from_start=True)  # already has funds
    report = run_onboard(dex, profile)
    airdrop = next(o for o in report.outcomes if o.name == "airdrop")
    assert airdrop.action == "skipped"                # funded → stage was done
    assert report.funded is True


def test_stage_progress_journaled(profile, dps_env):
    dex = _StubDex(_StubApi(), {"waleo.aleo": 7})
    run_onboard(dex, profile, referral_code="C", poll_seconds=0)
    names = [e["name"] for e in Journal(profile.journal_path).events()
             if e["type"] == "stage"]
    assert names == [s.name for s in REGISTRATION_STAGES]


def test_credentials_stage_refreshes_live_facade(profile, dps_env):
    refreshed = []

    class _RefreshingDex(_StubDex):
        def _refresh_credentials(self):
            refreshed.append(True)

    api = _StubApi()
    api._token = "jwt"
    dex = _RefreshingDex(api, {"waleo.aleo": 7}, funded_from_start=True)
    run_onboard(dex, profile)
    assert refreshed == [True]            # live provider picked up the new key


def test_airdrop_stage_refuses_on_mainnet(tmp_path):
    """The faucet endpoints are testnet-only; on mainnet say so, don't 404."""
    tmp_journal = tmp_path / "j.jsonl"
    from aleo_shield_swap.errors import NotFundedError
    from aleo_shield_swap.lifecycle import _Ctx, _airdrop_run

    class _P:
        network = "mainnet"
        address = "aleo1me"
        journal_path = tmp_journal

    ctx = _Ctx(dex=None, profile=_P(), referral_code=None,
               poll_seconds=0, timeout_seconds=0)
    try:
        _airdrop_run(ctx)
    except NotFundedError as exc:
        assert "no faucet on mainnet" in str(exc)
        assert "aleo1me" in str(exc)
    else:
        raise AssertionError("expected NotFundedError on mainnet")


def test_credentials_stage_survives_the_api_token_cap(profile, dps_env):
    """The DEX caps active durable tokens per account (5).  An account at the
    cap must still onboard — the cookie session serves this process — with
    the cap named in the stage detail, and no token stored."""
    from aleo_shield_swap.errors import DexApiError

    api = _StubApi()
    api._token = "jwt"

    def capped(name, expires_in_days=None):
        raise DexApiError(400, '{"error":"active token limit reached (5); revoke one first"}')

    api.create_api_token = capped
    dex = _StubDex(api, {"waleo.aleo": 7}, funded_from_start=True)
    report = run_onboard(dex, profile)
    creds = next(o for o in report.outcomes if o.name == "credentials")
    assert creds.action == "ran"
    assert "token limit" in creds.detail and "session" in creds.detail
    assert "dex_api_token" not in profile.credentials
    assert report.funded is True


def test_credentials_stage_reclaims_this_profiles_stale_token(profile, dps_env):
    """At the cap, the stale tokens are usually this profile's own earlier
    mints (same deterministic name, secret lost with the credentials):
    revoke the oldest of those — never anyone else's — and mint again."""
    from aleo_shield_swap.errors import DexApiError

    api = _StubApi()
    api._token = "jwt"
    name = f"shield-swap-profile-{profile.address[:16]}"

    def row(i, name_, created):
        return type("Row", (), {"id": i, "name": name_, "created_at": created,
                                "revoked_at": None})()

    api.tokens = [row("newer", name, "2026-09-01"), row("older", name, "2026-08-01"),
                  row("agent", "ss-agent-x", "2026-01-01")]
    api.revoked = []
    attempts = {"n": 0}

    def create(name_, expires_in_days=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise DexApiError(400, '{"error":"active token limit reached (5); revoke one first"}')
        return type("T", (), {"token": "ss_fresh"})()

    api.create_api_token = create
    dex = _StubDex(api, {"waleo.aleo": 7}, funded_from_start=True)
    report = run_onboard(dex, profile)
    creds = next(o for o in report.outcomes if o.name == "credentials")
    assert api.revoked == ["older"]                  # oldest same-name token only
    assert profile.credentials["dex_api_token"] == "ss_fresh"
    assert "revoked" in creds.detail and "minted" in creds.detail
