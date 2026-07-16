import pytest

from aleo_shield_swap.errors import (AirdropRateLimitedError,
                                     CredentialsMissingError, NotRedeemedError)
from aleo_shield_swap.journal import Journal
from aleo_shield_swap.lifecycle import REGISTRATION_STAGES, run_onboard
from aleo_shield_swap.profile import Profile


class _Tok:
    def __init__(self, wrapper_program):
        self.wrapper_program = wrapper_program


class _StubApi:
    def __init__(self, has_access=False):
        self._token = None
        self.has_access = has_access
        self.redeemed_with = None
        self.airdrops = 0

    def authenticate(self, address, sign):
        self._token = "jwt"
        return "jwt"

    def access_status(self):
        return type("S", (), {"has_access": self.has_access})()

    def redeem_code(self, code):
        self.redeemed_with = code
        self.has_access = True
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
    report = run_onboard(dex, profile, invite_code="CODE", poll_seconds=0)
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
    api = _StubApi(has_access=True)
    api._token = "oldjwt"
    dex = _StubDex(api, {"waleo.aleo": 7}, funded_from_start=True)
    report = run_onboard(dex, profile)
    assert all(o.action == "skipped" for o in report.outcomes)
    assert api.airdrops == 0


def test_redeem_without_code_raises_instructively(profile, dps_env):
    api = _StubApi(has_access=False)
    api._token = "jwt"                                # authenticated already
    dex = _StubDex(api, {"waleo.aleo": 7})
    with pytest.raises(NotRedeemedError):
        run_onboard(dex, profile)                     # no invite_code


def test_provisioning_failure_raises_instructively(profile, monkeypatch):
    monkeypatch.delenv("ALEO_E2E_API_KEY", raising=False)
    monkeypatch.delenv("ALEO_E2E_CONSUMER_ID", raising=False)
    monkeypatch.setattr("aleo_shield_swap.lifecycle.provision_provable_credentials",
                        lambda endpoint, username: (_ for _ in ()).throw(
                            CredentialsMissingError("POST /consumers -> 500")))
    api = _StubApi(has_access=True)
    api._token = "jwt"
    dex = _StubDex(api, {"waleo.aleo": 7})
    with pytest.raises(CredentialsMissingError, match="consumers"):
        run_onboard(dex, profile)


def test_credentials_auto_provision_both_systems(profile, monkeypatch):
    monkeypatch.delenv("ALEO_E2E_API_KEY", raising=False)
    monkeypatch.delenv("ALEO_E2E_CONSUMER_ID", raising=False)
    monkeypatch.setattr("aleo_shield_swap.lifecycle.provision_provable_credentials",
                        lambda endpoint, username: ("pk-auto", "cid-auto"))
    api = _StubApi(has_access=True)
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
    api = _StubApi(has_access=True)
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
    run_onboard(dex, profile, invite_code="C", poll_seconds=0)
    names = [e["name"] for e in Journal(profile.journal_path).events()
             if e["type"] == "stage"]
    assert names == [s.name for s in REGISTRATION_STAGES]


def test_credentials_stage_refreshes_live_facade(profile, dps_env):
    refreshed = []

    class _RefreshingDex(_StubDex):
        def _refresh_credentials(self):
            refreshed.append(True)

    api = _StubApi(has_access=True)
    api._token = "jwt"
    dex = _RefreshingDex(api, {"waleo.aleo": 7}, funded_from_start=True)
    run_onboard(dex, profile)
    assert refreshed == [True]            # live provider picked up the new key
