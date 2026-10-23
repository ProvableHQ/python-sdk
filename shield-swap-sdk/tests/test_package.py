import aleo_shield_swap


def test_version():
    assert aleo_shield_swap.__version__ == "0.8.0"


def test_lifecycle_exports():
    import aleo_shield_swap as pkg
    for name in ("Profile", "Journal", "OnboardReport", "SessionStatus",
                 "PositionView", "SwapBatchReport", "CollectReport",
                 "StageOutcome", "NotAuthenticatedError",
                 "NotFundedError", "AirdropPendingError",
                 "AirdropRateLimitedError", "CredentialsMissingError",
                 "blinded_identity_at", "REGISTRATION_STAGES"):
        assert hasattr(pkg, name), name
        assert name in pkg.__all__, name


def test_agent_guide_ships_in_the_package():
    import aleo_shield_swap as pkg
    guide = pkg.agent_guide()
    assert "Tier 1" in guide and "conversation pattern" in guide
    from importlib.resources import files
    skill = files("aleo_shield_swap").joinpath("skills/SKILL.md").read_text()
    assert "python -m aleo_shield_swap" in skill


def test_import_guard_names_the_aleo_sdk_upgrade(monkeypatch):
    import sys

    import pytest

    import aleo_shield_swap
    aleo_shield_swap._require_aleo_sdk()                               # the installed sdk has aleo.inventory
    monkeypatch.setitem(sys.modules, "aleo.inventory", None)           # simulate an aleo-sdk without it
    with pytest.raises(ImportError, match=r'needs aleo-sdk>=0\.8\.0 \(found .*\); run: pip install -U "aleo-sdk>=0\.8\.0"'):
        aleo_shield_swap._require_aleo_sdk()
