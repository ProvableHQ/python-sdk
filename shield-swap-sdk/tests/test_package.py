import aleo_shield_swap


def test_version():
    assert aleo_shield_swap.__version__ == "0.2.1"


def test_lifecycle_exports():
    import aleo_shield_swap as pkg
    for name in ("Profile", "Journal", "OnboardReport", "SessionStatus",
                 "PositionView", "SwapBatchReport", "CollectReport",
                 "StageOutcome", "NotAuthenticatedError", "NotRedeemedError",
                 "NotFundedError", "AirdropPendingError",
                 "AirdropRateLimitedError", "CredentialsMissingError",
                 "blinded_identity_at", "REGISTRATION_STAGES"):
        assert hasattr(pkg, name), name
        assert name in pkg.__all__, name
