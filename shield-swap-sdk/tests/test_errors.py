from aleo import AleoError

from aleo_shield_swap.errors import (
    DexApiError,
    InsufficientRecordsError,
    InvalidFeeTierError,
    PoolNotFoundError,
    PoolNotInitializedError,
    ShieldSwapError,
    SwapOutputNotFinalizedError,
)


def test_hierarchy():
    for exc in (SwapOutputNotFinalizedError("1field"), PoolNotFoundError("x"),
                PoolNotInitializedError("x"), InsufficientRecordsError("x"),
                InvalidFeeTierError("x"), DexApiError(500, "boom")):
        assert isinstance(exc, ShieldSwapError)
        assert isinstance(exc, AleoError)


def test_messages():
    e = SwapOutputNotFinalizedError("42field")
    assert "42field" in str(e) and "finalized" in str(e)
    assert e.swap_id == "42field"
    d = DexApiError(404, "not found")
    assert d.status == 404 and d.body == "not found"


def test_lifecycle_errors_teach_the_fix():
    from aleo_shield_swap.errors import (
        AirdropPendingError, AirdropRateLimitedError, CredentialsMissingError,
        NotAuthenticatedError, NotFundedError, NotRedeemedError, ShieldSwapError,
    )
    assert "dex.onboard(" in str(NotAuthenticatedError())
    assert "invite" in str(NotRedeemedError())
    assert "dex.onboard(" in str(NotRedeemedError())
    assert "airdrop" in str(NotFundedError())
    assert "dex.status()" in str(AirdropPendingError("job1"))
    assert AirdropPendingError("job1").job_id == "job1"
    assert "15 minutes" in str(AirdropRateLimitedError())
    assert "ALEO_E2E_API_KEY" in str(CredentialsMissingError())
    for cls in (NotAuthenticatedError, NotRedeemedError, NotFundedError,
                AirdropRateLimitedError, CredentialsMissingError):
        assert issubclass(cls, ShieldSwapError)
