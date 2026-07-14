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
