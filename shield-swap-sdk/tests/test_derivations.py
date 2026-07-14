"""Pool/tick key derivations vs vectors from the TS SDK's keys.test.ts."""
import pytest

from aleo_shield_swap.derivations import derive_pool_key, derive_tick_key

# Vectors copied verbatim from aleo-viem packages/shield-swap/test/keys.test.ts
TOKEN0 = "1234567890123456789field"
TOKEN1 = "9876543210987654321field"
POOL_KEY = "5004171258545595848890767719949996982906438837519254032156408929642095152812field"
TICK_KEY = "1831124990376748452345532981319547440239544385602656748610276036461414053413field"


def test_pool_key_vector():
    assert derive_pool_key(TOKEN0, TOKEN1, 3000) == POOL_KEY


def test_pool_key_is_order_independent():
    assert derive_pool_key(TOKEN1, TOKEN0, 3000) == POOL_KEY


def test_pool_key_accepts_bare_literals():
    assert derive_pool_key(TOKEN0.removesuffix("field"),
                           TOKEN1.removesuffix("field"), 3000) == POOL_KEY


def test_pool_key_fee_matters():
    assert derive_pool_key(TOKEN0, TOKEN1, 500) != POOL_KEY
    with pytest.raises(ValueError):
        derive_pool_key(TOKEN0, TOKEN1, 2**16)


def test_tick_key_vector():
    assert derive_tick_key(POOL_KEY, -600) == TICK_KEY
    assert derive_tick_key(POOL_KEY.removesuffix("field"), -600) == TICK_KEY
    assert derive_tick_key(POOL_KEY, 600) != TICK_KEY
    with pytest.raises(ValueError):
        derive_tick_key(POOL_KEY, 2**31)
