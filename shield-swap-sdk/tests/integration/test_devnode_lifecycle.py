"""Full AMM lifecycle on a devnode, through the ShieldSwap verbs.

Python analog of the TS suite's ``devnodeLifecycle.actions.e2e.test.ts``: a
non-admin user creates two pools (same pair, two fee tiers), mints and
resizes positions, swaps both directions and claims the outputs, collects
earnings, and burns out — with the on-chain mappings asserted after every
step.  Fully hermetic: vendored programs, local devnode, no live network.

Run with::

    python -m pytest tests/integration/test_devnode_lifecycle.py -m devnode -v

Requires the ``aleo-devnode`` binary (skips otherwise).  Deployments are
always proofless (dummy verifying keys — see ``devnode_amm``).  Executions
run on one of two ladders:

* default — every execution fully proven locally (slow: SNARK parameter
  downloads + proving-key synthesis on first use)
* ``ALEO_DEVNODE_UNPROVEN=1`` — proofless executions (the devnode skips
  proof verification); fast
"""
from __future__ import annotations

import re

import pytest

from aleo_shield_swap import ShieldSwap
from aleo_shield_swap.errors import SwapOutputNotFinalizedError
from aleo_shield_swap.tick_math import MIN_TICK

from .devnode_amm import AMM_PROGRAM, AmmDevnode, setup_amm_devnode

pytestmark = pytest.mark.devnode

POOLS = [
    {"fee": 3000, "tick_spacing": 60},
    {"fee": 500, "tick_spacing": 10},
]


def _position_numbers(plaintext: str) -> dict[str, int]:
    out = {}
    for name in ("liquidity", "tokens_owed0", "tokens_owed1"):
        m = re.search(rf"{name}:\s*(\d+)u128", plaintext)
        assert m, f"no {name} in position: {plaintext}"
        out[name] = int(m.group(1))
    return out


@pytest.fixture(scope="module")
def ctx() -> AmmDevnode:
    try:
        context = setup_amm_devnode()
    except Exception as exc:
        if "aleo-devnode not found" in str(exc):
            pytest.skip(f"aleo-devnode binary not available: {exc}")
        raise
    yield context
    context.stop()


@pytest.fixture(scope="module")
def dex(ctx) -> ShieldSwap:
    dex = ShieldSwap(ctx.aleo, program=AMM_PROGRAM)
    ctx.aleo.default_account = ctx.user
    return dex


class _Journey:
    """Mutable per-pool state threaded through the ordered tests."""

    def __init__(self):
        self.pools = [dict(p) for p in POOLS]


@pytest.fixture(scope="module")
def journey() -> _Journey:
    return _Journey()


def _refresh_nft(ctx, pool_case, tx_id):
    records = ctx.records_of(ctx.user, tx_id)
    nfts = [r for r in records if "tick_lower" in r]
    assert nfts, f"no PositionNFT record in {tx_id}"
    if "nft_record" not in pool_case:
        spacing = pool_case["tick_spacing"]
        assert re.search(rf"tick_lower:\s*{-10 * spacing}i32", nfts[0])
        assert re.search(rf"tick_upper:\s*{10 * spacing}i32", nfts[0])
    pool_case["nft_record"] = nfts[0]


def _position(ctx, pool_case):
    raw = ctx.read_mapping("positions", pool_case["position_token_id"])
    assert raw, f"positions[{pool_case['position_token_id']}] is empty"
    return _position_numbers(raw)


def _submit(ctx, call, account):
    """Drive a DexCall through the fixture's ladder: transact() on the proven
    ladder; hand-built unproven transaction otherwise."""
    from .devnode_amm import UNPROVEN

    if not UNPROVEN:
        result = call.transact(account)
        ctx.devnode.advance(1)
        return result
    # Unproven: reuse the DexCall's bound call + result builder.
    from aleo_shield_swap._calls import root_outputs

    net = ctx._net()
    process = ctx.aleo.process
    auth = call._bound.authorize(account).raw
    root = ctx.state_root()
    execution = net.Execution.from_authorization_unproven(auth, root)
    cost, _ = process.execution_cost(execution)
    fee_auth = process.authorize_fee_public(
        account.private_key, cost, 0, execution.execution_id)
    fee = net.Fee.from_authorization_unproven(fee_auth, root)
    tx = net.Transaction.from_execution(execution, fee)
    tx_id = ctx.submit_and_confirm(tx, "dex call")
    decoded = [
        {"program": str(t.program_id), "function": str(t.function_name),
         "outputs": list(t.outputs())}
        for t in ctx.aleo.network.get_transaction_object(tx_id).transitions()
    ]
    outputs = root_outputs(decoded, call._bound.program_id,
                           call._bound.function_name)
    return call._build(tx_id, outputs)


def test_admin_setup_landed(ctx):
    assert ctx.read_mapping("fee_tiers", "3000u16") == "true"
    assert ctx.read_mapping("fee_tiers", "500u16") == "true"
    assert ctx.read_mapping("tick_spacings", "60u32") == "true"
    assert ctx.read_mapping("fee_to_tick_spacing", "500u16") == "10u32"
    assert ctx.read_mapping("token_allowed", ctx.token0_field) == "true"
    assert ctx.read_mapping("token_decimals", ctx.token1_field) == "6u8"
    assert ctx.read_mapping("pool_creation_is_open", "true") == "true"
    assert ctx.read_mapping("admin", "true") == str(ctx.admin.address)


def test_non_admin_creates_two_pools(ctx, dex, journey):
    for pool_case in journey.pools:
        call = dex.create_pool(
            token0_id=ctx.token0_field, token1_id=ctx.token1_field,
            fee=pool_case["fee"], initial_tick=0,
            imports=ctx.imports, account=ctx.user)
        result = _submit(ctx, call, ctx.user)
        assert result.transaction_id.startswith("at1")

        # Authoritative parity: locally derived key == the chain's.
        pool_key = dex.derive_pool_key(ctx.token0_field, ctx.token1_field,
                                       pool_case["fee"])
        assert result.position_token_id == pool_key   # create_pool's field output
        pool_case["pool_key"] = pool_key

        assert ctx.read_mapping("initialized_pools", pool_key) == "true"
        pool_struct = ctx.read_mapping("pools", pool_key)
        assert "enabled: true" in pool_struct
        assert f"fee: {pool_case['fee']}u16" in pool_struct
        slot = dex.get_slot(pool_key)
        assert slot.tick == 0
        assert slot.tick_spacing == pool_case["tick_spacing"]
        assert slot.liquidity == 0
    assert journey.pools[0]["pool_key"] != journey.pools[1]["pool_key"]


def test_mint_opens_positions(ctx, dex, journey):
    for pool_case in journey.pools:
        record0 = ctx.privatize_token(ctx.user, ctx.token0_program, 200_000_000)
        record1 = ctx.privatize_token(ctx.user, ctx.token1_program, 200_000_000)
        spacing = pool_case["tick_spacing"]
        call = dex.mint(
            pool_key=pool_case["pool_key"],
            tick_lower=-10 * spacing, tick_upper=10 * spacing,
            amount0_desired=100_000_000, amount1_desired=100_000_000,
            token0_record=record0, token1_record=record1,
            imports=ctx.imports, account=ctx.user)
        result = _submit(ctx, call, ctx.user)
        assert result.position_token_id and result.position_token_id.endswith("field")
        pool_case["position_token_id"] = result.position_token_id
        _refresh_nft(ctx, pool_case, result.transaction_id)

        # Tick-key parity: the mint initialized tick_lower; the locally
        # derived key must locate it in the ticks mapping.
        tick_key = dex.derive_tick_key(pool_case["pool_key"], -10 * spacing)
        assert ctx.read_mapping("ticks", tick_key)

        pos = _position(ctx, pool_case)
        assert pos["liquidity"] > 0
        slot = dex.get_slot(pool_case["pool_key"])
        assert slot.liquidity == pos["liquidity"]


def test_increase_liquidity_grows_position(ctx, dex, journey):
    for pool_case in journey.pools:
        before = _position(ctx, pool_case)
        record0 = ctx.privatize_token(ctx.user, ctx.token0_program, 100_000_000)
        record1 = ctx.privatize_token(ctx.user, ctx.token1_program, 100_000_000)
        call = dex.increase_liquidity(
            pool_key=pool_case["pool_key"],
            amount0_desired=50_000_000, amount1_desired=50_000_000,
            position_record=pool_case["nft_record"],
            token0_record=record0, token1_record=record1,
            # The position's ticks are already initialized, so the contract
            # skips hint validation — the MIN sentinel passes through.
            tick_lower_hint=MIN_TICK - 1, tick_upper_hint=MIN_TICK - 1,
            imports=ctx.imports, account=ctx.user)
        result = _submit(ctx, call, ctx.user)
        _refresh_nft(ctx, pool_case, result.transaction_id)
        assert _position(ctx, pool_case)["liquidity"] > before["liquidity"]


def test_decrease_liquidity_settles_owed(ctx, dex, journey):
    for pool_case in journey.pools:
        before = _position(ctx, pool_case)
        call = dex.decrease_liquidity(
            pool_key=pool_case["pool_key"],
            liquidity_to_remove=before["liquidity"] // 4,
            position_record=pool_case["nft_record"],
            imports=ctx.imports, account=ctx.user)
        result = _submit(ctx, call, ctx.user)
        _refresh_nft(ctx, pool_case, result.transaction_id)
        after = _position(ctx, pool_case)
        assert after["liquidity"] == before["liquidity"] - before["liquidity"] // 4
        assert after["tokens_owed0"] + after["tokens_owed1"] > 0


def test_swaps_both_directions_and_claims(ctx, dex, journey):
    pool_case = journey.pools[0]
    for zero_for_one in (True, False):
        token_in_program = ctx.token0_program if zero_for_one else ctx.token1_program
        token_in_id = ctx.token0_field if zero_for_one else ctx.token1_field
        slot_before = dex.get_slot(pool_case["pool_key"])
        token_record = ctx.privatize_token(ctx.user, token_in_program, 20_000_000)

        call = dex.swap(
            pool_key=pool_case["pool_key"], token_in_id=token_in_id,
            amount_in=10_000_000,
            # A 10M swap against ~150M of liquidity moves the price a lot:
            # pin the floor at zero and assert on the claimed amount instead.
            expected_out=0, slippage_bps=0,
            token_record=token_record,
            imports=ctx.imports, account=ctx.user)
        handle = _submit(ctx, call, ctx.user)
        assert handle.swap_id and handle.swap_id.endswith("field")

        # The finalize computed the outcome — read it fresh from chain.
        output = dex.get_swap_output(handle)
        assert output.amount_out > 0

        claim = _submit(ctx, dex.claim_swap_output(
            handle, imports=ctx.imports, account=ctx.user), ctx.user)
        assert claim.transaction_id.startswith("at1")
        # The SDK-reported amount must match the chain's own computation.
        assert claim.amount_out == output.amount_out

        # The private output landed as a Token record of exactly amount_out.
        records = ctx.records_of(ctx.user, claim.transaction_id)
        assert any(f"amount: {output.amount_out}u128" in r for r in records), \
            f"no claimed Token record of {output.amount_out} in {records}"

        with pytest.raises(SwapOutputNotFinalizedError):
            dex.get_swap_output(handle)          # the claim consumed the entry

        slot_after = dex.get_slot(pool_case["pool_key"])
        if zero_for_one:
            assert slot_after.sqrt_price < slot_before.sqrt_price
        else:
            assert slot_after.sqrt_price > slot_before.sqrt_price


def test_collect_pays_out_owed(ctx, dex, journey):
    for pool_case in journey.pools:
        # tokens_owed updates lazily: collect folds accrued fees into owed
        # first, then pays — two passes drain fully.
        paid_out = 0
        for pass_no in range(2):
            owed = _position(ctx, pool_case)
            if owed["tokens_owed0"] + owed["tokens_owed1"] == 0:
                assert pass_no > 0, "collect precondition: position owes nothing"
                break
            call = dex.collect(
                pool_key=pool_case["pool_key"],
                amount0_requested=owed["tokens_owed0"],
                amount1_requested=owed["tokens_owed1"],
                position_record=pool_case["nft_record"],
                imports=ctx.imports, account=ctx.user)
            result = _submit(ctx, call, ctx.user)
            for r in ctx.records_of(ctx.user, result.transaction_id):
                m = re.search(r"amount:\s*(\d+)u128", r)
                if m:
                    paid_out += int(m.group(1))
            _refresh_nft(ctx, pool_case, result.transaction_id)
        after = _position(ctx, pool_case)
        assert after["tokens_owed0"] == 0 and after["tokens_owed1"] == 0
        assert paid_out > 0


def test_burn_exits_positions(ctx, dex, journey):
    for pool_case in journey.pools:
        remaining = _position(ctx, pool_case)
        if remaining["liquidity"] > 0:
            result = _submit(ctx, dex.decrease_liquidity(
                pool_key=pool_case["pool_key"],
                liquidity_to_remove=remaining["liquidity"],
                position_record=pool_case["nft_record"],
                imports=ctx.imports, account=ctx.user), ctx.user)
            _refresh_nft(ctx, pool_case, result.transaction_id)
        owed = _position(ctx, pool_case)
        if owed["tokens_owed0"] + owed["tokens_owed1"] > 0:
            result = _submit(ctx, dex.collect(
                pool_key=pool_case["pool_key"],
                amount0_requested=owed["tokens_owed0"],
                amount1_requested=owed["tokens_owed1"],
                position_record=pool_case["nft_record"],
                imports=ctx.imports, account=ctx.user), ctx.user)
            _refresh_nft(ctx, pool_case, result.transaction_id)

        result = _submit(ctx, dex.burn(
            pool_key=pool_case["pool_key"],
            position_record=pool_case["nft_record"], account=ctx.user), ctx.user)
        assert result.transaction_id.startswith("at1")
        # The position mapping entry is removed by the burn finalize.
        assert ctx.read_mapping("positions", pool_case["position_token_id"]) is None


def test_identifier_to_field_golden():
    from .devnode_amm import identifier_to_field

    # Golden values from the TS suite — pins the LE-byte token-id encoding.
    assert identifier_to_field("test_token_a") == "30135415236709662781336675700field"
    assert identifier_to_field("test_token_b") == "30444900246531007850061456756field"
