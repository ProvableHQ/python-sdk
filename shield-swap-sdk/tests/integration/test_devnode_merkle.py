"""Freeze-list Merkle exclusion proofs against the real deployed AMM.

Deploys the network's ``shield_swap.aleo``, ``shield_swap_freezelist.aleo``,
and two plain ARC-20s onto a devnode, populates the freeze list, and mints a
position — the transition that requires three separate non-inclusion proofs
(signer, recipient, withdrawal).

The unit tests in ``sdk/python/tests/test_merkle.py`` check the port against a
Python transcription of the verifier.  This checks it against the verifier
itself, compiled into the deployed bytecode, which is the only thing that
settles whether the proofs are actually right.

``test_placeholder_proof_is_rejected`` is the control: it shows the all-zero
literal the SDK ships today stops working the moment the list is non-empty, so
the passing mint is evidence about the proofs rather than about a list that
happens to accept anything.

Run with::

    pytest tests/integration/test_devnode_merkle.py -m devnode
"""
from __future__ import annotations

import pytest

from aleo import MerkleExclusionProof
from aleo_shield_swap import ShieldSwap
from aleo_shield_swap import _generated as g
from aleo_shield_swap._core import default_merkle_proofs, generate_field_nonce

from .devnode_merkle_stack import (
    AMM_PROGRAM,
    FREEZELIST_PROGRAM,
    MerkleDevnode,
    setup_merkle_devnode,
)

pytestmark = pytest.mark.devnode

FEE = 3000
INITIAL_TICK = 0
MINT_AMOUNT = 100_000_000
FUND_AMOUNT = 200_000_000

#: ``Poseidon4([1field, 0field, 0field])`` — the root while the list is empty.
EMPTY_TREE_ROOT = (
    "3642222252059314292809609689035560016959342421640560347114299934615987159853field"
)


@pytest.fixture(scope="module")
def ctx() -> MerkleDevnode:
    try:
        context = setup_merkle_devnode(freeze_count=5)
    except Exception as exc:
        if "aleo-devnode not found" in str(exc):
            pytest.skip(f"aleo-devnode binary not available: {exc}")
        raise
    yield context
    context.stop()


@pytest.fixture(scope="module")
def dex(ctx: MerkleDevnode) -> ShieldSwap:
    client = ShieldSwap(ctx.aleo, program=AMM_PROGRAM)
    ctx.aleo.default_account = ctx.user
    return client


@pytest.fixture(scope="module")
def pool_key(ctx: MerkleDevnode, dex: ShieldSwap) -> str:
    """A live pool over the two deployed ARC-20s."""
    call = dex.create_pool(
        token0_id=ctx.token0_id, token1_id=ctx.token1_id,
        fee=FEE, initial_tick=INITIAL_TICK, account=ctx.user)
    ctx.submit_call(call, ctx.user, "create_pool")
    key = dex.derive_pool_key(ctx.token0_id, ctx.token1_id, FEE)
    assert ctx.read_mapping(AMM_PROGRAM, "pools", key), "pool was not created"
    return key


def build_mint_inputs(ctx: MerkleDevnode, dex: ShieldSwap, pool_key: str, *,
                      signer_proofs: str, recipient_proofs: str,
                      withdrawal_proofs: str) -> list[str]:
    """``mint`` inputs in deployed order, with the proofs left to the caller.

    Mirrors ``ShieldSwap.mint``'s assembly.  It is rebuilt here rather than
    reused because the shipped method hardcodes the placeholder proof and takes
    no parameter to override it.
    """
    slot = dex.get_slot(pool_key)
    spacing = slot.tick_spacing
    lower, upper = -10 * spacing, 10 * spacing

    lower_hint = dex.find_tick_predecessor(pool_key, lower)
    upper_pred = dex.find_tick_predecessor(pool_key, upper)
    # The finalize inserts the lower tick before validating the upper hint.
    upper_hint = lower if lower > upper_pred else upper_pred

    request = g.MintPositionRequest(
        pool=pool_key, tick_lower=lower, tick_upper=upper,
        amount0_desired=MINT_AMOUNT, amount1_desired=MINT_AMOUNT,
        amount0_min=0, amount1_min=0,
        tick_lower_hint=lower_hint, tick_upper_hint=upper_hint,
    ).to_plaintext()

    record0 = ctx.privatize_token(ctx.user, ctx.token0_program, FUND_AMOUNT)
    record1 = ctx.privatize_token(ctx.user, ctx.token1_program, FUND_AMOUNT)

    return [
        generate_field_nonce(), record0, record1,
        str(ctx.user.address), str(ctx.user.address), request,
        ctx.token0_id, ctx.token1_id,
        signer_proofs, recipient_proofs, withdrawal_proofs,
    ]


def test_freeze_list_root_matches_the_locally_built_tree(ctx: MerkleDevnode):
    """The root the contract stores is the one the port computes.

    Nothing on chain recomputes the root — the manager supplies it — so this is
    the step that ties the local tree to the value every proof must reproduce.
    """
    assert len(ctx.frozen) == 5
    assert ctx.on_chain_root() == ctx.merkle.root(ctx.local_tree())


def test_freeze_list_is_no_longer_empty(ctx: MerkleDevnode):
    """Without this, a passing mint would prove nothing about the proofs."""
    assert ctx.on_chain_root() != EMPTY_TREE_ROOT


def test_frozen_addresses_cannot_be_proven_absent(ctx: MerkleDevnode):
    """A member is rejected client-side rather than on-chain."""
    with pytest.raises(ValueError, match="on the list"):
        ctx.exclusion_proof(ctx.frozen[0])


def test_placeholder_proof_is_rejected(ctx: MerkleDevnode, dex: ShieldSwap,
                                       pool_key: str):
    """The all-zero literal fails against a populated list.

    This is the control for :func:`test_mint_with_exclusion_proofs_succeeds`:
    without it, a mint could pass because the list accepts anything.

    The placeholder clears the *circuit* — two copies of a depth-1 all-zero
    path reconstruct the empty-tree root, and every real address sorts above
    ``0field``, so it is a genuine non-inclusion proof for an empty tree.  What
    stops it is the finalize's ``assert_valid_freeze_list_root``, which no
    longer matches once the list has entries.  The transaction is therefore
    *rejected* on chain rather than failing to authorize.
    """
    placeholder = default_merkle_proofs()
    inputs = build_mint_inputs(
        ctx, dex, pool_key, signer_proofs=placeholder,
        recipient_proofs=placeholder, withdrawal_proofs=placeholder)

    with pytest.raises(RuntimeError, match="was not accepted") as excinfo:
        ctx.execute(ctx.user, AMM_PROGRAM, "mint", inputs,
                    "mint with placeholder proofs")

    assert "'status': 'rejected'" in str(excinfo.value), (
        "expected the finalize's root check to reject the placeholder")


def test_mint_with_exclusion_proofs_succeeds(ctx: MerkleDevnode, dex: ShieldSwap,
                                             pool_key: str):
    """A real proof carries a mint through the deployed verifier.

    The signer, recipient, and withdrawal address are all the user here, so the
    same proof satisfies all three checks — the contract verifies each
    independently and requires all three roots to agree.
    """
    proof = ctx.exclusion_proof(str(ctx.user.address))
    inputs = build_mint_inputs(
        ctx, dex, pool_key, signer_proofs=proof,
        recipient_proofs=proof, withdrawal_proofs=proof)

    tx_id = ctx.execute(ctx.user, AMM_PROGRAM, "mint", inputs,
                        "mint with exclusion proofs")

    assert tx_id
    nfts = [r for r in ctx.records_of(ctx.user, tx_id) if "token_id" in r]
    assert nfts, "mint produced no PositionNFT"


def test_proof_still_verifies_after_the_list_grows(ctx: MerkleDevnode,
                                                   dex: ShieldSwap,
                                                   pool_key: str):
    """Freezing another address rotates the root; a fresh proof tracks it."""
    before = ctx.on_chain_root()
    ctx.freeze(str(ctx.aleo.account.create().address))
    after = ctx.on_chain_root()
    assert after != before, "freezing did not rotate the root"

    proof = ctx.exclusion_proof(str(ctx.user.address))
    inputs = build_mint_inputs(
        ctx, dex, pool_key, signer_proofs=proof,
        recipient_proofs=proof, withdrawal_proofs=proof)

    assert ctx.execute(ctx.user, AMM_PROGRAM, "mint", inputs,
                       "mint after root rotation")


def test_exclusion_proof_from_the_served_tree(ctx: MerkleDevnode):
    """A tree read back as serialized nodes proves the same as a built one.

    This is the shape ``AleoNetworkClient.get_freeze_list`` returns, so it
    covers the path a caller takes when the list comes from a service rather
    than from a local rebuild.
    """
    built = ctx.local_tree()

    served = MerkleExclusionProof().tree_from_nodes([str(n) for n in built])

    assert served == built
    assert (MerkleExclusionProof().exclusion_proof(served, str(ctx.user.address))
            == ctx.exclusion_proof(str(ctx.user.address)))
