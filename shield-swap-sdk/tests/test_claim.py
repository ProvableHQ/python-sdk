"""claim_swap_output() — input order per the deployed shield_swap.aleo:
[blinding_factor, blinded_address, swap_id, token_in, token_out,
 amount_out u128, amount_remaining u128, signer_proofs].
Wrapped output/refund tokens dispatch to the router claim variants with
trailing wrapper-proof arrays.  A zero remainder takes the no-refund
variants, which drop the amount_remaining input (and the refund proof)."""
import pytest

from aleo_shield_swap._core import default_merkle_proofs
from aleo_shield_swap._routing import ROUTER_ID
from aleo_shield_swap.client import ShieldSwap
from aleo_shield_swap.errors import SwapOutputNotFinalizedError
from aleo_shield_swap.types import ClaimResult, SwapHandle

from .conftest import POOL_TEXT, SLOT_TEXT, StubAleo

SWAP_OUTPUT_TEXT = (
    "{ recipient: 3field, caller: 4field, token_in: 1field, token_out: 2field, "
    "amount_out: 990000u128, amount_remaining: 5u128 }"
)
# Fully consumed input: nothing to refund.
SWAP_OUTPUT_NO_REFUND_TEXT = (
    "{ recipient: 3field, caller: 4field, token_in: 1field, token_out: 2field, "
    "amount_out: 990000u128, amount_remaining: 0u128 }"
)


def _handle(**over):
    base = dict(swap_id="77field", blinding_factor="11field", blinded_address="aleo1blinded",
                token_in_id="1field", token_out_id="2field", pool_key="5field",
                amount_in=10**9, transaction_id="at1req", program="shield_swap.aleo")
    base.update(over)
    return SwapHandle(**base)


def _stub(swap_outputs):
    return StubAleo(mappings={
        "pools": {"5field": POOL_TEXT},
        "slots": {"5field": SLOT_TEXT},
        "swap_outputs": swap_outputs,
    })


def test_claim_builds_exact_inputs_and_result():
    stub = _stub({"77field": SWAP_OUTPUT_TEXT})
    dex = ShieldSwap(stub)
    result = dex.claim_swap_output(_handle()).transact()
    fn, args = stub.last_call
    assert fn == "claim_swap_output"
    assert args == ["11field", "aleo1blinded", "77field",
                    "1field", "2field", "990000u128", "5u128",
                    default_merkle_proofs()]
    assert result == ClaimResult("at1stubtx", 990000, 5)


def test_claim_not_finalized_raises_before_any_call():
    stub = _stub({})
    with pytest.raises(SwapOutputNotFinalizedError):
        ShieldSwap(stub).claim_swap_output(_handle())
    assert stub.last_call is None            # no transaction was prepared


def test_claim_incomplete_handle_raises():
    stub = _stub({"77field": SWAP_OUTPUT_TEXT})
    with pytest.raises(ValueError, match="swap_id"):
        ShieldSwap(stub).claim_swap_output(_handle(swap_id=None))
    with pytest.raises(ValueError, match="blinding_factor"):
        ShieldSwap(stub).claim_swap_output(_handle(blinding_factor=None))


def _stub_wrapped(swap_outputs, wrapped):
    """Stub with the given token ids marked wrapped on chain."""
    return StubAleo(mappings={
        "pools": {"5field": POOL_TEXT},
        "slots": {"5field": SLOT_TEXT},
        "swap_outputs": swap_outputs,
        "from_wrapper_token_id": {t: "9field" for t in wrapped},
    })


def test_claim_wrapped_output_routes_through_router():
    stub = _stub_wrapped({"77field": SWAP_OUTPUT_TEXT}, wrapped={"2field"})
    ShieldSwap(stub).claim_swap_output(_handle()).transact()
    fn, args = stub.last_call
    assert stub.last_program == ROUTER_ID
    assert fn == "claim_to_wrapped_refund_arc20"
    assert len(args) == 9                       # + amm proofs + wrapper proofs
    assert args[7] == default_merkle_proofs()   # AMM signer proofs
    assert args[8] == default_merkle_proofs()   # wrapper proofs


def test_claim_wrapped_refund_routes_through_router():
    stub = _stub_wrapped({"77field": SWAP_OUTPUT_TEXT}, wrapped={"1field"})
    ShieldSwap(stub).claim_swap_output(_handle()).transact()
    fn, args = stub.last_call
    assert (stub.last_program, fn) == (ROUTER_ID, "claim_to_arc20_refund_wrapped")
    assert len(args) == 9


def test_claim_both_wrapped_routes_with_two_proof_arrays():
    stub = _stub_wrapped({"77field": SWAP_OUTPUT_TEXT}, wrapped={"1field", "2field"})
    ShieldSwap(stub).claim_swap_output(_handle()).transact()
    fn, args = stub.last_call
    assert (stub.last_program, fn) == (ROUTER_ID, "claim_to_wrapped_refund_wrapped")
    assert len(args) == 10
    assert args[8] == args[9] == default_merkle_proofs()


def test_claim_plain_stays_on_core():
    stub = _stub_wrapped({"77field": SWAP_OUTPUT_TEXT}, wrapped=set())
    ShieldSwap(stub).claim_swap_output(_handle()).transact()
    fn, _ = stub.last_call
    assert (stub.last_program, fn) == ("shield_swap.aleo", "claim_swap_output")


# ── No-refund variants (amount_remaining == 0) ───────────────────────────────

def test_claim_no_refund_plain_drops_remaining_input():
    stub = _stub_wrapped({"77field": SWAP_OUTPUT_NO_REFUND_TEXT}, wrapped=set())
    result = ShieldSwap(stub).claim_swap_output(_handle()).transact()
    fn, args = stub.last_call
    assert (stub.last_program, fn) == ("shield_swap.aleo", "claim_swap_output_no_refund")
    assert args == ["11field", "aleo1blinded", "77field",
                    "1field", "2field", "990000u128",
                    default_merkle_proofs()]           # no amount_remaining
    assert result == ClaimResult("at1stubtx", 990000, 0)


def test_claim_no_refund_wrapped_output_takes_one_receiver_proof():
    stub = _stub_wrapped({"77field": SWAP_OUTPUT_NO_REFUND_TEXT}, wrapped={"2field"})
    ShieldSwap(stub).claim_swap_output(_handle()).transact()
    fn, args = stub.last_call
    assert (stub.last_program, fn) == (ROUTER_ID, "claim_to_wrapped_no_refund")
    assert len(args) == 8
    assert args[6] == default_merkle_proofs()          # AMM signer proofs
    assert args[7] == default_merkle_proofs()          # output receiver proof


def test_claim_no_refund_wrapped_input_only_needs_no_receiver_proof():
    # The refund leg is wrapped but empty: the router variant still exists
    # (it must not pay through the core), yet takes no receiver proof.
    stub = _stub_wrapped({"77field": SWAP_OUTPUT_NO_REFUND_TEXT}, wrapped={"1field"})
    ShieldSwap(stub).claim_swap_output(_handle()).transact()
    fn, args = stub.last_call
    assert (stub.last_program, fn) == (ROUTER_ID, "claim_to_arc20_no_refund")
    assert len(args) == 7


def test_claim_no_refund_both_wrapped_proves_only_the_output():
    stub = _stub_wrapped({"77field": SWAP_OUTPUT_NO_REFUND_TEXT},
                         wrapped={"1field", "2field"})
    ShieldSwap(stub).claim_swap_output(_handle()).transact()
    fn, args = stub.last_call
    assert (stub.last_program, fn) == (ROUTER_ID, "claim_to_wrapped_no_refund")
    assert len(args) == 8
