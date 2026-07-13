"""claim_swap_output() — input order per TS claimSwapOutput.ts:
[blinding_factor, blinded_address, swap_id, token_in, token_out,
 amount_out u128, amount_remaining u128]"""
import pytest

from aleo_shield_swap.client import ShieldSwap
from aleo_shield_swap.errors import SwapOutputNotFinalizedError
from aleo_shield_swap.types import ClaimResult, SwapHandle

from .conftest import POOL_TEXT, SLOT_TEXT, StubAleo

SWAP_OUTPUT_TEXT = (
    "{ recipient: 3field, caller: 4field, token_in: 1field, token_out: 2field, "
    "amount_out: 990000u128, amount_remaining: 0u128, token_in_1: 1field, "
    "amount_remaining_1: 0u128, token_in_2: 1field, amount_remaining_2: 0u128 }"
)


def _handle(**over):
    base = dict(swap_id="77field", blinding_factor="11field", blinded_address="aleo1blinded",
                token_in_id="1field", token_out_id="2field", pool_key="5field",
                amount_in=10**9, transaction_id="at1req", program="shield_swap_v3.aleo")
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
                    "1field", "2field", "990000u128", "0u128"]
    assert result == ClaimResult("at1stubtx", 990000, 0)


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
