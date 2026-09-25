import json

import pytest

from aleo_bridge._calls import AleoCall, extract_tx_id, is_duplicate_submission, payload_transitions, root_outputs
from aleo_bridge.errors import ConfigurationError
from aleo_bridge.types import PreparedTx

PROGRAM, FN = "shielded_usdcx_wrapper.aleo", "private_burn"


def _call(fake_aleo, imports=None) -> AleoCall:
    bound = fake_aleo.programs.get(PROGRAM).functions[FN]("a", "2500000u128")
    return AleoCall(fake_aleo, bound, lambda tx_id, outs: (tx_id, outs), imports=imports)


def test_helpers():
    assert extract_tx_id("at1abc") == "at1abc"
    assert extract_tx_id({"transaction": {"id": "at1x"}}) == "at1x"
    assert extract_tx_id({"transaction_id": "at1y"}) == "at1y"
    with pytest.raises(ValueError):
        extract_tx_id({"nope": 1})
    decoded = [{"program": "tok.aleo", "function": "transfer", "outputs": [{"value": "999field"}]},
               {"program": "p.aleo", "function": "f", "outputs": [{"value": "77field"}]},
               {"program": "p.aleo", "function": "f", "outputs": ["78field"]}]
    assert root_outputs(decoded, "p.aleo", "f") == ["78field"]   # LAST matching transition is the root
    assert root_outputs(decoded, "p.aleo", "g") == []
    assert payload_transitions({"transaction_id": "x"}) is None
    assert payload_transitions({"transaction": {"execution": {"transitions": [{"program": "p", "function": "f", "outputs": []}]}}}) == \
        [{"program": "p", "function": "f", "outputs": []}]
    assert is_duplicate_submission(RuntimeError("Transaction 'at1x' already exists in the ledger"))
    assert not is_duplicate_submission(RuntimeError("duplicate transaction"))   # real double-spend, must propagate
    assert not is_duplicate_submission(RuntimeError("duplicate serial number"))
    assert not is_duplicate_submission(RuntimeError("insufficient fee"))


def test_attributes_and_simulate(fake_aleo):
    call = _call(fake_aleo)
    assert (call.program_id, call.function_name, call.inputs) == (PROGRAM, FN, ["a", "2500000u128"])
    assert call.simulate() == "simulated" and fake_aleo.simulated == [(PROGRAM, FN)]
    assert fake_aleo.submitted == [] and fake_aleo.delegated == []


def test_repr_never_leaks_input_literals(fake_aleo):
    # private_burn input 0 is a USDCx record plaintext; a secret nonce (private_mint input 3) is
    # just as sensitive — repr() must never print .inputs, only a count.
    record = "{ owner: aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px.private, amount: 5000000u128.private, _nonce: 7group.public }"
    secret_nonce = "7scalar"
    bound = fake_aleo.programs.get(PROGRAM).functions[FN](record, "2500000u128", "0u32", "[0field]", secret_nonce)
    call = AleoCall(fake_aleo, bound, lambda tx_id, outs: (tx_id, outs))
    text = repr(call)
    assert text == f"AleoCall({PROGRAM}/{FN}, inputs=5 literals)"
    assert record not in text and secret_nonce not in text
    assert call.inputs == [record, "2500000u128", "0u32", "[0field]", secret_nonce]   # accessor still exposes them


def test_prove_returns_prepared_tx_without_broadcast(fake_aleo):
    prepared = _call(fake_aleo).prove(priority_fee=5)
    assert isinstance(prepared, PreparedTx) and prepared.transaction_id == "at1built"
    assert json.loads(prepared.serialized)["id"] == "at1built"
    assert fake_aleo.fee_kwargs == [{"priority_fee": 5}] and fake_aleo.submitted == []


def test_transact_harvests_root_outputs_then_broadcasts(fake_aleo):
    tx_id, outputs = _call(fake_aleo).transact()
    assert tx_id == "at1built" and outputs == ["77field"]
    assert len(fake_aleo.submitted) == 1 and json.loads(fake_aleo.submitted[0])["id"] == "at1built"


def test_delegate_broadcast_uses_payload_transitions(fake_aleo):
    tx_id, outputs = _call(fake_aleo).delegate(wait=False)
    assert (tx_id, outputs) == ("at1delegated", ["77field"])
    assert fake_aleo.delegated[0]["broadcast"] is True and fake_aleo.waited == []
    _call(fake_aleo).delegate(wait_timeout=7.0)
    assert fake_aleo.waited == [("at1delegated", 7.0)]


def test_delegate_falls_back_to_fetching_when_payload_is_id_only(fake_aleo):
    fake_aleo.delegate_returns_id_only = True
    bound = fake_aleo.programs.get("hyp_warp_token_wbtc_v2.aleo").functions["transfer_remote"]("x")
    tx_id, outputs = AleoCall(fake_aleo, bound, lambda t, o: (t, o)).delegate(wait=False)
    assert tx_id == "at1delegated" and outputs == ["99field"]      # from get_transaction_object
    assert fake_aleo.waited == [("at1delegated", 180.0)]           # must wait before fetching


def test_delegate_prepared_and_submit_prepared(fake_aleo):
    call = _call(fake_aleo)
    prepared = call.delegate_prepared()
    assert fake_aleo.delegated[-1]["broadcast"] is False and fake_aleo.submitted == []
    assert prepared.transaction_id == "at1delegated" and json.loads(prepared.serialized)["execution"]["transitions"]
    tx_id, outputs = call.submit_prepared(prepared, wait=False)
    assert (tx_id, outputs) == ("at1delegated", ["77field"]) and fake_aleo.submitted == [prepared.serialized]


def test_delegate_without_broadcast_is_prepare_then_submit(fake_aleo):
    tx_id, outputs = _call(fake_aleo).delegate(broadcast=False, wait_timeout=9.0)
    assert tx_id == "at1delegated" and outputs == ["77field"]
    assert fake_aleo.delegated[-1]["broadcast"] is False and len(fake_aleo.submitted) == 1
    assert fake_aleo.waited == [("at1delegated", 9.0)]


def test_submit_prepared_treats_duplicate_as_success(fake_aleo):
    call = _call(fake_aleo)
    prepared = call.delegate_prepared()
    fake_aleo.duplicate_on_submit = True
    tx_id, outputs = call.submit_prepared(prepared)
    assert tx_id == "at1delegated" and outputs == ["77field"]


def test_submit_prepared_confirmation_timeout_propagates_after_broadcast(fake_aleo):
    from aleo.facade.errors import TransactionConfirmationTimeout

    call = _call(fake_aleo)
    prepared = call.delegate_prepared()
    fake_aleo.wait_raises = True
    with pytest.raises(TransactionConfirmationTimeout):
        call.submit_prepared(prepared)
    assert fake_aleo.submitted == [prepared.serialized]     # broadcast already happened


def test_delegate_prepared_requires_transaction_payload(fake_aleo):
    fake_aleo.delegate_returns_id_only = True
    with pytest.raises(ConfigurationError, match="did not return the transaction"):
        _call(fake_aleo).delegate_prepared()


def test_imports_are_registered_once_before_first_verb(fake_aleo):
    call = _call(fake_aleo, imports={"token_registry.aleo": "program token_registry.aleo;", PROGRAM: "program shielded_usdcx_wrapper.aleo;"})
    assert fake_aleo.registered == []
    call.simulate()
    assert fake_aleo.registered == ["token_registry.aleo", PROGRAM]
    call.simulate()
    assert fake_aleo.registered == ["token_registry.aleo", PROGRAM]        # idempotent
