"""End-to-end tests for the execute/prove/verify/fee/cost/transaction paths.

Fast tests (unmarked) are fully offline: `Process.execute(auth)` synthesizes the
circuit locally without a proof, and fee authorization / serial numbers are pure
local computation. (First-ever run downloads a small ~3 MB SRS chunk to ~/.aleo.)

Slow tests (`@pytest.mark.slow`) require:
  * network access to the mainnet REST endpoint (for `trace.prepare`, which
    fetches the latest global state root), and
  * SNARK proving parameter downloads on first use (hundreds of MB, cached in
    ~/.aleo/resources) — the first run can take several minutes.
Run them locally with: python -m pytest python/tests -v -m slow
They are excluded from CI via -m "not slow".

Endpoint note (verified empirically): Query.rest() wants the BASE url
`https://api.explorer.provable.com/v1` — snarkvm's REST query appends the
network path (`/mainnet/stateRoot/latest`) itself; passing `.../v1/mainnet`
would double the network segment.
"""

import json

import pytest

from aleo.mainnet import (
    Authorization,
    Execution,
    Field,
    Identifier,
    Locator,
    PrivateKey,
    Process,
    ProgramID,
    Query,
    RecordCiphertext,
    Transaction,
    Value,
    ViewKey,
)
from conftest import load_vectors

ENDPOINT = "https://api.explorer.provable.com/v1"

CREDITS = ProgramID.from_string("credits.aleo")
TRANSFER_PUBLIC = Identifier.from_string("transfer_public")


def _authorize_transfer_public(process: Process, private_key: PrivateKey) -> Authorization:
    recipient = str(private_key.address)
    return process.authorize(
        private_key,
        CREDITS,
        TRANSFER_PUBLIC,
        [Value.parse(recipient), Value.parse("10u64")],
    )


# ---------------------------------------------------------------------------
# Fast / offline tests
# ---------------------------------------------------------------------------


def test_execute_transfer_public():
    process = Process.load()
    pk = PrivateKey.random()
    auth = _authorize_transfer_public(process, pk)

    response, trace = process.execute(auth)

    outputs = response.outputs
    assert isinstance(outputs, list)
    assert len(outputs) == 1  # transfer_public returns a single future
    assert "transfer_public" in str(outputs[0])

    transitions = trace.transitions()
    assert len(transitions) == 1
    assert str(transitions[0].program_id) == "credits.aleo"
    assert str(transitions[0].function_name) == "transfer_public"
    assert trace.is_fee() is False


def test_authorize_fee_public():
    process = Process.load()
    pk = PrivateKey.random()

    fee_auth = process.authorize_fee_public(pk, 1000, 0, Field.zero())
    assert fee_auth is not None

    # JSON round-trip
    round_tripped = Authorization.from_json(fee_auth.to_json())
    assert str(round_tripped) == str(fee_auth)


def test_serial_number_deterministic():
    v = load_vectors("records.json")["decrypt_kat"]

    plaintext = RecordCiphertext.from_string(v["ciphertext"]).decrypt(
        ViewKey.from_string(v["view_key"])
    )
    owner_key = PrivateKey.from_string(v["private_key"])
    # Sanity: the vendored private key really owns the KAT record.
    assert str(owner_key.address) == v["owner"]

    record_view_key = Field.from_string(v["record_view_key"])
    args = (owner_key, CREDITS, Identifier.from_string("credits"), record_view_key)

    sn1 = plaintext.serial_number(*args)
    sn2 = plaintext.serial_number(*args)

    assert str(sn1) == str(sn2)  # deterministic
    assert str(sn1).endswith("field")


# ---------------------------------------------------------------------------
# Slow / network tests (proving)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def process():
    return Process.load()


@pytest.fixture(scope="module")
def signer():
    return PrivateKey.random()


@pytest.fixture(scope="module")
def proven_execution(process, signer):
    """Authorize + execute + prove a credits.aleo/transfer_public execution.

    Module-scoped so the expensive proof (and first-run parameter downloads,
    which can take minutes) happens exactly once across the slow tests.
    """
    auth = _authorize_transfer_public(process, signer)
    _, trace = process.execute(auth)
    trace.prepare(Query.rest(ENDPOINT))
    # NOTE: despite the .pyi stub saying `locator: str`, the native binding
    # requires a Locator object; a plain str raises TypeError.
    return trace.prove_execution(Locator.from_string("credits.aleo/transfer_public"))


@pytest.mark.slow
def test_prove_and_verify_execution(process, proven_execution):
    execution = proven_execution

    # verify_execution raises on failure; returning None means success.
    assert process.verify_execution(execution) is None

    # Cost: (total, (storage, finalize)) in microcredits.
    total, (storage, finalize) = process.execution_cost(execution)
    assert total > 0
    assert storage > 0
    assert finalize >= 0
    assert total == storage + finalize

    # Negative: tamper with the signer commitment (scm). Determined
    # empirically: tampering tcm/inputs/outputs is rejected already at
    # Execution.from_json ("Transition ID mismatch"), because the transition
    # id is recomputed on deserialization — but scm is not part of the
    # transition id, so the tampered execution parses fine and must then fail
    # proof verification.
    tampered = json.loads(execution.to_json())
    assert tampered["transitions"][0]["scm"] != "0field"
    tampered["transitions"][0]["scm"] = "0field"
    tampered_execution = Execution.from_json(json.dumps(tampered))
    with pytest.raises(Exception):
        process.verify_execution(tampered_execution)


@pytest.mark.slow
def test_prove_fee_and_transaction(process, signer, proven_execution):
    execution = proven_execution
    execution_id = execution.execution_id

    # Authorize and prove a public fee bound to this execution id.
    fee_auth = process.authorize_fee_public(signer, 1_000_000, 0, execution_id)
    _, fee_trace = process.execute(fee_auth)
    assert fee_trace.is_fee() is True
    assert fee_trace.is_fee_public() is True

    fee_trace.prepare(Query.rest(ENDPOINT))
    fee = fee_trace.prove_fee()
    assert fee.is_fee_public() is True
    assert str(fee.payer) == str(signer.address)

    # verify_fee raises on failure; it must be bound to the same execution id.
    assert process.verify_fee(fee, execution_id) is None
    with pytest.raises(Exception):
        process.verify_fee(fee, Field.zero())

    # Assemble a transaction and JSON round-trip it.
    tx = Transaction.from_execution(execution, fee)
    tx_json = tx.to_json()
    parsed = json.loads(tx_json)
    assert parsed["type"] == "execute"
    round_tripped = Transaction.from_json(tx_json)
    assert round_tripped.to_json() == tx_json
