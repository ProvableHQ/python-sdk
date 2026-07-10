"""Tests for the ProvingRequest and ExecutionRequest types (delegated proving).

Ported from the ProvableHQ ts-sdk wasm crate
(wasm/src/synthesizer/proving_request.rs). These are mainnet fixtures; the SDK
under test is built with the mainnet feature (network ID 0).

The `authorization()` accessor is compared to the bare AUTHORIZATION fixture by
JSON string, mirroring the wasm `.equals()` check (the Python `Authorization`
class does not expose `__eq__`).
"""

from aleo.mainnet import (
    Authorization,
    ExecutionRequest,
    PrivateKey,
    ProvingRequest,
)

from conftest import load_vectors

VECTORS = load_vectors("proving_request.json")
PROVING_REQUEST = VECTORS["PUZZLE_SPINNER_V002_PROVING_REQUEST"]
AUTHORIZATION = VECTORS["PUZZLE_SPINNER_V002_AUTHORIZATION"]

# Beacon private key -- matches the wasm-side fixtures.
BEACON_PRIVATE_KEY = "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH"

# transfer_public-style Request-variant inputs (from the wasm test).
TRANSFER_ADDRESS = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
TRANSFER_INPUTS = [TRANSFER_ADDRESS, "100u64"]
TRANSFER_INPUT_TYPES = ["address.public", "u64.public"]


def _sample_execution_request() -> ExecutionRequest:
    private_key = PrivateKey.from_string(BEACON_PRIVATE_KEY)
    return ExecutionRequest.sign(
        private_key,
        "credits.aleo",
        "transfer_public",
        TRANSFER_INPUTS,
        TRANSFER_INPUT_TYPES,
        None,  # root_tvk
        None,  # program_checksum
        True,  # is_root
        False,  # is_dynamic
    )


# ---- Authorization variant (legacy) -------------------------------------


def test_proving_request_serialization_roundtrip():
    proving_request = ProvingRequest.from_string(PROVING_REQUEST)

    byte_roundtrip = ProvingRequest.from_bytes(bytes(proving_request.bytes()))
    string_roundtrip = ProvingRequest.from_string(str(proving_request))

    assert proving_request == byte_roundtrip
    assert proving_request == string_roundtrip


def test_proving_request_accessor_methods_give_correct_authorizations():
    proving_request = ProvingRequest.from_string(PROVING_REQUEST)

    authorization = Authorization.from_json(AUTHORIZATION)
    # Mirror the wasm `.equals()` check via JSON (Authorization has no __eq__).
    assert proving_request.authorization().to_json() == authorization.to_json()
    assert proving_request.fee_authorization().is_fee_public()
    assert proving_request.broadcast is False
    assert proving_request.kind() == "authorization"


def test_fee_authorization_binds_to_execution_id():
    """The fee authorization's execution-id input must equal the main
    authorization's execution ID. This pins our to_execution_id (W4c) against
    the reference fixture."""
    import json

    proving_request = ProvingRequest.from_string(PROVING_REQUEST)
    fee = json.loads(proving_request.fee_authorization().to_json())
    # credits.aleo/fee_public inputs: [base_fee, priority_fee, execution_id].
    execution_id_input = fee["requests"][0]["inputs"][2]
    expected = str(proving_request.authorization().to_execution_id())
    assert execution_id_input == expected


# ---- Request variant ----------------------------------------------------


def test_request_variant_construction_and_kind():
    request = _sample_execution_request()
    proving_request = ProvingRequest.from_request(request, None, False)

    assert proving_request.kind() == "request"
    assert proving_request.is_request()
    assert not proving_request.is_authorization()
    assert proving_request.request() is not None
    assert proving_request.fee_request() is None
    assert proving_request.broadcast is False


def test_request_variant_byte_roundtrip():
    request = _sample_execution_request()
    proving_request = ProvingRequest.from_request(request, None, True)

    # Roundtrip via the explicit Request reader; the byte layout carries no
    # discriminator so the Authorization reader would not work here.
    roundtripped = ProvingRequest.from_bytes_request(bytes(proving_request.bytes()))

    assert proving_request == roundtripped
    assert roundtripped.kind() == "request"
    assert roundtripped.broadcast is True


def test_request_variant_string_roundtrip_autodetects():
    request = _sample_execution_request()
    proving_request = ProvingRequest.from_request(request, None, False)

    # JSON shape carries the variant -- from_string picks the right one via
    # untagged serde dispatch on disjoint field names.
    roundtripped = ProvingRequest.from_string(str(proving_request))

    assert roundtripped.kind() == "request"
    assert proving_request == roundtripped


def test_request_variant_accessors_raise_on_authorization_methods():
    request = _sample_execution_request()
    proving_request = ProvingRequest.from_request(request, None, False)

    try:
        proving_request.authorization()
        raise AssertionError("expected authorization() to raise")
    except Exception as err:
        assert "Request variant" in str(err)

    # fee_authorization() returns None rather than raising.
    assert proving_request.fee_authorization() is None


def test_authorization_variant_raises_on_request_accessor():
    proving_request = ProvingRequest.from_string(PROVING_REQUEST)

    try:
        proving_request.request()
        raise AssertionError("expected request() to raise")
    except Exception as err:
        assert "Authorization variant" in str(err)

    assert proving_request.fee_request() is None
