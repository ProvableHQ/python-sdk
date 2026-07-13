"""Tests for F1 facade: Aleo client, HTTPProvider, helpers, typed exceptions.

All tests are mocked — no live network required.
"""
from __future__ import annotations

import pytest
import responses as resp_lib

from aleo import (
    Aleo,
    HTTPProvider,
    AleoError,
    TransactionNotFound,
    ProgramNotFound,
    ExecutionError,
    TransactionConfirmationTimeout,
    AleoNetworkError,
    AleoProvingError,
)

BASE = "https://api.provable.com/v2"
MAINNET_HOST = f"{BASE}/mainnet"
TESTNET_HOST = f"{BASE}/testnet"

# A real mainnet address for address-validation tests
REAL_ADDRESS = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"


# ---------------------------------------------------------------------------
# HTTPProvider construction
# ---------------------------------------------------------------------------


def test_provider_defaults() -> None:
    p = HTTPProvider(BASE)
    assert p.url == BASE
    assert p.network == "mainnet"
    assert p.api_key is None
    assert p.prover_uri is None


def test_provider_custom_network() -> None:
    p = HTTPProvider(BASE, network="testnet")
    assert p.network == "testnet"


def test_provider_invalid_network() -> None:
    with pytest.raises(ValueError, match="Invalid network"):
        HTTPProvider(BASE, network="devnet")


def test_provider_repr() -> None:
    p = HTTPProvider(BASE)
    r = repr(p)
    assert "HTTPProvider" in r
    assert "mainnet" in r


def test_provider_builds_client_with_correct_base_and_network() -> None:
    """Provider builds an AleoNetworkClient with the right host and network."""
    p = HTTPProvider(BASE, network="mainnet")
    client = p._build_client()
    # The underlying client sets _host = f"{base}/{network}"
    assert client._network == "mainnet"
    assert client._host == f"{BASE}/mainnet"


def test_provider_builds_testnet_client() -> None:
    p = HTTPProvider(BASE, network="testnet")
    client = p._build_client()
    assert client._network == "testnet"
    assert client._host == f"{BASE}/testnet"


def test_provider_passes_api_key_to_client() -> None:
    p = HTTPProvider(BASE, api_key="my-key")
    client = p._build_client()
    assert client.api_key == "my-key"


def test_provider_passes_prover_uri_to_client() -> None:
    p = HTTPProvider(BASE, prover_uri="https://prover.example.com")
    client = p._build_client()
    # AleoNetworkClient appends /{network} to prover_uri
    assert client._prover_uri == "https://prover.example.com/mainnet"


def test_provider_passes_custom_headers_to_client() -> None:
    p = HTTPProvider(BASE, headers={"X-Custom": "value"})
    client = p._build_client()
    assert client.headers.get("X-Custom") == "value"


# ---------------------------------------------------------------------------
# Aleo client construction
# ---------------------------------------------------------------------------


def test_aleo_requires_provider() -> None:
    with pytest.raises(TypeError, match="HTTPProvider"):
        Aleo("not-a-provider")  # type: ignore[arg-type]


def test_aleo_nested_provider_class() -> None:
    """Aleo.HTTPProvider is the same class as the top-level HTTPProvider."""
    assert Aleo.HTTPProvider is HTTPProvider


def test_aleo_repr() -> None:
    a = Aleo(HTTPProvider(BASE))
    assert "Aleo" in repr(a)
    assert "mainnet" in repr(a)


def test_aleo_escape_hatches_accessible() -> None:
    p = HTTPProvider(BASE)
    a = Aleo(p)
    # provider escape hatch
    assert a.provider is p
    # network_client escape hatch
    from aleo.network_client import AleoNetworkClient
    assert isinstance(a.network_client, AleoNetworkClient)


# ---------------------------------------------------------------------------
# Lazy process loading
# ---------------------------------------------------------------------------


def test_process_not_loaded_on_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Process.load() must NOT be called just from constructing Aleo(provider)."""
    loaded: list[bool] = []

    # Patch mainnet.Process.load to record calls
    import aleo.mainnet as mn  # type: ignore[attr-defined]
    original_load = mn.Process.load  # type: ignore[attr-defined]

    def spy_load() -> object:  # type: ignore[return-value]
        loaded.append(True)
        return original_load()

    monkeypatch.setattr(mn.Process, "load", spy_load)

    _ = Aleo(HTTPProvider(BASE))
    assert loaded == [], "Process.load() was called during construction"


def test_process_loaded_on_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accessing .process triggers Process.load() exactly once."""
    import aleo.mainnet as mn  # type: ignore[attr-defined]
    original_load = mn.Process.load  # type: ignore[attr-defined]
    loaded: list[bool] = []

    def spy_load() -> object:  # type: ignore[return-value]
        loaded.append(True)
        return original_load()

    monkeypatch.setattr(mn.Process, "load", spy_load)

    a = Aleo(HTTPProvider(BASE))
    # Access process twice — load should only happen once (cached)
    proc1 = a.process
    proc2 = a.process
    assert proc1 is proc2
    assert len(loaded) == 1, f"Expected 1 load, got {len(loaded)}"


# ---------------------------------------------------------------------------
# Network identity
# ---------------------------------------------------------------------------


def test_network_id_mainnet() -> None:
    a = Aleo(HTTPProvider(BASE, network="mainnet"))
    assert a.network_id == 0


def test_network_name_mainnet() -> None:
    a = Aleo(HTTPProvider(BASE, network="mainnet"))
    assert a.network_name == "mainnet"


def test_network_id_testnet() -> None:
    a = Aleo(HTTPProvider(BASE, network="testnet"))
    assert a.network_id == 1


def test_network_name_testnet() -> None:
    a = Aleo(HTTPProvider(BASE, network="testnet"))
    assert a.network_name == "testnet"


# ---------------------------------------------------------------------------
# is_connected
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_is_connected_true() -> None:
    resp_lib.add(resp_lib.GET, f"{MAINNET_HOST}/block/height/latest", json=12345)
    a = Aleo(HTTPProvider(BASE))
    assert a.is_connected() is True


@resp_lib.activate
def test_is_connected_false_on_network_error() -> None:
    resp_lib.add(
        resp_lib.GET,
        f"{MAINNET_HOST}/block/height/latest",
        body=Exception("Connection refused"),
    )
    a = Aleo(HTTPProvider(BASE))
    assert a.is_connected() is False


@resp_lib.activate
def test_is_connected_false_on_http_error() -> None:
    resp_lib.add(
        resp_lib.GET,
        f"{MAINNET_HOST}/block/height/latest",
        status=503,
        body="Service Unavailable",
    )
    a = Aleo(HTTPProvider(BASE))
    assert a.is_connected() is False


# ---------------------------------------------------------------------------
# get_balance
# ---------------------------------------------------------------------------


@resp_lib.activate
def test_get_balance_parses_mapping_value() -> None:
    """get_balance returns the integer microcredits from the mapping."""
    resp_lib.add(
        resp_lib.GET,
        f"{MAINNET_HOST}/program/credits.aleo/mapping/account/{REAL_ADDRESS}",
        json=1234567,
    )
    a = Aleo(HTTPProvider(BASE))
    bal = a.get_balance(REAL_ADDRESS)
    assert bal == 1234567


@resp_lib.activate
def test_get_balance_empty_returns_zero() -> None:
    """get_balance returns 0 when the mapping value is null/empty."""
    resp_lib.add(
        resp_lib.GET,
        f"{MAINNET_HOST}/program/credits.aleo/mapping/account/{REAL_ADDRESS}",
        json=None,
    )
    a = Aleo(HTTPProvider(BASE))
    assert a.get_balance(REAL_ADDRESS) == 0


@resp_lib.activate
def test_get_balance_string_value() -> None:
    """get_balance handles a string-typed numeric response."""
    resp_lib.add(
        resp_lib.GET,
        f"{MAINNET_HOST}/program/credits.aleo/mapping/account/{REAL_ADDRESS}",
        json="500000",
    )
    a = Aleo(HTTPProvider(BASE))
    assert a.get_balance(REAL_ADDRESS) == 500000


@resp_lib.activate
def test_get_balance_network_error_returns_zero() -> None:
    """get_balance returns 0 on network errors rather than propagating."""
    resp_lib.add(
        resp_lib.GET,
        f"{MAINNET_HOST}/program/credits.aleo/mapping/account/{REAL_ADDRESS}",
        status=404,
        body="Not found",
    )
    a = Aleo(HTTPProvider(BASE))
    assert a.get_balance(REAL_ADDRESS) == 0


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


def test_to_microcredits_integer() -> None:
    a = Aleo(HTTPProvider(BASE))
    assert a.to_microcredits(1) == 1_000_000


def test_to_microcredits_float() -> None:
    a = Aleo(HTTPProvider(BASE))
    assert a.to_microcredits(1.5) == 1_500_000


def test_to_microcredits_zero() -> None:
    a = Aleo(HTTPProvider(BASE))
    assert a.to_microcredits(0) == 0


def test_from_microcredits_round_trip() -> None:
    a = Aleo(HTTPProvider(BASE))
    assert a.from_microcredits(1_500_000) == 1.5


def test_unit_conversion_round_trip() -> None:
    a = Aleo(HTTPProvider(BASE))
    original = 2.25
    assert a.from_microcredits(a.to_microcredits(original)) == original


# ---------------------------------------------------------------------------
# is_valid_address
# ---------------------------------------------------------------------------


def test_is_valid_address_real_address() -> None:
    a = Aleo(HTTPProvider(BASE))
    assert a.is_valid_address(REAL_ADDRESS) is True


def test_is_valid_address_junk() -> None:
    a = Aleo(HTTPProvider(BASE))
    assert a.is_valid_address("not_an_address") is False


def test_is_valid_address_empty() -> None:
    a = Aleo(HTTPProvider(BASE))
    assert a.is_valid_address("") is False


def test_is_valid_address_partial() -> None:
    a = Aleo(HTTPProvider(BASE))
    assert a.is_valid_address("aleo1") is False


# ---------------------------------------------------------------------------
# default_account settable
# ---------------------------------------------------------------------------


def test_default_account_settable() -> None:
    a = Aleo(HTTPProvider(BASE))
    assert a.default_account is None
    # Use a plain object as a stand-in for an Account
    sentinel = object()
    a.default_account = sentinel
    assert a.default_account is sentinel


# ---------------------------------------------------------------------------
# Typed exception hierarchy
# ---------------------------------------------------------------------------


def test_transaction_not_found_is_aleo_error() -> None:
    exc = TransactionNotFound("at1abc")
    assert isinstance(exc, AleoError)
    assert "at1abc" in str(exc)
    assert exc.tx_id == "at1abc"


def test_program_not_found_is_aleo_error() -> None:
    exc = ProgramNotFound("credits.aleo")
    assert isinstance(exc, AleoError)
    assert "credits.aleo" in str(exc)
    assert exc.program_id == "credits.aleo"


def test_execution_error_is_aleo_error() -> None:
    exc = ExecutionError("authorize failed", detail="snarkvm detail")
    assert isinstance(exc, AleoError)
    assert exc.detail == "snarkvm detail"


def test_transaction_confirmation_timeout_is_aleo_error() -> None:
    exc = TransactionConfirmationTimeout("at1abc", 45.0)
    assert isinstance(exc, AleoError)
    assert exc.tx_id == "at1abc"
    assert exc.timeout == 45.0


def test_except_aleo_error_catches_all_subclasses() -> None:
    """A single 'except AleoError' must catch every facade error subclass."""
    subclass_exceptions = [
        TransactionNotFound("tx1"),
        ProgramNotFound("prog.aleo"),
        ExecutionError("fail"),
        TransactionConfirmationTimeout("tx2", 30.0),
    ]
    for exc in subclass_exceptions:
        try:
            raise exc
        except AleoError:
            pass  # expected
        else:
            pytest.fail(f"{type(exc).__name__} was not caught by 'except AleoError'")


def test_internal_errors_are_aleo_errors() -> None:
    """Network/proving/scanner errors subclass AleoError so one except catches all."""
    from aleo.facade.errors import (
        AleoError,
        AleoNetworkError,
        AleoProvingError,
        RecordScannerRequestError,
        UUIDError,
    )
    for err in (AleoNetworkError, AleoProvingError, RecordScannerRequestError, UUIDError):
        assert issubclass(err, AleoError), f"{err.__name__} must subclass AleoError"
    # A raised network error is caught by `except AleoError`.
    try:
        raise AleoNetworkError("boom", status=503)
    except AleoError:
        pass
    else:
        pytest.fail("AleoNetworkError not caught by 'except AleoError'")


# ---------------------------------------------------------------------------
# Top-level aleo module exports
# ---------------------------------------------------------------------------


def test_top_level_exports() -> None:
    """Key facade types are importable directly from the top-level aleo package."""
    import aleo
    assert hasattr(aleo, "Aleo")
    assert hasattr(aleo, "HTTPProvider")
    assert hasattr(aleo, "AleoError")
    assert hasattr(aleo, "TransactionNotFound")
    assert hasattr(aleo, "ProgramNotFound")
    assert hasattr(aleo, "ExecutionError")
    assert hasattr(aleo, "TransactionConfirmationTimeout")
