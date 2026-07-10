"""Tests for F5 facade: the bound-call verb ladder (aleo.facade.call).

All tests here are FAST and OFFLINE:

* ``authorize`` / ``simulate`` / ``call`` are purely local (no proof, no
  network), so they run for real.
* ``delegate`` is exercised with the DPS submit MOCKED — the main authorization
  is built locally (fast), only the network POST is stubbed.
* ``build_transaction`` / ``transact`` (which prepare a trace against the live
  endpoint and prove) live in the slow e2e suite, not here.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from aleo import Aleo, HTTPProvider
from aleo.mainnet import Program as RawProgram, PrivateKey
from aleo.facade.programs import Program, PreparedCall
from aleo.facade.call import BoundCall, AuthorizationResult, TransactionResult
from aleo.facade.errors import ExecutionError

BASE = "https://api.provable.com/v2"

CREDITS_SOURCE = RawProgram.credits().source


def _client(network: str = "mainnet") -> Aleo:
    return Aleo(HTTPProvider(BASE, network=network))


def _credits_program(a: Aleo) -> Program:
    raw = RawProgram.from_source(CREDITS_SOURCE)
    return Program(a, raw)


def _account(a: Aleo) -> Any:
    return a.account.from_private_key(PrivateKey.random())


def _bound(a: Aleo, acct: Any) -> BoundCall:
    prog = _credits_program(a)
    bc = prog.functions.transfer_public(str(acct.address), 10)
    assert isinstance(bc, BoundCall)
    return bc


# ---------------------------------------------------------------------------
# Wiring: functions.<name>(...) returns a BoundCall (a PreparedCall subclass)
# ---------------------------------------------------------------------------


def test_functions_return_bound_call() -> None:
    a = _client()
    acct = _account(a)
    bc = _bound(a, acct)
    assert isinstance(bc, BoundCall)
    assert isinstance(bc, PreparedCall)  # inherits F4 coercion + slots
    assert bc.program_id == "credits.aleo"
    assert bc.function_name == "transfer_public"
    assert bc.args == [str(acct.address), "10u64"]
    assert bc.signature == "transfer_public(address, u64)"


# ---------------------------------------------------------------------------
# authorize / simulate / call — local, inspectable
# ---------------------------------------------------------------------------


def test_authorize_builds_authorization_result() -> None:
    a = _client()
    acct = _account(a)
    bc = _bound(a, acct)

    result = bc.authorize(acct)
    assert isinstance(result, AuthorizationResult)
    assert result.function_name == "transfer_public"
    assert result.execution_id is not None
    transitions = result.transitions()
    assert len(transitions) == 1
    assert str(transitions[0].function_name) == "transfer_public"


def test_authorization_result_inspection() -> None:
    a = _client()
    acct = _account(a)
    bc = _bound(a, acct)

    result = bc.authorize(acct)
    outputs = result.outputs
    assert isinstance(outputs, list) and len(outputs) == 1
    decoded = result.decoded()
    assert len(decoded) == 1
    entry = decoded[0]
    assert entry["program"] == "credits.aleo"
    assert entry["function"] == "transfer_public"
    assert isinstance(entry["inputs"], list)
    assert isinstance(entry["outputs"], list)
    # transfer_public produces a single future output.
    assert "transfer_public" in str(entry["outputs"])
    # .raw hatch is the underlying network Authorization.
    assert result.raw is not None
    assert str(result.raw.function_name()) == "transfer_public"


def test_simulate_and_call_are_authorize_aliases() -> None:
    a = _client()
    acct = _account(a)
    bc = _bound(a, acct)

    r_auth = bc.authorize(acct)
    r_sim = bc.simulate(acct)
    r_call = bc.call(acct)
    # Equal in every observable, structure-level way (raw strings differ only
    # by per-authorize randomness).
    for other in (r_sim, r_call):
        assert other.function_name == r_auth.function_name
        assert len(other.transitions()) == len(r_auth.transitions())
        assert other.decoded()[0]["program"] == r_auth.decoded()[0]["program"]
        assert other.decoded()[0]["function"] == r_auth.decoded()[0]["function"]


# ---------------------------------------------------------------------------
# Account defaulting
# ---------------------------------------------------------------------------


def test_authorize_uses_default_account() -> None:
    a = _client()
    acct = _account(a)
    a.default_account = acct
    bc = _bound(a, acct)

    result = bc.authorize()  # no account arg → default_account
    assert result.function_name == "transfer_public"


def test_authorize_errors_without_account() -> None:
    a = _client()
    acct = _account(a)
    bc = _bound(a, acct)  # note: a.default_account is None
    with pytest.raises(ValueError, match="default_account is not set"):
        bc.authorize()


# ---------------------------------------------------------------------------
# delegate — DPS submit MOCKED
# ---------------------------------------------------------------------------


def _mock_submit(a: Aleo) -> MagicMock:
    """Replace the network client's submit_proving_request with a mock."""
    mock = MagicMock(return_value={"transaction_id": "at1mockdelegate"})
    a._client.submit_proving_request = mock  # type: ignore[attr-defined]
    return mock


def test_delegate_default_fee_master_pays() -> None:
    """DEFAULT: fee_authorization=None → the prover's fee master pays."""
    a = _client()
    acct = _account(a)
    bc = _bound(a, acct)
    mock = _mock_submit(a)

    out = bc.delegate(acct)

    assert out == {"transaction_id": "at1mockdelegate"}
    mock.assert_called_once()
    request = mock.call_args.args[0]
    # The submitted ProvingRequest carries NO fee authorization by default.
    assert request.fee_authorization() is None
    # …and the main authorization is the transfer_public call.
    assert str(request.authorization().function_name()) == "transfer_public"
    assert request.broadcast is True


def test_delegate_broadcast_flag_propagates() -> None:
    a = _client()
    acct = _account(a)
    bc = _bound(a, acct)
    mock = _mock_submit(a)

    bc.delegate(acct, broadcast=False)
    request = mock.call_args.args[0]
    assert request.broadcast is False
    assert request.fee_authorization() is None


def test_delegate_pay_own_fee_attaches_fee_authorization() -> None:
    """pay_own_fee=True → a self-paid public fee authorization is attached.

    Binding a self-paid fee needs the execution id, which normally comes from a
    real prove (slow + network).  We wrap the real ``Process`` in a light shim
    that stubs only the network-touching execute/prepare/prove-execution chain
    and delegates ``authorize_fee_public`` / ``execution_cost`` to the real
    process, so the produced fee authorization is genuine.
    """
    a = _client()
    acct = _account(a)
    bc = _bound(a, acct)
    mock = _mock_submit(a)

    from aleo.mainnet import Field

    real_process = a.process

    class _FakeExecution:
        @property
        def execution_id(self) -> Any:
            return Field.zero()

    class _FakeTrace:
        def prepare(self, _query: Any) -> None:
            return None

        def prove_execution(self, _locator: str) -> Any:
            return _FakeExecution()

    class _ProcessShim:
        def authorize(self, *args: Any) -> Any:
            return real_process.authorize(*args)

        def execute(self, _auth: Any) -> Any:
            return (None, _FakeTrace())

        def execution_cost(self, _execution: Any) -> Any:
            return (1000, (900, 100))

        def authorize_fee_public(self, *args: Any) -> Any:
            return real_process.authorize_fee_public(*args)

    # Swap the lazily-cached process for the shim (property returns _process).
    a._process = _ProcessShim()  # type: ignore[attr-defined]

    bc.delegate(acct, pay_own_fee=True)
    request = mock.call_args.args[0]
    # A real (public) fee authorization is now attached.
    fee_auth = request.fee_authorization()
    assert fee_auth is not None
    assert fee_auth.is_fee_public() is True


# ---------------------------------------------------------------------------
# Private fee with no record provider → clear facade error
# ---------------------------------------------------------------------------


def test_private_fee_without_record_provider_errors() -> None:
    a = _client()
    acct = _account(a)
    bc = _bound(a, acct)
    # As of F6 record_provider defaults to aleo.records; the "no provider"
    # error path requires the user to have explicitly cleared it.
    a.record_provider = None
    assert a.record_provider is None

    # Reach the fee-sourcing directly (avoids proving): resolve a private fee
    # record with no provider set.
    with pytest.raises(ExecutionError, match="record provider"):
        bc._resolve_fee_record(None)


# ---------------------------------------------------------------------------
# decode_transition on a locally-built transition (fast, no prove)
# ---------------------------------------------------------------------------


def test_decode_transition_on_transition_object() -> None:
    a = _client()
    acct = _account(a)
    bc = _bound(a, acct)
    auth = bc.authorize(acct)
    transition = auth.transitions()[0]

    decoded = a.decode_transition(transition)
    assert decoded["program"] == "credits.aleo"
    assert decoded["function"] == "transfer_public"
    assert isinstance(decoded["inputs"], list)
    assert isinstance(decoded["outputs"], list)


def test_decode_transition_by_tx_id_uses_network() -> None:
    """A string id fetches the tx via network_client and decodes a transition."""
    a = _client()
    acct = _account(a)
    bc = _bound(a, acct)
    transition = bc.authorize(acct).transitions()[0]
    tid = str(transition.id)

    fake_tx = MagicMock()
    fake_tx.transitions.return_value = [transition]
    a._client.get_transaction_object = MagicMock(return_value=fake_tx)  # type: ignore[attr-defined]

    decoded = a.decode_transition(tid)
    assert decoded["function"] == "transfer_public"
    a._client.get_transaction_object.assert_called_once_with(tid)  # type: ignore[attr-defined]
