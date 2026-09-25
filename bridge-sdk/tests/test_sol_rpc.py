import base64
import json

import pytest

pytest.importorskip("solders")
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import MessageV0, to_bytes_versioned
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.compute_budget import set_compute_unit_limit

from aleo_bridge import sol
from aleo_bridge.errors import BridgeError

SIG = str(Signature.from_bytes(bytes([9]) * 64))
ADDR = "4LZtvKvBAM8Hcf5tuL5R7xYj9JC12v6ho8igDnwzo6WC"
HASH = "8YGT2pZwyZe94qBpGzWfY2TMEVcwaQ1bXAE7YAgpUaM7"


class FakeResponse:
    def __init__(self, body, status_code=200, invalid_json=False):
        self._body, self.status_code, self._invalid = body, status_code, invalid_json

    def json(self):
        if self._invalid:
            raise ValueError("bad json")
        return self._body


class FakeSession:
    """Answers each POST from a queue of bodies (the last repeats) and records every request."""

    def __init__(self, *bodies, status_code=200, invalid_json=False):
        self.bodies = list(bodies) or [{"jsonrpc": "2.0", "id": 1, "result": None}]
        self.status_code, self.invalid_json = status_code, invalid_json
        self.calls: list[dict] = []

    def post(self, url, *, json, timeout, headers):
        self.calls.append({"url": url, "json": json, "timeout": timeout, "headers": headers})
        body = self.bodies.pop(0) if len(self.bodies) > 1 else self.bodies[0]
        return FakeResponse(body, self.status_code, self.invalid_json)

    def last(self):
        return self.calls[-1]["json"]["method"], self.calls[-1]["json"]["params"]


def ok(result):
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def ctx(value):
    return ok({"context": {"slot": 1}, "value": value})


def client(*bodies, **kw):
    session = FakeSession(*bodies, **kw)
    return sol.SolanaRpcClient("http://rpc.test", session=session), session


def test_request_envelope_and_defaults():
    rpc, session = client(ok(42))
    assert rpc.get_block_height().value == 42
    call = session.calls[0]
    assert call["url"] == "http://rpc.test" and call["timeout"] == 30.0
    assert call["json"]["jsonrpc"] == "2.0" and call["headers"]["cache-control"] == "no-cache"
    assert session.last() == ("getBlockHeight", [{"commitment": "confirmed"}])
    assert rpc.commitment == "confirmed" and rpc.url == "http://rpc.test"


def test_latest_blockhash():
    rpc, session = client(ctx({"blockhash": HASH, "lastValidBlockHeight": 123456789}))
    value = rpc.get_latest_blockhash().value
    assert value.blockhash == Hash.from_string(HASH) and value.last_valid_block_height == 123456789
    assert session.last() == ("getLatestBlockhash", [{"commitment": "confirmed"}])
    rpc2, _ = client(ctx({"blockhash": "", "lastValidBlockHeight": 1}))
    with pytest.raises(BridgeError, match="getLatestBlockhash returned an invalid result"):
        rpc2.get_latest_blockhash()


def test_balance_and_rent_and_blockhash_validity():
    rpc, session = client(ctx(1_000_000_000))
    assert rpc.get_balance(Pubkey.from_string(ADDR)).value == 1_000_000_000
    assert session.last() == ("getBalance", [ADDR, {"commitment": "confirmed"}])
    rpc, session = client(ok(890_880))
    assert rpc.get_minimum_balance_for_rent_exemption(0).value == 890_880
    assert session.last() == ("getMinimumBalanceForRentExemption", [0, {"commitment": "confirmed"}])
    with pytest.raises(BridgeError, match="non-negative integer"):
        rpc.get_minimum_balance_for_rent_exemption(-1)
    rpc, session = client(ctx(True))
    assert rpc.is_blockhash_valid(Hash.from_string(HASH)).value is True
    assert session.last() == ("isBlockhashValid", [HASH, {"commitment": "confirmed"}])
    rpc, _ = client(ctx(-1))
    with pytest.raises(BridgeError, match="getBalance returned an invalid result"):
        rpc.get_balance(Pubkey.from_string(ADDR))


def test_account_info():
    encoded = base64.b64encode(bytes([1, 2, 3, 4])).decode()
    rpc, session = client(ctx({"data": [encoded, "base64"], "owner": ADDR, "lamports": 1, "executable": False, "rentEpoch": 0}))
    account = rpc.get_account_info(Pubkey.from_string(ADDR)).value
    assert account.data == bytes([1, 2, 3, 4]) and account.lamports == 1 and account.owner == ADDR
    assert session.last() == ("getAccountInfo", [ADDR, {"encoding": "base64", "commitment": "confirmed"}])
    rpc, _ = client(ctx(None))
    assert rpc.get_account_info(Pubkey.from_string(ADDR)).value is None
    rpc, _ = client(ctx({"data": ["abc", "base58"], "owner": ADDR, "lamports": 1}))
    with pytest.raises(BridgeError, match="invalid base64 account data"):
        rpc.get_account_info(Pubkey.from_string(ADDR))


def test_fee_for_message_sends_the_versioned_message_bytes():
    payer = Keypair()
    message = MessageV0.try_compile(payer.pubkey(), [set_compute_unit_limit(400_000)], [], Hash.from_string(HASH))
    rpc, session = client(ctx(10_000))
    assert rpc.get_fee_for_message(message).value == 10_000
    method, params = session.last()
    assert method == "getFeeForMessage"
    assert params == [base64.b64encode(to_bytes_versioned(message)).decode(), {"commitment": "confirmed"}]
    rpc, _ = client(ctx(None))
    assert rpc.get_fee_for_message(message).value is None


def test_send_raw_transaction_params_and_preflight_error_details():
    rpc, session = client(ok(SIG))
    assert rpc.send_raw_transaction(b"\x01\x02\x03").value == Signature.from_string(SIG)
    method, params = session.last()
    assert method == "sendTransaction"
    assert params == [base64.b64encode(b"\x01\x02\x03").decode(), {"encoding": "base64", "skipPreflight": False, "preflightCommitment": "confirmed"}]
    rpc, session = client(ok(SIG))
    rpc.send_raw_transaction(b"\x01", sol.SendOptions(skip_preflight=True, preflight_commitment="processed"))
    assert session.last()[1][1] == {"encoding": "base64", "skipPreflight": True, "preflightCommitment": "processed"}
    failing = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32002, "message": "Transaction simulation failed",
                                                     "data": {"err": {"InstructionError": [0, "Custom"]}, "logs": ["Program log: insufficient lamports"]}}}
    rpc, _ = client(failing)
    with pytest.raises(BridgeError, match="insufficient lamports"):
        rpc.send_raw_transaction(b"\x01")


def test_signature_statuses():
    rpc, session = client(ctx([None]))
    assert rpc.get_signature_statuses([Signature.from_string(SIG)], search_transaction_history=True).value == [None]
    assert session.last() == ("getSignatureStatuses", [[SIG], {"searchTransactionHistory": True}])
    rpc, _ = client(ctx([{"err": {"InstructionError": [0, "Custom"]}, "confirmationStatus": "processed", "slot": 1, "confirmations": None}]))
    status = rpc.get_signature_statuses([Signature.from_string(SIG)]).value[0]
    assert status.err == {"InstructionError": [0, "Custom"]} and status.confirmation_status == "processed"
    rpc, _ = client(ctx([{"err": None, "confirmationStatus": "finalized", "slot": 1, "confirmations": None}]))
    assert rpc.get_signature_statuses([Signature.from_string(SIG)]).value[0].confirmation_status == "finalized"
    rpc, _ = client(ctx([{"err": None, "confirmationStatus": "mystery", "slot": 1}]))
    with pytest.raises(BridgeError, match="unsupported confirmation status"):
        rpc.get_signature_statuses([Signature.from_string(SIG)])
    rpc, _ = client(ctx([{}]))
    with pytest.raises(BridgeError, match="invalid status"):
        rpc.get_signature_statuses([Signature.from_string(SIG)])


def test_get_transaction():
    rpc, session = client(ok({"slot": 5, "meta": {"logMessages": ["Program log: hi"]}, "transaction": {}}))
    value = rpc.get_transaction(Signature.from_string(SIG), max_supported_transaction_version=0).value
    assert value.slot == 5 and value.transaction.meta.log_messages == ["Program log: hi"]
    assert session.last() == ("getTransaction", [SIG, {"encoding": "json", "commitment": "confirmed", "maxSupportedTransactionVersion": 0}])
    rpc, _ = client(ok(None))
    assert rpc.get_transaction(Signature.from_string(SIG)).value is None
    rpc, _ = client(ok({"slot": 5, "meta": None, "transaction": {}}))
    assert rpc.get_transaction(Signature.from_string(SIG)).value.transaction.meta is None
    rpc, _ = client(ok({"slot": 5, "meta": {"logMessages": "not-a-list"}}))
    with pytest.raises(BridgeError, match="invalid logs"):
        rpc.get_transaction(Signature.from_string(SIG))


def test_transport_errors_are_bridge_errors():
    rpc, _ = client({}, status_code=500)
    with pytest.raises(BridgeError, match="500"):
        rpc.get_block_height()
    rpc, _ = client({"jsonrpc": "2.0", "id": 1, "error": {"code": -32602, "message": "Invalid param"}})
    with pytest.raises(BridgeError, match="Invalid param"):
        rpc.get_block_height()
    rpc, _ = client({}, invalid_json=True)
    with pytest.raises(BridgeError, match="invalid JSON"):
        rpc.get_block_height()
    rpc, _ = client({"jsonrpc": "2.0", "id": 1})
    with pytest.raises(BridgeError, match="invalid result envelope"):
        rpc.get_block_height()
    rpc, _ = client(ok("1"))
    with pytest.raises(BridgeError, match="invalid result"):
        rpc.get_block_height()
