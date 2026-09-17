import pytest
from eth_account import Account
from web3 import Web3
from web3.middleware import SignAndSendRawMiddlewareBuilder

from aleo_bridge.errors import ConfigurationError
from tests.fakes.fake_web3 import fake_web3, tx_hash_for

KEY = "0x" + "11" * 32
ACCT = Account.from_key(KEY)
TO = "0x0000000000000000000000000000000000000002"


def test_exactly_one_transport():
    from aleo_bridge.eth import Ethereum

    with pytest.raises(ConfigurationError, match="exactly one of rpc_url or w3"):
        Ethereum()
    with pytest.raises(ConfigurationError, match="exactly one of rpc_url or w3"):
        Ethereum("http://localhost:8545", w3=fake_web3())


def test_at_most_one_signer():
    from aleo_bridge.eth import Ethereum

    with pytest.raises(ConfigurationError, match="at most one of signer or private_key"):
        Ethereum(w3=fake_web3(), signer=ACCT, private_key=KEY)


def test_rpc_url_builds_http_provider_lazily():
    from web3 import HTTPProvider

    from aleo_bridge.eth import Ethereum

    conn = Ethereum("http://127.0.0.1:1", private_key=KEY)
    assert isinstance(conn.w3.provider, HTTPProvider)
    assert conn.address == ACCT.address and conn.can_sign


def test_w3_plus_signer_form_and_cached_chain_id():
    from aleo_bridge.eth import Ethereum

    w3 = fake_web3(chain_id=11155111)
    conn = Ethereum(w3=w3, signer=ACCT)
    assert conn.w3 is w3 and conn.address == ACCT.address and conn.can_sign
    assert conn.chain_id == 11155111
    assert conn.chain_id == 11155111 and w3.provider.methods.count("eth_chainId") == 1


def test_w3_alone_is_read_only_without_default_account():
    from aleo_bridge.eth import Ethereum

    conn = Ethereum(w3=fake_web3())
    assert conn.address is None and not conn.can_sign
    with pytest.raises(ConfigurationError, match="read-only"):
        conn.send_transaction({"to": TO, "value": 1, "data": "0x"})
    with pytest.raises(ConfigurationError, match="read-only"):
        conn.require_address()


def test_w3_default_account_uses_callers_middleware():
    from aleo_bridge.eth import Ethereum

    w3 = fake_web3()
    w3.middleware_onion.inject(SignAndSendRawMiddlewareBuilder.build(ACCT), layer=0)
    w3.eth.default_account = ACCT.address
    conn = Ethereum(w3=w3)
    assert conn.address == ACCT.address and conn.can_sign
    h = conn.send_transaction({"to": TO, "value": 1, "data": "0x"})
    assert h == tx_hash_for(1)
    assert "eth_sendRawTransaction" in w3.provider.methods          # the caller's middleware signed
    assert w3.provider.sent[0]["from"] == ACCT.address and w3.provider.sent[0]["value"] == 1


def test_local_account_path_signs_and_sends_raw():
    from aleo_bridge.eth import Ethereum

    w3 = fake_web3()
    conn = Ethereum(w3=w3, private_key=KEY)
    h = conn.send_transaction({"to": TO, "value": 7, "data": "0x"})
    assert h == tx_hash_for(1)
    assert "eth_sendRawTransaction" in w3.provider.methods and "eth_sendTransaction" not in w3.provider.methods
    sent = w3.provider.sent[0]
    assert sent["from"] == ACCT.address and sent["to"] == Web3.to_checksum_address(TO) and sent["value"] == 7
    # eth.py's fee-filling (EIP-1559 path, since the fake's eth_getBlockByNumber carries baseFeePerGas):
    #   nonce <- eth_getTransactionCount(sender, "pending")
    #   gas <- eth_estimateGas(...) * 12 // 10                       (a 20% buffer)
    #   maxPriorityFeePerGas <- eth_maxPriorityFeePerGas
    #   maxFeePerGas <- baseFeePerGas * 2 + maxPriorityFeePerGas
    assert sent["nonce"] == 0
    assert sent["gas"] == 150_000 * 12 // 10
    assert sent["maxPriorityFeePerGas"] == 10**8
    assert sent["maxFeePerGas"] == 10**9 * 2 + 10**8
    assert "gasPrice" not in sent


def test_legacy_gas_price_path_when_no_base_fee():
    """With no ``baseFeePerGas`` on the latest block (pre-EIP-1559 chain), eth.py falls back
    to a plain ``gasPrice`` from ``eth_gasPrice`` and sets no 1559 fee fields."""
    from aleo_bridge.eth import Ethereum

    w3 = fake_web3(legacy=True)
    conn = Ethereum(w3=w3, private_key=KEY)
    h = conn.send_transaction({"to": TO, "value": 3, "data": "0x"})
    assert h == tx_hash_for(1)
    sent = w3.provider.sent[0]
    assert sent["gasPrice"] == 10**9
    assert "maxFeePerGas" not in sent and "maxPriorityFeePerGas" not in sent
    assert "eth_maxPriorityFeePerGas" not in w3.provider.methods


def test_sender_mismatch_is_refused():
    from aleo_bridge.eth import Ethereum

    conn = Ethereum(w3=fake_web3(), private_key=KEY)
    with pytest.raises(ConfigurationError, match="does not match the configured account"):
        conn.send_transaction({"from": TO, "to": TO, "value": 0, "data": "0x"})


def test_wait_for_receipt_returns_none_on_timeout_and_dict_on_success():
    from aleo_bridge.eth import Ethereum

    w3 = fake_web3()
    conn = Ethereum(w3=w3, private_key=KEY)
    h = conn.send_transaction({"to": TO, "value": 0, "data": "0x"})
    w3.provider.pending.add(h)
    assert conn.wait_for_receipt(h, timeout_seconds=0.01, poll_seconds=0.001) is None
    assert conn.get_receipt(h) is None
    w3.provider.pending.clear()
    receipt = conn.wait_for_receipt(h, timeout_seconds=1.0, poll_seconds=0.001)
    assert receipt is not None and int(receipt["status"]) == 1 and int(receipt["blockNumber"]) == 0x11
    assert Web3.to_hex(conn.get_receipt(h)["transactionHash"]) == h


def test_get_receipt_missing_hash_is_none():
    from aleo_bridge.eth import Ethereum

    conn = Ethereum(w3=fake_web3())
    assert conn.get_receipt("0x" + "99" * 32) is None


def test_from_env():
    from aleo_bridge.eth import Ethereum

    assert Ethereum.from_env({}) is None
    with pytest.raises(ConfigurationError, match="both EVM_PRIVATE_KEY and ETHEREUM_RPC_URL"):
        Ethereum.from_env({"EVM_PRIVATE_KEY": KEY})
    with pytest.raises(ConfigurationError, match="both EVM_PRIVATE_KEY and ETHEREUM_RPC_URL"):
        Ethereum.from_env({"ETHEREUM_RPC_URL": "http://127.0.0.1:1"})
    conn = Ethereum.from_env({"EVM_PRIVATE_KEY": KEY, "ETHEREUM_RPC_URL": "http://127.0.0.1:1"})
    assert conn is not None and conn.address == ACCT.address and conn.w3.provider.endpoint_uri == "http://127.0.0.1:1"
