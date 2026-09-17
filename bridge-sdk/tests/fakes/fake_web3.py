"""A hand-rolled JSON-RPC provider behind a real ``web3.Web3``.

The real web3 contract/ABI/event/signing stack runs unchanged; only the
transport is fake, so tests exercise the exact calldata, event decoding and
raw-transaction signing that production uses. State is plain dicts the test
mutates directly.

The Aleo-side fake facade lives in ``tests/conftest.py`` (``FakeAleo``,
``bridge``/``fake_aleo`` fixtures) — this module holds only web3-specific
fakes so there is exactly one Aleo fake in the suite.
"""
from __future__ import annotations

from typing import Any, Callable

from eth_abi import decode, encode
from eth_account import Account
from eth_utils import keccak, to_checksum_address
from hexbytes import HexBytes
from web3 import Web3
from web3.providers import BaseProvider

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
BLOCK_HASH = "0x" + "cd" * 32
SELECTORS = {
    keccak(text="balanceOf(address)")[:4]: "balanceOf",
    keccak(text="allowance(address,address)")[:4]: "allowance",
    keccak(text="quoteTransferRemote(uint32,bytes32,uint256)")[:4]: "quoteTransferRemote",
    keccak(text="delivered(bytes32)")[:4]: "delivered",
}
TOPIC_DISPATCH_ID = "0x" + keccak(text="DispatchId(bytes32)").hex()
TOPIC_SENT_TRANSFER_REMOTE = "0x" + keccak(text="SentTransferRemote(uint32,bytes32,uint256)").hex()
TOPIC_DEPOSITED_TO_REMOTE = "0x" + keccak(
    text="DepositedToRemote(address,uint256,address,bytes32,uint32,bytes32,uint256,bytes)"
).hex()


def tx_hash_for(n: int) -> str:
    """Deterministic hash of the n-th (1-based) transaction the fake accepted."""
    return "0x" + keccak(text=f"fake-tx-{n}").hex()


def _hex(n: int) -> str:
    return hex(n)


def event_log(address: str, topics: list[str], data: str, *, log_index: int, tx_hash: str,
              block_number: int = 0x11) -> dict:
    return {"address": to_checksum_address(address), "topics": topics, "data": data,
            "logIndex": _hex(log_index), "blockNumber": _hex(block_number), "transactionHash": tx_hash,
            "transactionIndex": "0x0", "blockHash": BLOCK_HASH, "removed": False}


def dispatch_id_log(mailbox: str, message_id: bytes, *, tx_hash: str, log_index: int = 5, block_number: int = 0x11) -> dict:
    return event_log(mailbox, [TOPIC_DISPATCH_ID, "0x" + message_id.hex()], "0x",
                     log_index=log_index, tx_hash=tx_hash, block_number=block_number)


def sent_transfer_remote_log(router: str, *, destination: int, recipient32: bytes, amount: int, tx_hash: str,
                             log_index: int = 4, block_number: int = 0x65) -> dict:
    topics = [TOPIC_SENT_TRANSFER_REMOTE, "0x" + encode(["uint32"], [destination]).hex(), "0x" + recipient32.hex()]
    return event_log(router, topics, "0x" + encode(["uint256"], [amount]).hex(),
                     log_index=log_index, tx_hash=tx_hash, block_number=block_number)


def deposited_log(xreserve: str, *, local_token: str, depositor: str, remote_recipient32: bytes, value: int,
                  remote_domain: int, remote_token32: bytes, max_fee: int, hook_data: bytes, tx_hash: str,
                  log_index: int = 3, block_number: int = 0x65) -> dict:
    topics = [TOPIC_DEPOSITED_TO_REMOTE, "0x" + encode(["address"], [local_token]).hex(),
              "0x" + encode(["address"], [depositor]).hex(), "0x" + remote_recipient32.hex()]
    data = encode(["uint256", "uint32", "bytes32", "uint256", "bytes"],
                  [value, remote_domain, remote_token32, max_fee, hook_data])
    return event_log(xreserve, topics, "0x" + data.hex(), log_index=log_index, tx_hash=tx_hash, block_number=block_number)


def _decode_raw(raw: bytes) -> dict:
    """Signed raw tx → {to, value, data, from, nonce, gas, ...fee fields}. Typed (0x02) and legacy envelopes.

    Keeps whichever fee fields the sender actually filled in (``gasPrice`` for a
    legacy/type-0 envelope, ``maxFeePerGas``/``maxPriorityFeePerGas`` for a type-2
    one) so tests can assert on the exact values ``eth.py``'s fee-filling logic
    computed, not just that *some* transaction was sent.
    """
    from eth_account.typed_transactions import TypedTransaction

    sender = Account.recover_transaction(raw)
    if raw[0] <= 0x7F:
        fields = TypedTransaction.from_bytes(HexBytes(raw)).as_dict()
    else:
        import rlp
        from eth_account._utils.legacy_transactions import Transaction

        fields = rlp.decode(raw, Transaction).as_dict()
    data = fields.get("data", b"")
    data_hex = data if isinstance(data, str) else "0x" + bytes(data).hex()
    tx = {"to": to_checksum_address(fields["to"]), "value": int(fields.get("value", 0)), "data": data_hex, "from": sender}
    for key in ("nonce", "gas", "gasPrice", "maxFeePerGas", "maxPriorityFeePerGas"):
        if fields.get(key) is not None:
            tx[key] = int(fields[key])
    return tx


class FakeRpcProvider(BaseProvider):
    """State: balances, allowances, router quotes, delivered ids, sent txs, receipts, history logs."""

    def __init__(self, *, chain_id: int = 1, eth_balances: dict[str, int] | None = None,
                 token_balances: dict[tuple[str, str], int] | None = None,
                 allowances: dict[tuple[str, str, str], int] | None = None,
                 quotes: dict[str, list[tuple[str, int]]] | None = None,
                 delivered: set[str] | None = None, legacy: bool = False) -> None:
        super().__init__()
        self.chain_id = chain_id
        self.legacy = legacy                              # True: eth_getBlockByNumber omits baseFeePerGas
        self.eth_balances = {to_checksum_address(k): v for k, v in (eth_balances or {}).items()}
        self.token_balances = {(to_checksum_address(t), to_checksum_address(o)): v
                               for (t, o), v in (token_balances or {}).items()}
        self.allowances = {(to_checksum_address(t), to_checksum_address(o), to_checksum_address(s)): v
                           for (t, o, s), v in (allowances or {}).items()}
        self.quotes = {to_checksum_address(r): q for r, q in (quotes or {}).items()}
        self.delivered = {d.lower() for d in (delivered or set())}
        self.sent: list[dict] = []                       # {to, value, data, from, hash} in send order
        self.pending: set[str] = set()                   # hashes whose receipt stays None
        self.reverted: set[str] = set()                  # hashes whose receipt has status 0
        self.receipt_delay: dict[str, int] = {}          # hash -> remaining polls that return None before mined
        self.receipt_poll_counts: dict[str, int] = {}     # hash -> eth_getTransactionReceipt calls seen for it
        self.receipt_logs: Callable[[dict], list[dict]] = lambda tx: []   # logs for a sent tx's receipt
        self.history_logs: list[dict] = []               # served by eth_getLogs (filtered by address/fromBlock)
        self.transactions: dict[str, dict] = {}          # extra eth_getTransactionByHash answers
        self.receipts: dict[str, dict] = {}              # extra eth_getTransactionReceipt answers
        self.block_number = 0x10
        self.methods: list[str] = []

    def _ok(self, result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": 1, "result": result}

    def add_receipt(self, tx_hash: str, *, status: int = 1, logs: list[dict] | None = None, block_number: int = 0x65,
                    sender: str = ZERO_ADDRESS, to: str = ZERO_ADDRESS) -> None:
        """Serve a receipt for a hash the fake never accepted itself (recovery / status tests)."""
        self.receipts[tx_hash] = {
            "transactionHash": tx_hash, "status": _hex(status), "blockNumber": _hex(block_number), "blockHash": BLOCK_HASH,
            "transactionIndex": "0x0", "from": to_checksum_address(sender), "to": to_checksum_address(to),
            "cumulativeGasUsed": "0x1", "gasUsed": "0x1", "effectiveGasPrice": "0x1", "type": "0x2",
            "contractAddress": None, "logsBloom": "0x" + "00" * 256, "logs": logs or []}

    def add_transaction(self, tx_hash: str, *, sender: str, to: str, block_number: int = 0x65) -> None:
        """Serve eth_getTransactionByHash for a hash the fake never accepted itself."""
        self.transactions[tx_hash] = {
            "hash": tx_hash, "from": to_checksum_address(sender), "to": to_checksum_address(to), "input": "0x", "value": "0x0",
            "blockNumber": _hex(block_number), "blockHash": BLOCK_HASH, "nonce": "0x0", "gas": "0x1", "gasPrice": "0x1",
            "transactionIndex": "0x0", "type": "0x2", "chainId": _hex(self.chain_id), "v": "0x0", "r": "0x0", "s": "0x0"}

    def make_request(self, method: str, params: Any) -> dict:
        self.methods.append(method)
        if method == "eth_chainId":
            return self._ok(_hex(self.chain_id))
        if method == "eth_blockNumber":
            return self._ok(_hex(self.block_number))
        if method == "eth_gasPrice":
            return self._ok(_hex(10**9))
        if method == "eth_maxPriorityFeePerGas":
            return self._ok(_hex(10**8))
        if method == "eth_feeHistory":
            return self._ok({"baseFeePerGas": [_hex(10**9)] * 2, "gasUsedRatio": [0.5],
                             "oldestBlock": "0x1", "reward": [[_hex(10**8)]]})
        if method == "eth_getBlockByNumber":
            block = {"number": _hex(self.block_number), "gasLimit": _hex(30_000_000),
                     "gasUsed": "0x0", "timestamp": "0x0", "hash": "0x" + "ab" * 32, "parentHash": "0x" + "00" * 32,
                     "transactions": [], "difficulty": "0x0", "extraData": "0x", "logsBloom": "0x" + "00" * 256,
                     "miner": ZERO_ADDRESS, "mixHash": "0x" + "00" * 32, "nonce": "0x0000000000000000",
                     "receiptsRoot": "0x" + "00" * 32, "sha3Uncles": "0x" + "00" * 32, "size": "0x1",
                     "stateRoot": "0x" + "00" * 32, "totalDifficulty": "0x0",
                     "transactionsRoot": "0x" + "00" * 32, "uncles": []}
            if not self.legacy:
                block["baseFeePerGas"] = _hex(10**9)
            return self._ok(block)
        if method == "eth_getTransactionCount":
            return self._ok(_hex(len(self.sent)))
        if method == "eth_estimateGas":
            return self._ok(_hex(150_000))
        if method == "eth_getBalance":
            return self._ok(_hex(self.eth_balances.get(to_checksum_address(params[0]), 0)))
        if method == "eth_call":
            return self._ok(self._call(params[0]))
        if method == "eth_sendRawTransaction":
            return self._ok(self._accept(_decode_raw(bytes.fromhex(params[0][2:]))))
        if method == "eth_sendTransaction":
            p = params[0]
            raw_value = p.get("value", 0)
            value = int(raw_value, 16) if isinstance(raw_value, str) else int(raw_value)
            return self._ok(self._accept({"to": to_checksum_address(p["to"]), "value": value,
                                          "data": p.get("data", "0x"), "from": to_checksum_address(p["from"])}))
        if method == "eth_getTransactionReceipt":
            h = params[0]
            self.receipt_poll_counts[h] = self.receipt_poll_counts.get(h, 0) + 1
            return self._ok(self._receipt(h))
        if method == "eth_getLogs":
            f = params[0]
            addr = f.get("address")
            addrs = {to_checksum_address(a) for a in (addr if isinstance(addr, list) else [addr])} if addr else None
            raw_from = f.get("fromBlock", 0)
            from_block = int(raw_from, 16) if isinstance(raw_from, str) else int(raw_from)
            return self._ok([log for log in self.history_logs
                             if (addrs is None or log["address"] in addrs) and int(log["blockNumber"], 16) >= from_block])
        if method == "eth_getTransactionByHash":
            h = params[0]
            if h in self.transactions:
                return self._ok(self.transactions[h])
            sent = next((t for t in self.sent if t["hash"] == h), None)
            if sent is None:
                return self._ok(None)
            return self._ok({"hash": h, "from": sent["from"], "to": sent["to"], "input": sent["data"],
                             "value": _hex(sent["value"]), "blockNumber": _hex(self.block_number + 1),
                             "blockHash": BLOCK_HASH, "nonce": "0x0", "gas": "0x1", "gasPrice": "0x1",
                             "transactionIndex": "0x0", "type": "0x2", "chainId": _hex(self.chain_id),
                             "v": "0x0", "r": "0x0", "s": "0x0"})
        raise NotImplementedError(method)

    def _accept(self, tx: dict) -> str:
        self.sent.append(tx)
        tx["hash"] = tx_hash_for(len(self.sent))
        return tx["hash"]

    def _receipt(self, h: str) -> dict | None:
        if h in self.pending:
            return None
        delay = self.receipt_delay.get(h, 0)
        if delay > 0:
            self.receipt_delay[h] = delay - 1
            return None
        if h in self.receipts:
            return self.receipts[h]
        sent = next((t for t in self.sent if t["hash"] == h), None)
        if sent is None:
            return None
        return {"transactionHash": h, "status": "0x0" if h in self.reverted else "0x1",
                "blockNumber": _hex(self.block_number + 1), "blockHash": BLOCK_HASH, "transactionIndex": "0x0",
                "from": sent["from"], "to": sent["to"], "cumulativeGasUsed": "0x1", "gasUsed": "0x1",
                "effectiveGasPrice": "0x1", "type": "0x2", "contractAddress": None,
                "logsBloom": "0x" + "00" * 256, "logs": self.receipt_logs(sent)}

    def _call(self, call: dict) -> str:
        to = to_checksum_address(call["to"])
        data = bytes.fromhex(call["data"][2:])
        name = SELECTORS.get(data[:4])
        args = data[4:]
        if name == "balanceOf":
            (owner,) = decode(["address"], args)
            return "0x" + encode(["uint256"], [self.token_balances.get((to, to_checksum_address(owner)), 0)]).hex()
        if name == "allowance":
            owner, spender = decode(["address", "address"], args)
            key = (to, to_checksum_address(owner), to_checksum_address(spender))
            return "0x" + encode(["uint256"], [self.allowances.get(key, 0)]).hex()
        if name == "quoteTransferRemote":
            return "0x" + encode(["(address,uint256)[]"], [self.quotes.get(to, [])]).hex()
        if name == "delivered":
            (message_id,) = decode(["bytes32"], args)
            return "0x" + encode(["bool"], [("0x" + message_id.hex()) in self.delivered]).hex()
        raise NotImplementedError(f"eth_call selector {data[:4].hex()} to {to}")


def fake_web3(**config: Any) -> Web3:
    """A real ``Web3`` over ``FakeRpcProvider``; reach the state through ``w3.provider``."""
    return Web3(FakeRpcProvider(**config))


def make_bridge(*, ethereum: Any = None, environment: str | None = None, checkpoints: Any = None,
               **aleo_kwargs: Any) -> Any:
    """A ``Bridge`` over a fresh ``FakeAleo`` (mainnet unless *environment* says otherwise), wired
    with *ethereum* so ``bridge.eth`` works.

    ``checkpoints`` forwards to ``Bridge(..., checkpoints=...)`` (a ``CheckpointStore``, e.g.
    ``FileCheckpointStore``); it is never a ``FakeAleo`` constructor argument.

    Kept here (rather than in ``tests/conftest.py``) so ``eth.py`` tests can import one fixture
    factory alongside ``fake_web3`` without pulling in pytest fixtures.
    """
    from aleo_bridge import Bridge

    from tests.conftest import FakeAleo, default_mappings

    aleo_kwargs.setdefault("mappings", default_mappings())
    if environment is not None:
        aleo_kwargs.setdefault("network_name", environment)
    return Bridge(FakeAleo(**aleo_kwargs), ethereum=ethereum, environment=environment, checkpoints=checkpoints)
