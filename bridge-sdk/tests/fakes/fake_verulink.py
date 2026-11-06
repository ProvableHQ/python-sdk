"""Real Web3 ABI codec with deterministic RPC and Aleo service responses."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from eth_abi import encode
from web3 import Web3
from web3.providers import BaseProvider

from aleo_bridge import Bridge, Ethereum, prepare, DEFAULT_REGISTRY
from aleo_bridge._verulink_abi import ABI
from aleo_bridge._verulink_deployment import METADATA as M
from aleo_bridge._verulink_wire import address_literal
from tests.conftest import FakeAleo

FIXTURE = json.loads((Path(__file__).parents[1] / "fixtures/verulink-mainnet.json").read_text())
ALEO = FIXTURE["toBsc"]["message"]["sender"]
EVM = FIXTURE["toAleo"]["message"]["sender"]


class Rpc(BaseProvider):
    def __init__(self):
        super().__init__()
        self.codec = Web3().eth.contract(abi=ABI)
        self.chain_id, self.height = 56, 60_000_000
        self.values = {"paused": False, "self": [int(M["evmDomain"]), M["evmTokenService"]],
                       "destChainId": int(M["aleoDomain"]), "erc20Bridge": M["evmBridge"],
                       "supportedTokens": [M["evmToken"], "0x" + "00" * 20, M["aleoTokenId"], M["aleoServiceAddress"], 10_000_000, 10**15, True],
                       "feeCollector": M["evmTokenService"], "platformFees": 100, "decimals": 6,
                       "balanceOf": 10**15, "allowance": 10**15, "quorumRequired": 3, "isAttestor": True,
                       "consumedPackets": bytes(32), "holding": "0x" + "88" * 20}
        self.calls = []
        self.transactions, self.receipts = {}, {}

    def make_request(self, method, params):
        if method == "eth_chainId":
            result = hex(self.chain_id)
        elif method == "eth_call":
            function, args = self.codec.decode_function_input(params[0]["data"])
            self.calls.append((function.fn_name, args))
            if function.fn_name in ("publicTokenSend", "tokenReceive"):
                result = "0x"
            else:
                value = True if function.fn_name == "approve" else self.values[function.fn_name]
                types = [o["type"] for o in function.abi["outputs"]]
                result = "0x" + encode(types, [value] if len(types) == 1 else value).hex()
        elif method == "eth_getBalance":
            result = hex(10**18)
        elif method == "eth_blockNumber":
            result = hex(self.height)
        elif method == "eth_getTransactionReceipt":
            result = self.receipts.get(params[0])
        elif method == "eth_getTransactionByHash":
            result = self.transactions.get(params[0])
        else:
            raise AssertionError(f"Unexpected RPC: {method}")
        return {"jsonrpc": "2.0", "id": 1, "result": result}


class Harness:
    def __init__(self, outbound=True):
        self.outbound = outbound
        self.rpc = Rpc()
        self.evm = Ethereum(w3=Web3(self.rpc))
        self.evm.w3.eth.default_account = Web3.to_checksum_address(EVM)
        self.sent, self.saved = [], []
        self.evm.send_transaction = self.send
        key = f"{{ chain_id: {M['evmDomain']}u128, token_id: {M['aleoTokenId']} }}"
        mappings = {
            M["aleoTokenService"]: {name: {key: value} for name, value in {
                "platform_fee": "100u32", "min_transfers": "10000000u64", "max_transfers": "1000000000000000u64",
                "status": "false", "other_chain_token_service": address_literal(M["evmTokenService"], True),
                "other_chain_token_address": address_literal(M["evmToken"], True)}.items()},
            M["aleoBridge"]: {"bridge_settings": {"1u8": "3u8", "3u8": "1u8"},
                "supported_chains": {M["evmDomain"] + "u128": "true"}, "supported_services": {M["aleoServiceAddress"]: "true"},
                "sequences": {M["evmDomain"] + "u128": "2u64"}, "out_packets": {
                    f"{{ chain_id: {M['evmDomain']}u128, sequence: 1u64 }}": FIXTURE["aleoPacket"]},
                "attestors": {s["signer"]: "true" for s in FIXTURE["toAleo"]["status"]["signatures"]}},
            "credits.aleo": {"account": {ALEO: "1000000000000000u64"}}}
        self.aleo = FakeAleo(mappings=mappings)
        self.aleo.default_account.address = ALEO
        self.aleo.confirmed_transactions[FIXTURE["aleoSource"]["transaction"]["id"]] = copy.deepcopy(FIXTURE["aleoSource"])
        self.aleo._client = SimpleNamespace(_get=lambda *args: "ab1historical")
        self.aleo.network.get_block_by_hash = lambda *args: {"header": {"metadata": {"height": 10461039}}}
        tx = copy.deepcopy(FIXTURE["bscSource"]["tx"])
        tx["blockNumber"] = hex(int(tx["blockNumber"]))
        receipt = copy.deepcopy(FIXTURE["bscSource"]["receipt"])
        receipt["blockNumber"] = hex(int(receipt["blockNumber"]))
        receipt["status"] = "0x1"
        self.rpc.transactions[tx["hash"]] = tx
        self.rpc.receipts[tx["hash"]] = receipt
        self.bridge = Bridge(self.aleo, evm={"bsc": self.evm})
        self.rows = [copy.deepcopy(FIXTURE["toBsc" if outbound else "toAleo"])]
        self.bridge.verulink.session = SimpleNamespace(get=lambda *args, **kw: SimpleNamespace(
            status_code=200, json=lambda: {"data": {"currentPage": 1, "totalPages": 1, "data": self.rows}}))
        self.plan = prepare(DEFAULT_REGISTRY, source_chain="aleo" if outbound else "bsc", source_asset="aleo",
                            destination_chain="bsc" if outbound else "aleo", amount="10",
                            recipient=self.rows[0]["message"]["receiver"], sender=self.rows[0]["message"]["sender"])

    def send(self, tx):
        self.sent.append(tx)
        return "0x" + f"{len(self.sent):064x}"

    def quote(self, amount="11.000001"):
        return self.bridge.quote(route=self.plan.route_id, amount=amount, recipient=self.plan.recipient, sender=self.plan.sender)
