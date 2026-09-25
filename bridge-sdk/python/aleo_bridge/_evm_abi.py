"""Minimal ABI fragments for the reviewed Ethereum deployments (brief §3.1, §3.2, §3.7). Pure data."""
from __future__ import annotations

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# The single EVM chain the bridge drives per environment (base/hyperevm carry only metadata-required routes).
EVM_CHAIN_BY_ENVIRONMENT = {"mainnet": "ethereum", "testnet": "sepolia"}

ERC20_ABI = [
    {"type": "function", "name": "balanceOf", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}], "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "allowance", "stateMutability": "view",
     "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
     "outputs": [{"name": "", "type": "uint256"}]},
    {"type": "function", "name": "approve", "stateMutability": "nonpayable",
     "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "", "type": "bool"}]},
]

WARP_ROUTE_ABI = [
    {"type": "function", "name": "quoteTransferRemote", "stateMutability": "view",
     "inputs": [{"name": "destination", "type": "uint32"}, {"name": "recipient", "type": "bytes32"},
                {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "quotes", "type": "tuple[]",
                  "components": [{"name": "token", "type": "address"}, {"name": "amount", "type": "uint256"}]}]},
    {"type": "function", "name": "transferRemote", "stateMutability": "payable",
     "inputs": [{"name": "destination", "type": "uint32"}, {"name": "recipient", "type": "bytes32"},
                {"name": "amount", "type": "uint256"}],
     "outputs": [{"name": "messageId", "type": "bytes32"}]},
    {"type": "event", "name": "SentTransferRemote", "anonymous": False,
     "inputs": [{"name": "destination", "type": "uint32", "indexed": True},
                {"name": "recipient", "type": "bytes32", "indexed": True},
                {"name": "amount", "type": "uint256", "indexed": False}]},
]

XRESERVE_ABI = [
    {"type": "function", "name": "depositToRemote", "stateMutability": "nonpayable",
     "inputs": [{"name": "value", "type": "uint256"}, {"name": "remoteDomain", "type": "uint32"},
                {"name": "remoteRecipient", "type": "bytes32"}, {"name": "localToken", "type": "address"},
                {"name": "maxFee", "type": "uint256"}, {"name": "hookData", "type": "bytes"}],
     "outputs": []},
    {"type": "event", "name": "DepositedToRemote", "anonymous": False,
     "inputs": [{"name": "localToken", "type": "address", "indexed": True},
                {"name": "value", "type": "uint256", "indexed": False},
                {"name": "localDepositor", "type": "address", "indexed": True},
                {"name": "remoteRecipient", "type": "bytes32", "indexed": True},
                {"name": "remoteDomain", "type": "uint32", "indexed": False},
                {"name": "remoteToken", "type": "bytes32", "indexed": False},
                {"name": "maxFee", "type": "uint256", "indexed": False},
                {"name": "hookData", "type": "bytes", "indexed": False}]},
]

MAILBOX_ABI = [
    {"type": "event", "name": "DispatchId", "anonymous": False,
     "inputs": [{"name": "messageId", "type": "bytes32", "indexed": True}]},
    {"type": "function", "name": "delivered", "stateMutability": "view",
     "inputs": [{"name": "id", "type": "bytes32"}], "outputs": [{"name": "", "type": "bool"}]},
]

__all__ = ["ERC20_ABI", "EVM_CHAIN_BY_ENVIRONMENT", "MAILBOX_ABI", "WARP_ROUTE_ABI", "XRESERVE_ABI", "ZERO_ADDRESS"]
