"""Minimal CCTP V2 interfaces pinned to Veil PR #148."""
from ._evm_abi import ERC20_ABI


def function(name, inputs, outputs=(), view=False):
    return {"type": "function", "name": name, "stateMutability": "view" if view else "nonpayable",
            "inputs": [{"name": n, "type": t} for n, t in inputs],
            "outputs": [{"name": "", "type": t} for t in outputs]}


def event(name, inputs):
    return {"type": "event", "name": name, "anonymous": False,
            "inputs": [{"name": n, "type": t, "indexed": i} for n, t, i in inputs]}


BURN_INPUTS = [("amount", "uint256"), ("destinationDomain", "uint32"), ("mintRecipient", "bytes32"),
               ("burnToken", "address"), ("destinationCaller", "bytes32"), ("maxFee", "uint256"),
               ("minFinalityThreshold", "uint32")]
MESSENGER_ABI = [function("depositForBurn", BURN_INPUTS),
                 function("depositForBurnWithHook", BURN_INPUTS + [("hookData", "bytes")])]
TRANSMITTER_ABI = [function("usedNonces", [("nonce", "bytes32")], ["uint256"], True),
                   function("receiveMessage", [("message", "bytes"), ("attestation", "bytes")], ["bool"]),
                   event("MessageSent", [("message", "bytes", False)]),
                   event("MessageReceived", [("caller", "address", True), ("sourceDomain", "uint32", False),
                         ("nonce", "bytes32", True), ("sender", "bytes32", False),
                         ("finalityThresholdExecuted", "uint32", True), ("messageBody", "bytes", False)])]
TOKEN_ABI = ERC20_ABI + [event("Transfer", [("from", "address", True), ("to", "address", True),
                                          ("value", "uint256", False)])]
