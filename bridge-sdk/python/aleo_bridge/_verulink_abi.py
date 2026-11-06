"""Verulink service wire ABI (Veil PR 217, a2071c0b)."""
from typing import Any


def field(name: str, kind: str, components: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"name": name, "type": kind, **({"components": components} if components else {})}


def packet(outgoing: bool) -> list[dict[str, Any]]:
    src, dst = ("address", "string") if outgoing else ("string", "address")
    return [field("version", "uint256"), field("sequence", "uint256"),
            field("sourceTokenService", "tuple", [field("chainId", "uint256"), field("addr", src)]),
            field("destTokenService", "tuple", [field("chainId", "uint256"), field("addr", dst)]),
            field("message", "tuple", [field("senderAddress", src), field("destTokenAddress", dst),
                                       field("amount", "uint256"), field("receiverAddress", dst)]),
            field("height", "uint256")]


def function(name: str, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]], *, write: bool = False) -> dict[str, Any]:
    return {"type": "function", "name": name, "stateMutability": "nonpayable" if write else "view",
            "inputs": inputs, "outputs": outputs}


def event(name: str, inputs: list[dict[str, Any]], indexed: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"type": "event", "name": name, "anonymous": False,
            "inputs": [{**x, "indexed": x["name"] in indexed} for x in inputs]}


ABI = [
    function("publicTokenSend", [field("tokenAddress", "address"), field("amount", "uint256"), field("receiver", "string")], [], write=True),
    function("tokenReceive", [field("packet", "tuple", packet(False)), field("signatures", "bytes")], [], write=True),
    function("self", [], [field("chainId", "uint256"), field("addr", "address")]),
    function("supportedTokens", [field("token", "address")], [
        field("tokenAddress", "address"), field("vault", "address"), field("destTokenAddress", "string"),
        field("destTokenService", "string"), field("minValue", "uint256"), field("maxValue", "uint256"), field("enabled", "bool")]),
    function("platformFees", [field("token", "address")], [field("", "uint256")]),
    function("balanceOf", [field("owner", "address")], [field("", "uint256")]),
    function("allowance", [field("owner", "address"), field("spender", "address")], [field("", "uint256")]),
    function("approve", [field("spender", "address"), field("amount", "uint256")], [field("", "bool")], write=True),
    function("isAttestor", [field("signer", "address")], [field("", "bool")]),
    function("consumedPackets", [field("sequence", "uint256")], [field("", "bytes32")]),
    event("PacketDispatched", [field("packet", "tuple", packet(True)), field("data", "bytes")]),
    event("Consumed", [field("chainId", "uint256"), field("sequence", "uint256"), field("packetHash", "bytes32"), field("quorum", "uint8")]),
    event("Transfer", [field("from", "address"), field("to", "address"), field("value", "uint256")], ("from", "to")),
]
for _name, _type in (("paused", "bool"), ("destChainId", "uint256"), ("erc20Bridge", "address"),
                     ("holding", "address"), ("feeCollector", "address"), ("decimals", "uint8"), ("quorumRequired", "uint256")):
    ABI.append(function(_name, [], [field("", _type)]))
