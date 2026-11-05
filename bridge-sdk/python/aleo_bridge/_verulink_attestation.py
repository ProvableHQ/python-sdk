"""Verify current attestor membership and signed screening votes locally."""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote, urlsplit

from .errors import BridgeError
from ._verulink_wire import (Packet, ZERO_ALEO, address, aleo_packet_literal, collector_packet, decimal,
                             integer, obj, packet_hash, same, same_packet)


def fetch_signatures(c: Any, session: Any, packet: Packet) -> list[dict[str, str]]:
    base = urlsplit(c.collector_url)
    if base.scheme != "https" or not base.netloc or base.username or base.password or base.query or base.fragment:
        raise BridgeError("Verulink collector requires an HTTPS URL without credentials")
    for page in range(1, 101):
        url = f"{c.collector_url.rstrip('/')}/packet/wallet/{quote(packet['sender'], safe='')}?page={page}"
        response = session.get(url, timeout=30)
        if response.status_code != 200:
            raise BridgeError(f"Verulink signature collector returned HTTP {response.status_code}")
        body = obj(obj(response.json()).get("data"))
        rows, pages = body.get("data"), body.get("totalPages")
        if (type(body.get("currentPage")) is not int or body["currentPage"] != page
                or type(pages) is not int or pages < 0 or not isinstance(rows, list) or len(rows) > 1000):
            raise BridgeError("Invalid Verulink collector pagination")
        for value in rows:
            row = obj(value)
            if (str(row.get("sourceChain")) != packet["sourceChain"]
                    or str(row.get("destinationChain")) != packet["destinationChain"]
                    or decimal(row.get("sequence")) != packet["sequence"]):
                continue
            if not same_packet(packet, collector_packet(row)):
                raise BridgeError("Verulink collector packet does not match confirmed source data")
            signatures = obj(row.get("status")).get("signatures")
            if not isinstance(signatures, list) or len(signatures) > 100:
                raise BridgeError("Invalid Verulink signature list")
            return [signature(value) for value in signatures]
        if page >= pages:
            return []
    raise BridgeError("Verulink collector history exceeds the 100-page lookup bound")


def signature(value: Any) -> dict[str, str]:
    value = obj(value)
    if not isinstance(value.get("signer"), str) or not isinstance(value.get("sign"), str):
        raise BridgeError("Invalid Verulink attestor signature")
    return {"signer": value["signer"], "sign": value["sign"]}


def _aleo_votes(c: Any, packet: Packet, signatures: list[dict[str, str]]) -> list[bool]:
    from importlib import import_module
    sdk = import_module(f"aleo.{c.network}")
    bhp = sdk.BHP256()
    digest = bhp.hash(sdk.Plaintext.from_string(aleo_packet_literal(packet)).to_bits_le())
    messages = [str(bhp.hash(sdk.Plaintext.from_string(
        f"{{ packet_hash: {digest}, screening_passed: {vote} }}").to_bits_le())) for vote in ("true", "false")]
    result = []
    for value in signatures:
        try:
            addr = sdk.Address.from_string(value["signer"])
            sign = sdk.Signature.from_string(value["sign"])
            if sign.verify_value(addr, messages[0]):
                result.append(True)
            elif sign.verify_value(addr, messages[1]):
                result.append(False)
            else:
                raise ValueError("unverified signature")
        except ValueError as exc:
            raise BridgeError("Invalid Verulink Aleo attestor signature") from exc
    return result


def _evm_votes(packet: Packet, signatures: list[dict[str, str]]) -> list[bool]:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    from eth_utils.crypto import keccak
    digest = packet_hash(packet)
    result = []
    for value in signatures:
        for vote in (1, 2):
            try:
                recovered = Account.recover_message(encode_defunct(keccak(digest + bytes([vote]))), signature=value["sign"])
            except (ValueError, TypeError) as exc:
                raise BridgeError("Invalid Verulink EVM attestor signature") from exc
            if same(recovered, value["signer"]):
                result.append(vote == 1)
                break
        else:
            raise BridgeError("Invalid Verulink EVM attestor signature")
    return result


def verified_signatures(c: Any, packet: Packet, signatures: list[dict[str, str]]) -> list[dict[str, str]] | None:
    threshold = integer(c.read(c.evm_bridge, "quorumRequired") if c.outbound else
                        c.mapping(c.aleo_bridge, "bridge_settings", "1u8"), "attestor quorum", 100 if c.outbound else 5)
    if threshold < 1:
        raise BridgeError("Verulink attestor quorum cannot be zero")
    if len(signatures) < threshold:
        return None
    if len(signatures) > (100 if c.outbound else 5):
        raise BridgeError("Verulink attestor array exceeds its bound")
    seen = set()
    for raw in signatures:
        value = signature(raw)
        signer, signed = value["signer"], value["sign"]
        key = signer.lower() if c.outbound else signer
        if key in seen or signer == ZERO_ALEO or signer.lower() == "0x" + "00" * 20:
            raise BridgeError("Duplicate or zero Verulink attestor")
        seen.add(key)
        if c.outbound:
            if not re.fullmatch(r"0x[0-9a-fA-F]{40}", signer) or not re.fullmatch(r"0x[0-9a-fA-F]{130}", signed):
                raise BridgeError("Invalid EVM attestor signature encoding")
            member = c.read(c.evm_bridge, "isAttestor", [address(signer)]) is True
        else:
            if not re.fullmatch(r"aleo1[0-9a-z]{58}", signer) or not re.fullmatch(r"sign1[0-9a-z]+", signed):
                raise BridgeError("Invalid Aleo attestor signature encoding")
            member = c.mapping(c.aleo_bridge, "attestors", signer) is not None
        if not member:
            raise BridgeError("Unknown Verulink attestor")
    votes = _evm_votes(packet, signatures) if c.outbound else _aleo_votes(c, packet, signatures)
    positive = [value for value, vote in zip(signatures, votes) if vote]
    if len(signatures) - len(positive) >= threshold:
        raise BridgeError("Verulink attestors screened this packet negatively; public delivery is unavailable")
    return sorted(positive, key=lambda x: x["signer"].lower()) if len(positive) >= threshold else None
