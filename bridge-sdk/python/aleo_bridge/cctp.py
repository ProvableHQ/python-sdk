"""Move native USDC between Arc and Ethereum, Base, or Arbitrum through Circle CCTP V2.

_source: ProvableHQ/veil packages/bridge/src/protocols/cctp/evm.ts @
3c3b457bd5f63620657321893a2487e489750d24. Network effects stay behind lifecycle verbs.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace
from typing import Any, TYPE_CHECKING, cast
from collections.abc import Callable, Iterator, Iterable

import requests

from ._cctp_message import address_bytes, validate_message, immutable_message, FORWARD_HOOK, CctpMessage
from ._cctp_abi import MESSENGER_ABI, TRANSMITTER_ABI, TOKEN_ABI
from .errors import AttestationError, ConfigurationError, InvalidAmountError, BridgeError
from .registry import Route
from .types import CctpOptions, EvmCctpQuote, Fee, Plan, Receipt, Status, TERMINAL, normalize_cctp
from .units import format_decimal_amount, parse_decimal_amount

if TYPE_CHECKING:
    from .checkpoint import Checkpoint
    from .eth import Ethereum


@dataclass(frozen=True)
class Metadata:
    route: Route
    source_chain: str
    destination_chain: str
    source_chain_id: int
    destination_chain_id: int
    source_domain: int
    destination_domain: int
    messenger: str
    transmitter: str
    source_token: str
    destination_token: str
    attestation_url: str


class CctpCheckpointError(BridgeError):
    """A transaction may be submitted; recover using ``checkpoint``, never execute again."""

    def __init__(self, tx_hash: str, checkpoint: Checkpoint) -> None:
        super().__init__(f"CCTP transaction {tx_hash} was submitted but checkpoint notification failed; recover the attached checkpoint")
        self.broadcast_id = tx_hash
        self.checkpoint = checkpoint


def _uint(value: Any, field: str, maximum: int = 2**256 - 1) -> int:
    if (isinstance(value, bool) or not isinstance(value, (int, str))
            or not re.fullmatch(r"[0-9]+", str(value)) or int(value) > maximum):
        raise ConfigurationError(f"Invalid CCTP {field}")
    return int(value)


def _basis_points(amount: int, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str, float)) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", str(value)):
        raise AttestationError("Invalid Circle minimumFee")
    whole, _, fraction = str(value).partition(".")
    denominator = 10 ** len(fraction) * 10_000
    return (amount * int(whole + fraction) + denominator - 1) // denominator


class CctpModule:
    """Quote CCTP fees and track delivery through the bridge's shared lifecycle."""

    def __init__(self, bridge: Any) -> None:
        self.bridge = bridge
        self.circle_session: Any = None

    def _metadata(self, plan: Plan) -> Metadata:
        from .lifecycle import resolve_route
        resolved = resolve_route(self.bridge.registry, plan)
        route, src, dst = resolved.route, resolved.source_asset, resolved.destination_asset
        if plan.protocol != "cctp" or not route.active or plan.environment != self.bridge.environment:
            raise ConfigurationError("CCTP plan does not match an active route in this environment")
        for asset, chain in ((src, resolved.source_chain), (dst, resolved.destination_chain)):
            if (chain.family != "evm" or asset.key != "usdc" or asset.decimals != 6
                    or asset.locator is None or asset.locator.kind != "evm-contract"):
                raise ConfigurationError("CCTP requires canonical six-decimal EVM USDC")
            address_bytes(asset.locator.value)
        address_bytes(plan.recipient)
        if plan.sender is not None:
            address_bytes(plan.sender)
        amount = parse_decimal_amount(plan.amount, 6)
        if type(plan.amount_atomic) is not int or amount != plan.amount_atomic or not 0 < amount < 2**256:
            raise InvalidAmountError("CCTP amount must be a matching positive uint256")
        if plan.mint_mode != "public":
            raise ConfigurationError("CCTP does not support Aleo mint modes")
        options = normalize_cctp(plan.cctp)
        if options.max_fee is not None and parse_decimal_amount(options.max_fee, 6) >= amount:
            raise InvalidAmountError("CCTP max_fee must be less than the burn amount")
        for chain, key in ((resolved.source_chain, "sourceDomain"), (resolved.destination_chain, "destinationDomain")):
            domain = chain.protocol_domains.get("cctp")
            if type(domain) is not int or route.meta_int(key) != domain:
                raise ConfigurationError("CCTP route domains must match configured chain domains")
        messenger, transmitter = route.meta_str("tokenMessenger"), route.meta_str("messageTransmitter")
        address_bytes(messenger)
        address_bytes(transmitter)
        url = route.meta_str("attestationBaseUrl")
        if not url.startswith("https://"):
            raise ConfigurationError("CCTP attestation URL must use HTTPS")
        assert src.locator is not None and dst.locator is not None
        return Metadata(route, src.chain_id, dst.chain_id,
                        _uint(route.meta_int("sourceChainId"), "source chain", 2**32 - 1),
                        _uint(route.meta_int("destinationChainId"), "destination chain", 2**32 - 1),
                        route.meta_int("sourceDomain"), route.meta_int("destinationDomain"),
                        messenger, transmitter, src.locator.value, dst.locator.value, url.rstrip("/"))

    def _json(self, url: str) -> Any:
        if self.circle_session is None:
            self.circle_session = requests.Session()
        try:
            response = self.circle_session.get(url, timeout=30)
            if response.status_code == 404:
                return None
            if response.status_code != 200:
                raise AttestationError(f"Circle CCTP API returned HTTP {response.status_code}")
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            raise AttestationError("Circle CCTP request failed") from exc

    def quote(self, plan: Plan) -> EvmCctpQuote:
        """Read current CCTP fees and fix the maximum USDC deduction without signing."""
        m = self._metadata(plan)
        options = normalize_cctp(plan.cctp)
        finality = 1000 if options.speed == "fast" else 2000
        suffix = "?forward=true" if options.forwarding else ""
        response = self._json(f"{m.attestation_url}/v2/burn/USDC/fees/{m.source_domain}/{m.destination_domain}{suffix}")
        if not isinstance(response, list):
            raise AttestationError("Circle returned invalid CCTP fees")
        entries = [cast(dict[str, Any], f) for f in cast(list[Any], response) if isinstance(f, dict)]
        matches = [f for f in entries if type(f.get("finalityThreshold")) is int
                   and f["finalityThreshold"] == finality]
        if len(matches) != 1:
            raise AttestationError("Circle does not uniquely quote the requested CCTP finality")
        fee = matches[0]
        protocol = _basis_points(plan.amount_atomic, fee.get("minimumFee"))
        forwarding = 0
        if options.forwarding:
            forward = fee.get("forwardFee")
            if not isinstance(forward, dict):
                raise AttestationError("Invalid Circle forwarding fee")
            forward = cast(dict[str, Any], forward)
            forwarding = _uint(forward.get("medium", forward.get("med")), "forwarding fee")
        required = protocol + forwarding
        cap = required if options.max_fee is None else parse_decimal_amount(options.max_fee, 6)
        if required > cap:
            raise InvalidAmountError("Live CCTP fees exceed the approved max_fee; request a new quote")
        if cap >= plan.amount_atomic:
            raise InvalidAmountError("CCTP fees must be less than the burn amount")
        receive = plan.amount_atomic - (cap if options.forwarding else required)
        plan = replace(plan, cctp=CctpOptions(options.speed, options.forwarding, format_decimal_amount(cap, 6)))
        fees = [Fee("protocol", m.destination_chain, plan.destination_asset_id, format_decimal_amount(protocol, 6), True)]
        if options.forwarding:
            fees.append(Fee("relayer", m.destination_chain, plan.destination_asset_id,
                            format_decimal_amount(forwarding, 6), True))
        return EvmCctpQuote("evm-cctp", plan, tuple(fees), format_decimal_amount(receive, 6), plan.amount_atomic,
                           receive, protocol, forwarding, cap, finality, options.forwarding)

    def _connection(self, chain: str, expected: int) -> Ethereum:
        conn: Ethereum = self.bridge.evm(chain).conn
        if conn.chain_id != expected:
            raise ConfigurationError(f"CCTP transport must use EVM chain {expected}")
        return conn

    @staticmethod
    def _contract(conn: Ethereum, address: str, abi: list[dict[str, Any]]) -> Any:
        from web3 import Web3
        return conn.w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)

    @staticmethod
    def _hash(value: Any) -> str:
        if not isinstance(value, str) or not re.fullmatch(r"0x[0-9a-fA-F]{64}", value):
            raise ConfigurationError("Invalid CCTP transaction hash")
        return value

    @staticmethod
    def _hex(value: Any) -> str:
        from web3 import Web3
        return Web3.to_hex(value) if not isinstance(value, str) else value

    def _success(self, mined: Any, tx_hash: str) -> None:
        if self._hex(mined["transactionHash"]).lower() != tx_hash.lower() or mined["status"] != 1:
            raise BridgeError("CCTP transaction reverted or receipt hash differs")

    def _confirm(self, conn: Ethereum, tx_hash: str, interval: float, timeout: float) -> Any:
        deadline = time.monotonic() + timeout
        while True:
            mined = conn.get_receipt(tx_hash)
            if mined is not None:
                self._success(mined, tx_hash)
                return mined
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(max(interval, 0.01), remaining))

    def _broadcast(self, conn: Ethereum, tx: dict[str, Any], receipt: Receipt,
                   emit: Callable[[Receipt], Any], *, plan: Plan, approval: bool = False, destination: bool = False) -> Receipt:
        def submitted(tx_hash: str) -> Receipt:
            self._hash(tx_hash)
            state = dict(receipt.protocol_state)
            if approval:
                state["approvalTxIds"] = [*state.get("approvalTxIds", []), tx_hash]
                return receipt.replace(id=tx_hash, status=Status.SOURCE_APPROVAL_PENDING, protocol_state=state)
            if destination:
                return receipt.replace(destination_tx_id=tx_hash, status=Status.DESTINATION_CONFIRMING, next_action=None)
            if conn.last_broadcast_nonce is not None:
                state["sourceNonce"] = str(conn.last_broadcast_nonce)
            return receipt.replace(id=tx_hash, source_tx_id=tx_hash, status=Status.SOURCE_CONFIRMING, protocol_state=state)
        def persist(state: Receipt, tx_hash: str) -> None:
            from .checkpoint import create_checkpoint
            checkpoint = create_checkpoint(plan, state, self.bridge.registry)
            try:
                # Save the transaction independently of user notification. A callback
                # failure must never leave only an earlier approval in the journal.
                store = getattr(self.bridge, "checkpoints", None)
                if store is not None:
                    store.save(checkpoint)
                emit(state)
            except Exception as exc:
                raise CctpCheckpointError(tx_hash, checkpoint) from exc
        try:
            tx_hash = conn.send_transaction(tx)
        except BridgeError as exc:
            if getattr(exc, "broadcast_id", None):
                tx_hash = self._hash(getattr(exc, "broadcast_id"))
                persist(submitted(tx_hash), tx_hash)
            raise
        state = submitted(tx_hash)
        persist(state, tx_hash)
        return state

    def _approvals_confirmed(self, conn: Ethereum, m: Metadata, plan: Plan, receipt: Receipt) -> bool:
        from web3.exceptions import TransactionNotFound
        from eth_abi.abi import decode
        from eth_utils.crypto import keccak
        values = receipt.protocol_state.get("approvalTxIds")
        if not isinstance(values, list):
            raise ConfigurationError("Invalid CCTP approval checkpoint")
        sender = plan.sender or receipt.protocol_state.get("sourceSender")
        if not isinstance(sender, str):
            raise ConfigurationError("CCTP recovery requires the source sender")
        address_bytes(sender)
        for value in cast(list[Any], values):
            tx_hash = self._hash(value)
            try:
                tx = conn.w3.eth.get_transaction(tx_hash)
            except TransactionNotFound:
                return False
            mined = conn.get_receipt(tx_hash)
            if mined is None:
                return False
            self._success(mined, tx_hash)
            if (self._hex(tx["hash"]).lower() != tx_hash.lower() or tx["from"].lower() != sender.lower()
                    or (tx.get("to") or "").lower() != m.source_token.lower() or tx.get("value", 0) != 0):
                raise ConfigurationError("CCTP approval does not match the source account and token")
            raw = bytes(tx["input"])
            if len(raw) != 68 or raw[:4] != keccak(text="approve(address,uint256)")[:4]:
                raise ConfigurationError("CCTP approval calldata is invalid")
            spender, amount = decode(["address", "uint256"], raw[4:])
            if spender.lower() != m.messenger.lower() or amount != plan.amount_atomic:
                raise ConfigurationError("CCTP approval does not match the planned allowance")
        return True

    def execute(self, plan: Plan, *, on_checkpoint: Callable[[Receipt], Any], poll_seconds: float = 1.0,
                timeout_seconds: float = 120.0, resume: Receipt | None = None) -> Receipt:
        from web3 import Web3
        m = self._metadata(plan)
        if resume is not None and resume.source_tx_id:
            return self.get_status(plan, resume)
        conn = self._connection(m.source_chain, m.source_chain_id)
        sender = conn.require_address()
        if plan.sender and sender.lower() != plan.sender.lower():
            raise ConfigurationError("CCTP source wallet differs from the planned sender")
        if resume and sender.lower() != str(resume.protocol_state.get("sourceSender")).lower():
            raise ConfigurationError("CCTP resumed wallet differs from checkpoint sender")
        priced = self.quote(replace(plan, sender=sender))
        state = resume or Receipt(plan.route_id, "cctp", Status.PREPARED,
                  protocol_state={"routeId": plan.route_id, "sourceSender": sender, "approvalTxIds": []})
        if resume and not self._approvals_confirmed(conn, m, plan, state):
            return state.replace(status=Status.SOURCE_APPROVAL_PENDING)
        if resume:
            reconciled = self.get_status(plan, state)
            if reconciled.status != Status.SOURCE_SUBMISSION_PENDING:
                return reconciled
        token = self._contract(conn, m.source_token, TOKEN_ABI)
        if token.functions.balanceOf(sender).call() < plan.amount_atomic:
            raise BridgeError("Insufficient source USDC balance for CCTP burn")
        if conn.w3.eth.get_balance(sender) <= 0:
            raise BridgeError("Source wallet requires native gas funds for CCTP approval and burn")
        if token.functions.allowance(sender, Web3.to_checksum_address(m.messenger)).call() < plan.amount_atomic:
            data = token.encode_abi("approve", args=[Web3.to_checksum_address(m.messenger), plan.amount_atomic])
            state = self._broadcast(conn, {"to": token.address, "data": data}, state, on_checkpoint, plan=priced.plan, approval=True)
            if not self._confirm(conn, state.id, poll_seconds, timeout_seconds):
                return state
        priced = self.quote(priced.plan)
        args: list[Any] = [plan.amount_atomic, m.destination_domain, address_bytes(plan.recipient),
                Web3.to_checksum_address(m.source_token), bytes(32), priced.max_fee_atomic, priced.min_finality_threshold]
        function = "depositForBurn"
        if priced.forwarding:
            args.append(FORWARD_HOOK)
            function += "WithHook"
        messenger = self._contract(conn, m.messenger, MESSENGER_ABI)
        state = self._broadcast(conn, {"to": messenger.address, "data": messenger.encode_abi(function, args=args)},
                                state, on_checkpoint, plan=priced.plan)
        if not self._confirm(conn, self._hash(state.source_tx_id), poll_seconds, timeout_seconds):
            return state
        return self.get_status(priced.plan, state)

    def _check_message(self, raw: bytes, m: Metadata, plan: Plan, sender: str) -> CctpMessage:
        options = normalize_cctp(plan.cctp)
        if options.max_fee is None:
            raise ConfigurationError("CCTP verification requires the approved fee cap")
        return validate_message(raw, source_domain=m.source_domain, destination_domain=m.destination_domain,
               messenger=m.messenger, source_token=m.source_token, sender=sender, recipient=plan.recipient,
               amount_atomic=plan.amount_atomic, max_fee_atomic=parse_decimal_amount(options.max_fee, 6),
               finality=1000 if options.speed == "fast" else 2000, forwarding=options.forwarding)

    def _events(self, conn: Ethereum, address: str, abi: list[dict[str, Any]], name: str,
                logs: Iterable[Any]) -> Iterator[tuple[dict[str, Any], Any]]:
        event = getattr(self._contract(conn, address, abi).events, name)()
        for log in logs:
            if log["address"].lower() != address.lower():
                continue
            try:
                decoded = event.process_log(log)
            except Exception:
                # Unrelated logs have diverse Web3/eth-abi decoding errors.
                # This boundary includes no RPC or other network operations.
                continue
            yield decoded["args"], log

    def _destination_matches(self, conn: Ethereum, mined: Any, m: Metadata,
                             message: CctpMessage, recipient: str) -> bool:
        received = any(e["sourceDomain"] == m.source_domain and bytes(e["nonce"]) == message.nonce
                    and bytes(e["sender"]) == message.messenger and bytes(e["messageBody"]) == message.raw[148:]
                    and e["finalityThresholdExecuted"] == message.finality
                    for e, _ in self._events(conn, m.transmitter, TRANSMITTER_ABI, "MessageReceived", mined["logs"]))
        minted = any(int(e["from"], 16) == 0 and e["to"].lower() == recipient.lower()
                    and e["value"] == message.amount - message.fee
                    for e, _ in self._events(conn, m.destination_token, TOKEN_ABI, "Transfer", mined["logs"]))
        return received and minted

    def get_status(self, plan: Plan, receipt: Receipt) -> Receipt:
        from web3 import Web3
        from eth_utils.crypto import keccak
        m = self._metadata(plan)
        if receipt.protocol != "cctp" or receipt.protocol_state.get("routeId") != plan.route_id:
            raise ConfigurationError("CCTP receipt belongs to a different route")
        if receipt.status in TERMINAL:
            return receipt
        source = self._connection(m.source_chain, m.source_chain_id)
        if not receipt.source_tx_id:
            confirmed = self._approvals_confirmed(source, m, plan, receipt)
            if not confirmed:
                return receipt.replace(status=Status.SOURCE_APPROVAL_PENDING)
            return self._reconcile_approval(source, m, plan, receipt)
        tx_hash = self._hash(receipt.source_tx_id)
        mined = source.get_receipt(tx_hash)
        if mined is None:
            return receipt.replace(status=Status.SOURCE_CONFIRMING)
        if mined["status"] == 0:
            return receipt.replace(status=Status.FAILED, protocol_state={**receipt.protocol_state, "error": "CCTP source burn reverted"})
        self._success(mined, tx_hash)
        sender = plan.sender or receipt.protocol_state.get("sourceSender")
        if not isinstance(sender, str):
            raise ConfigurationError("CCTP verification requires the source sender")
        address_bytes(sender)
        messages: list[CctpMessage] = []
        for event, _ in self._events(source, m.transmitter, TRANSMITTER_ABI, "MessageSent", mined["logs"]):
            try:
                messages.append(self._check_message(bytes(event["message"]), m, plan, sender))
            except AttestationError:
                continue
        if len(messages) != 1:
            raise AttestationError("Source receipt must contain exactly one CCTP message matching the intent")
        response = self._json(f"{m.attestation_url}/v2/messages/{m.source_domain}?transactionHash={tx_hash}")
        response = cast(dict[str, Any], response) if isinstance(response, dict) else {}
        if "sourceTxHash" in response and str(response["sourceTxHash"]).lower() != tx_hash.lower():
            raise AttestationError("Circle returned messages for a different source transaction")
        candidates: list[tuple[dict[str, Any], bytes]] = []
        entries: Any = response.get("messages", [])
        for entry in cast(list[Any], entries) if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            entry = cast(dict[str, Any], entry)
            try:
                raw = bytes.fromhex(entry["message"][2:])
                if not entry["message"].startswith("0x") or immutable_message(raw) != immutable_message(messages[0].raw):
                    continue
            except (ValueError, KeyError, TypeError, AttestationError):
                continue
            candidates.append((entry, raw))
        if len(candidates) > 1:
            raise AttestationError("Circle returned ambiguous CCTP messages")
        if not candidates:
            return receipt.replace(status=Status.ATTESTATION_PENDING)
        entry, raw = candidates[0]
        if entry.get("status") != "complete" or not isinstance(entry.get("attestation"), str) or not re.fullmatch(r"0x(?:[0-9a-fA-F]{2})+", entry["attestation"]):
            return receipt.replace(status=Status.ATTESTATION_PENDING)
        message = self._check_message(raw, m, plan, sender)
        if message.nonce == bytes(32) or message.finality < message.min_finality or message.fee > message.max_fee:
            raise AttestationError("Invalid Circle CCTP attested nonce, finality, or fee")
        dest = self._connection(m.destination_chain, m.destination_chain_id)
        used = self._contract(dest, m.transmitter, TRANSMITTER_ABI).functions.usedNonces(message.nonce).call() != 0
        state = {**receipt.protocol_state, "message": "0x"+raw.hex(), "attestation": entry["attestation"],
                 "nonce": "0x"+message.nonce.hex(), "nonceUsed": used, "sourceSender": sender}
        receipt = receipt.replace(protocol_state=state)
        forwarded = entry.get("forwardTxHash")
        dest_hash = receipt.destination_tx_id or (forwarded if isinstance(forwarded, str) and re.fullmatch(r"0x[0-9a-fA-F]{64}", forwarded) else None)
        if used and not dest_hash:
            head = dest.w3.eth.block_number
            floor = max(0, head-9999)
            end = head
            topics = ["0x"+keccak(text="MessageReceived(address,uint32,bytes32,bytes32,uint32,bytes)").hex(), None, "0x"+message.nonce.hex()]
            while end >= floor:
                start = max(floor, end-999)
                logs = dest.w3.eth.get_logs({"address": Web3.to_checksum_address(m.transmitter), "topics": topics,
                                              "fromBlock": start, "toBlock": end})
                for event, log in self._events(dest, m.transmitter, TRANSMITTER_ABI, "MessageReceived", logs):
                    if bytes(event["nonce"]) == message.nonce and event["sourceDomain"] == m.source_domain:
                        dest_hash = self._hash(self._hex(log["transactionHash"]))
                        break
                if dest_hash:
                    break
                end = start-1
        failed = False
        if dest_hash:
            delivered = dest.get_receipt(self._hash(dest_hash))
            if delivered is None:
                return receipt.replace(status=Status.DESTINATION_CONFIRMING, destination_tx_id=dest_hash, next_action=None)
            if delivered["status"] == 1:
                self._success(delivered, dest_hash)
                if not self._destination_matches(dest, delivered, m, message, plan.recipient):
                    raise AttestationError("CCTP destination receipt does not prove the expected USDC mint")
                return receipt.replace(status=Status.COMPLETED if used else Status.DESTINATION_CONFIRMING,
                                       destination_tx_id=dest_hash, next_action=None)
            failed = True
            receipt = receipt.replace(destination_tx_id=None, protocol_state={**state, "forwardingFailed": True})
        if used or (normalize_cctp(plan.cctp).forwarding and not failed):
            return receipt.replace(status=Status.DELIVERY_PENDING, next_action=None)
        return receipt.replace(status=Status.DESTINATION_ACTION_REQUIRED,
                               next_action={"kind": "cctp-mint", "chainId": m.destination_chain})

    def _reconcile_approval(self, conn: Ethereum, m: Metadata, plan: Plan, receipt: Receipt) -> Receipt:
        """A stale approval is not authority to repeat a later burn. Reconcile mined history
        and refuse an unaccounted-for source nonce before offering a resumption."""
        from eth_utils.crypto import keccak
        from web3 import Web3
        approvals = receipt.protocol_state["approvalTxIds"]
        if not approvals:
            raise ConfigurationError("CCTP recovery needs a submitted approval or burn")
        mined = [conn.get_receipt(h) for h in approvals]
        start = min(int(r["blockNumber"]) for r in mined if r is not None)
        head = conn.w3.eth.block_number
        if head < start:
            raise ConfigurationError("CCTP RPC head precedes the confirmed approval; retry recovery on a consistent provider")
        head_hash = self._hex(conn.w3.eth.get_block(head)["hash"])
        sender = plan.sender or str(receipt.protocol_state["sourceSender"])
        last_nonce = max(int(conn.w3.eth.get_transaction(h)["nonce"]) for h in approvals)
        found: set[str] = set()
        while start <= head:
            end = min(head, start+999)
            logs = conn.w3.eth.get_logs({"address": Web3.to_checksum_address(m.transmitter),
                    "topics": ["0x"+keccak(text="MessageSent(bytes)").hex()], "fromBlock": start, "toBlock": end})
            for event, log in self._events(conn,m.transmitter,TRANSMITTER_ABI,"MessageSent",logs):
                try:
                    self._check_message(bytes(event["message"]),m,plan,sender)
                except AttestationError:
                    continue
                candidate = self._hash(self._hex(log["transactionHash"]))
                transaction = conn.w3.eth.get_transaction(candidate)
                if (self._hex(transaction["hash"]).lower() != candidate.lower()
                        or transaction["from"].lower() != sender.lower()
                        or int(transaction["nonce"]) <= last_nonce):
                    continue
                found.add(candidate)
            start = end+1
        if len(found) > 1:
            raise ConfigurationError("Multiple CCTP burns match this approval; recover the intended source transaction explicitly")
        confirmed_nonce = conn.w3.eth.get_transaction_count(sender, head)
        pending_nonce = conn.w3.eth.get_transaction_count(sender, "pending")
        if self._hex(conn.w3.eth.get_block(head)["hash"]) != head_hash:
            raise ConfigurationError("CCTP head block changed during recovery; retry on a consistent provider")
        if found:
            tx_hash = next(iter(found))
            return self.get_status(plan,receipt.replace(id=tx_hash,source_tx_id=tx_hash,status=Status.SOURCE_CONFIRMING))
        # The scan covers confirmed activity through this exact head, including
        # unrelated transfers after the approval. Only later activity is unresolved.
        if confirmed_nonce < last_nonce+1:
            raise ConfigurationError("CCTP RPC nonce precedes the confirmed approval; retry on a consistent provider")
        if pending_nonce > confirmed_nonce:
            return receipt.replace(status=Status.SOURCE_APPROVAL_PENDING,
                  protocol_state={**receipt.protocol_state,"sourceError":"Later source activity is unresolved; recover its burn hash before resuming"})
        return receipt.replace(status=Status.SOURCE_SUBMISSION_PENDING)

    def recover(self, plan: Plan, checkpoint: Checkpoint, *, approval_replacement: dict[str, str] | None = None) -> Receipt:
        from web3.exceptions import TransactionNotFound
        m = self._metadata(plan)
        receipt = self.from_checkpoint(plan, checkpoint)
        if approval_replacement is not None:
            if receipt.source_tx_id:
                raise ConfigurationError("Cannot replace approvals after a CCTP burn was submitted")
            if not isinstance(cast(object, approval_replacement), dict) or set(approval_replacement) != {"original_transaction_id", "replacement_transaction_id"}:
                raise ConfigurationError("Invalid approval replacement selection")
            original = self._hash(approval_replacement["original_transaction_id"])
            replacement = self._hash(approval_replacement["replacement_transaction_id"])
            active = list(receipt.protocol_state["approvalTxIds"])
            normalized = [v.lower() for v in active]
            if original.lower() not in normalized or replacement.lower() in normalized:
                raise ConfigurationError("Replacement must identify a saved approval and a different transaction")
            conn = self._connection(m.source_chain, m.source_chain_id)
            try:
                original_tx = conn.w3.eth.get_transaction(original)
            except TransactionNotFound:
                original_tx = None
            if original_tx is not None or conn.get_receipt(original) is not None:
                raise ConfigurationError("Original approval is still visible; reconcile it first")
            test = receipt.replace(protocol_state={**receipt.protocol_state, "approvalTxIds": [replacement]})
            if not self._approvals_confirmed(conn, m, plan, test):
                raise ConfigurationError("Replacement approval must be confirmed before recovery")
            active[normalized.index(original.lower())] = replacement
            receipt = receipt.replace(id=active[-1], protocol_state={**receipt.protocol_state, "approvalTxIds": active,
                     "replacedApprovalTxIds": [*receipt.protocol_state.get("replacedApprovalTxIds", []), original]})
        return self.get_status(plan, receipt)

    @classmethod
    def from_checkpoint(cls, plan: Plan, checkpoint: Checkpoint) -> Receipt:
        source, destination = checkpoint.source or {}, checkpoint.destination or {}
        if source.get("preparedTransaction") is not None or destination.get("preparedTransaction") is not None:
            raise ConfigurationError("CCTP checkpoints cannot contain prepared Aleo transactions")
        active = source.get("approvalTransactionIds", [])
        replaced = source.get("replacedApprovalTransactionIds", [])
        for values in (active, replaced):
            if not isinstance(values, list):
                raise ConfigurationError("Invalid CCTP approval checkpoint")
            for value in cast(list[Any], values):
                cls._hash(value)
        tx_hash, dest_hash = source.get("transactionId"), destination.get("transactionId")
        if tx_hash is not None:
            cls._hash(tx_hash)
        if dest_hash is not None:
            cls._hash(dest_hash)
        if not tx_hash and (not active or dest_hash):
            raise ConfigurationError("CCTP checkpoint contains no submitted source transaction")
        options = normalize_cctp(plan.cctp)
        if options.max_fee is None:
            raise ConfigurationError("CCTP checkpoint must preserve its approved fee cap")
        if plan.sender is None:
            raise ConfigurationError("CCTP checkpoint requires the source sender")
        address_bytes(plan.sender)
        return Receipt(tx_hash or active[-1], "cctp", Status.SOURCE_CONFIRMING if tx_hash else Status.SOURCE_APPROVAL_PENDING,
                       source_tx_id=tx_hash, destination_tx_id=dest_hash,
                       protocol_state={"routeId": plan.route_id, "sourceSender": plan.sender,
                                       "approvalTxIds": list(active), "replacedApprovalTxIds": list(replaced)})

    def complete(self, plan: Plan, receipt: Receipt, *, manual_mint: bool = False,
                 on_checkpoint: Callable[[Receipt], Any]) -> Receipt:
        if type(manual_mint) is not bool:
            raise ConfigurationError("manual_mint must be boolean")
        m = self._metadata(plan)
        state = self.get_status(plan, receipt)
        if state.status in (Status.COMPLETED, Status.DESTINATION_CONFIRMING):
            return state
        fallback = manual_mint and state.status == Status.DELIVERY_PENDING and state.protocol_state.get("nonceUsed") is False
        if state.status != Status.DESTINATION_ACTION_REQUIRED and not fallback:
            raise ConfigurationError("CCTP transfer is not ready for manual destination minting")
        conn = self._connection(m.destination_chain, m.destination_chain_id)
        sender = conn.require_address()
        if conn.w3.eth.get_balance(sender) <= 0:
            raise BridgeError("Destination wallet requires native gas funds for CCTP mint")
        transmitter = self._contract(conn, m.transmitter, TRANSMITTER_ABI)
        args = [bytes.fromhex(state.protocol_state[k][2:]) for k in ("message", "attestation")]
        return self._broadcast(conn, {"to": transmitter.address, "data": transmitter.encode_abi("receiveMessage", args=args)},
                               state, on_checkpoint, plan=plan, destination=True)
