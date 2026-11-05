"""Move public ALEO between Aleo and BSC with an explicit destination claim.

Quote and execute through ``Bridge``. Save checkpoints and use ``wait``,
``recover``, ``resume``, and ``complete`` for an existing transfer.
"""
from __future__ import annotations

from dataclasses import replace
import time
from typing import Any, Callable

import requests

from ._verulink_context import Context
from ._verulink_wire import U64_MAX, ZERO_ALEO, address_literal, incoming, integer, same
from ._verulink_source import (inputs, prepared_transaction, tx_hash, validate_aleo_claim, validate_aleo_source)
from ._verulink_attestation import verified_signatures
from . import _verulink_status as status
from .checkpoint import Checkpoint, create_checkpoint
from .errors import BridgeError
from .types import Fee, Plan, Receipt, Status, VerulinkQuote
from .units import format_decimal_amount, parse_decimal_amount


class VerulinkCheckpointError(BridgeError):
    """A submission needs recovery from the attached checkpoint after persistence failed."""

    def __init__(self, tx_id: str, checkpoint: Checkpoint) -> None:
        super().__init__(f"Verulink transaction {tx_id} may be submitted; recover the attached checkpoint, never execute again")
        self.broadcast_id = tx_id
        self.checkpoint = checkpoint


class VerulinkModule:
    """Read public ALEO bridge fees and verify Verulink delivery."""

    def __init__(self, bridge: Any) -> None:
        self.bridge = bridge
        self.session: Any = requests.Session()

    def context(self, plan: Plan) -> Context:
        return Context(self.bridge, plan)

    def quote(self, plan: Plan) -> VerulinkQuote:
        """Read current fees and fix the minimum public ALEO output without signing."""
        c = self.context(plan)
        c.assert_active()
        rate, minimum, maximum = c.configuration()
        amount = plan.amount_atomic
        if c.outbound and amount * rate > U64_MAX:
            raise BridgeError("Verulink Aleo fee multiplication exceeds u64")
        fee = amount * rate // 100_000
        net = amount - fee
        if net <= 0 or net < minimum:
            raise BridgeError(f"Verulink net amount is below minimum {format_decimal_amount(minimum, 6)} ALEO")
        if net > maximum:
            raise BridgeError(f"Verulink net amount exceeds maximum {format_decimal_amount(maximum, 6)} ALEO")
        output = format_decimal_amount(net, 6)
        fees = (Fee("protocol", c.source.chain_id, c.source.id, format_decimal_amount(fee, 6), True),
                Fee("network", c.source.chain_id, None, None, True),
                Fee("network", c.destination.chain_id, None, None, True))
        return VerulinkQuote("aleo-verulink" if c.outbound else "evm-verulink", replace(plan, amount_out=output),
                             fees, output, amount, fee, rate)

    def _save(self, plan: Plan, receipt: Receipt, emit: Callable[[Receipt], Any], *, submitted: bool = False) -> None:
        checkpoint = create_checkpoint(plan, receipt, self.bridge.registry)
        try:
            # Persist independently of user notification; a failed callback must not lose a hash.
            if self.bridge.checkpoints is not None:
                self.bridge.checkpoints.save(checkpoint)
            emit(receipt)
        except Exception as exc:
            if submitted:
                raise VerulinkCheckpointError(receipt.destination_tx_id or receipt.id, checkpoint) from exc
            raise

    def _send(self, c: Context, plan: Plan, receipt: Receipt, to: str, name: str, args: list[Any],
              emit: Callable[[Receipt], Any], *, approval: bool = False, destination: bool = False) -> Receipt:
        sender = c.evm.require_address()
        data = c.contract(to).encode_abi(name, args=args)
        tx = {"from": sender, "to": to, "chainId": c.evm_chain_id, "data": data, "value": 0}
        c.evm.w3.eth.call({k: v for k, v in tx.items() if k != "chainId"})

        def submitted(tx_id: str) -> Receipt:
            tx_hash(tx_id)
            if approval:
                return status.update(receipt.replace(id=tx_id), Status.SOURCE_APPROVAL_PENDING,
                                     approvalTxIds=[*receipt.protocol_state.get("approvalTxIds", []), tx_id])
            if destination:
                return status.update(receipt.replace(destination_tx_id=tx_id), Status.DESTINATION_CONFIRMING)
            return status.update(receipt.replace(id=tx_id, source_tx_id=tx_id), Status.SOURCE_CONFIRMING)

        try:
            tx_id = c.evm.send_transaction(tx)
        except BridgeError as exc:
            if getattr(exc, "broadcast_id", None):
                self._save(plan, submitted(getattr(exc, "broadcast_id")), emit, submitted=True)
            raise
        result = submitted(tx_id)
        self._save(plan, result, emit, submitted=True)
        return result

    def _prepare(self, c: Context, function: str, args: list[str], proving: str) -> Any:
        from .lifecycle import _prepare_aleo
        call = self.bridge._call(c.aleo_service, function, args, lambda tx_id, outputs: tx_id)
        return _prepare_aleo(call, proving)

    def execute(self, plan: Plan, *, on_checkpoint: Callable[[Receipt], Any], proving: str = "delegate",
                mode: str | None = None, poll_seconds: float = 1.0, timeout_seconds: float = 120.0) -> Receipt:
        """Submit public source funds after checking fresh fees and peer configuration."""
        from .lifecycle import _connected_aleo_address, submit_serialized
        if mode not in (None, "public", "public-as-signer", "signer"):
            raise BridgeError("Verulink only supports public source balances")
        c = self.context(plan)

        def price() -> VerulinkQuote:
            q = self.quote(plan)
            assert q.amount_out is not None
            if plan.amount_out is not None and parse_decimal_amount(q.amount_out, 6) < parse_decimal_amount(plan.amount_out, 6):
                raise BridgeError("Verulink fee increased since quoting; quote again before submission")
            return q

        priced = price()
        sender = _connected_aleo_address(self.bridge) if c.outbound else c.evm.require_address()
        if not sender or (plan.sender is not None and not same(sender, plan.sender)):
            raise BridgeError("Verulink source sender does not match the wallet account")
        state = {"routeId": plan.route_id, "sourceSender": sender}
        if c.outbound:
            if integer(c.mapping("credits.aleo", "account", sender), "public balance") < plan.amount_atomic:
                raise BridgeError("Insufficient public ALEO balance for Verulink")
            args = [address_literal(plan.recipient), f"{plan.amount_atomic}u64", c.evm_domain + "u128",
                    address_literal(c.evm_service), address_literal(c.evm_token), f"{priced.fee_atomic}u64"]
            prepared = self._prepare(c, "token_send_public", args, proving)
            tx = prepared_transaction(prepared.serialized, prepared.transaction_id)
            validate_aleo_source(c, replace(plan, sender=sender), tx)
            pending = Receipt(prepared.transaction_id, "verulink", Status.SOURCE_SUBMISSION_PENDING,
                              protocol_state={**state, "preparedTransaction": prepared.serialized})
            self._save(plan, pending, on_checkpoint)
            submit_serialized(self.bridge, prepared.serialized, prepared.transaction_id)
            result = Receipt(prepared.transaction_id, "verulink", Status.SOURCE_CONFIRMING,
                             source_tx_id=prepared.transaction_id, protocol_state=state)
            self._save(plan, result, on_checkpoint, submitted=True)
            return result
        if integer(c.read(c.evm_token, "balanceOf", [sender]), "wrapped balance", 2**256 - 1) < plan.amount_atomic:
            raise BridgeError("Insufficient wrapped ALEO balance")
        if c.evm.w3.eth.get_balance(sender) <= 0:
            raise BridgeError("BNB is required to pay Verulink source gas")
        pending = Receipt("unsubmitted", "verulink", Status.PREPARED, protocol_state=state)
        if integer(c.read(c.evm_token, "allowance", [sender, c.evm_service]), "allowance", 2**256 - 1) < plan.amount_atomic:
            pending = self._send(c, plan, pending, c.evm_token, "approve", [c.evm_service, plan.amount_atomic], on_checkpoint, approval=True)
            deadline = time.monotonic() + timeout_seconds
            while True:
                result = status.approval_status(c, replace(plan, sender=sender), pending)
                if result.status == Status.FAILED:
                    return result
                if result.status == Status.SOURCE_SUBMISSION_PENDING:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return pending
                time.sleep(min(max(0.01, poll_seconds), remaining))
            if integer(c.read(c.evm_token, "allowance", [sender, c.evm_service]), "allowance", 2**256 - 1) < plan.amount_atomic:
                raise BridgeError("Confirmed Verulink approval did not grant the required allowance")
        price()  # Approval confirmation may take minutes; re-read before the burn.
        return self._send(c, plan, pending, c.evm_service, "publicTokenSend", [c.evm_token, plan.amount_atomic, plan.recipient], on_checkpoint)

    def get_status(self, plan: Plan, receipt: Receipt) -> Receipt:
        """Verify source, attestors and destination delivery without submitting."""
        return status.get_status(self.context(plan), plan, receipt, self.session)

    @staticmethod
    def from_checkpoint(plan: Plan, checkpoint: Checkpoint) -> Receipt:
        source, dest = checkpoint.source or {}, checkpoint.destination or {}
        outbound = plan.source_asset_id.split("/")[0] == "aleo"
        prepared, claim = source.get("preparedTransaction"), dest.get("preparedTransaction")
        approvals = source.get("approvalTransactionIds", [])
        if not isinstance(approvals, list):
            raise BridgeError("Invalid Verulink approval checkpoint")
        for value in approvals:
            tx_hash(value)
        tx_id, dest_id = source.get("transactionId"), dest.get("transactionId")
        if prepared is not None and (not outbound or tx_id or dest or approvals):
            raise BridgeError("Invalid Verulink prepared source checkpoint")
        if claim is not None and (outbound or dest_id):
            raise BridgeError("Invalid Verulink prepared destination checkpoint")
        state: dict[str, Any] = {"routeId": plan.route_id, "sourceSender": plan.sender, "approvalTxIds": approvals}
        if prepared is not None:
            if not isinstance(prepared, dict):
                raise BridgeError("Invalid Verulink prepared source checkpoint")
            tx = prepared_transaction(prepared.get("serializedTransaction"), prepared.get("transactionId"))
            return Receipt(tx["id"], "verulink", Status.SOURCE_SUBMISSION_PENDING,
                           protocol_state={**state, "preparedTransaction": prepared["serializedTransaction"]})
        if not tx_id and (not approvals or dest) or outbound and approvals:
            raise BridgeError("Invalid Verulink checkpoint source identifiers")
        if tx_id is not None and (not isinstance(tx_id, str) or not tx_id):
            raise BridgeError("Invalid Verulink source transaction id")
        if tx_id and not outbound:
            tx_hash(tx_id)
        if dest_id is not None and (not isinstance(dest_id, str) or not dest_id):
            raise BridgeError("Invalid Verulink claim transaction id")
        if dest_id and outbound:
            tx_hash(dest_id)
        receipt_id = tx_id or approvals[-1]
        if claim is not None:
            if not isinstance(claim, dict):
                raise BridgeError("Invalid Verulink prepared destination checkpoint")
            tx = prepared_transaction(claim.get("serializedTransaction"), claim.get("transactionId"))
            receipt_id = tx["id"]
            state["preparedDestinationTransaction"] = claim["serializedTransaction"]
        return Receipt(receipt_id, "verulink", Status.DESTINATION_CONFIRMING if dest_id else
                       Status.SOURCE_CONFIRMING if tx_id else Status.SOURCE_APPROVAL_PENDING,
                       source_tx_id=tx_id, destination_tx_id=dest_id, protocol_state=state)

    def recover(self, plan: Plan, checkpoint: Checkpoint) -> Receipt:
        """Reconstruct a saved transfer and read its current chain evidence."""
        c = self.context(plan)
        r = self.from_checkpoint(plan, checkpoint)
        if "preparedTransaction" in r.protocol_state:
            validate_aleo_source(c, plan, prepared_transaction(r.protocol_state["preparedTransaction"], r.id))
            return r
        return self.get_status(plan, r)

    def resume(self, plan: Plan, receipt: Receipt, *, on_checkpoint: Callable[[Receipt], Any]) -> Receipt:
        """Rebroadcast the saved Aleo bytes; an ambiguous EVM source needs its wallet hash."""
        from .lifecycle import submit_serialized
        c = self.context(plan)
        if not c.outbound:
            raise BridgeError("Verulink approval-only recovery cannot safely resubmit; recover with the source transaction hash from the wallet")
        c.assert_active()
        c.configuration()
        serialized = receipt.protocol_state.get("preparedTransaction")
        if not isinstance(serialized, str):
            raise BridgeError("Invalid Verulink prepared source transaction")
        tx = prepared_transaction(serialized, receipt.id)
        validate_aleo_source(c, plan, tx)
        self._save(plan, receipt, on_checkpoint)
        submit_serialized(self.bridge, serialized, receipt.id)
        state = dict(receipt.protocol_state)
        state.pop("preparedTransaction", None)
        result = receipt.replace(source_tx_id=receipt.id, status=Status.SOURCE_CONFIRMING, next_action=None, protocol_state=state)
        self._save(plan, result, on_checkpoint, submitted=True)
        return result

    def complete(self, plan: Plan, receipt: Receipt, *, on_checkpoint: Callable[[Receipt], Any], proving: str = "delegate") -> Receipt:
        """Claim a positively attested packet using fresh replay and membership checks."""
        from .lifecycle import submit_serialized
        c = self.context(plan)
        if receipt.destination_tx_id:
            raise BridgeError("Verulink claim already submitted; recover or wait instead of submitting again")
        if receipt.status != Status.DESTINATION_ACTION_REQUIRED or (receipt.next_action or {}).get("kind") != "verulink-claim":
            raise BridgeError("Verulink receipt has no destination claim ready")
        c.assert_active(destination_only=True)
        c.configuration()
        ready = self.get_status(plan, receipt)
        if ready.status != Status.DESTINATION_ACTION_REQUIRED:
            raise BridgeError("Verulink packet is consumed or no longer has a positive attestor quorum")
        p, signatures = ready.protocol_state["packet"], ready.protocol_state["signatures"]
        if c.outbound:
            return self._send(c, plan, ready, c.evm_service, "tokenReceive", [incoming(p), bytes.fromhex("".join(s["sign"][2:] for s in signatures))],
                              on_checkpoint, destination=True)
        serialized = receipt.protocol_state.get("preparedDestinationTransaction")
        if serialized is not None:
            tx = prepared_transaction(serialized, receipt.id)
            pending = ready
        else:
            signers, signed = [s["signer"] for s in signatures], [s["sign"] for s in signatures]
            while len(signers) < 5:
                signers.append(ZERO_ALEO)
                signed.append(signed[0])
            args = [address_literal(p["sender"]), p["recipient"], p["amount"] + "u64", p["sequence"] + "u64", p["height"] + "u64",
                    "[" + ", ".join(signers) + "]", "[" + ", ".join(signed) + "]", p["sourceChain"] + "u128", address_literal(p["sourceAddress"])]
            prepared = self._prepare(c, "token_receive_public", args, proving)
            serialized = prepared.serialized
            tx = prepared_transaction(serialized, prepared.transaction_id)
            pending = ready.replace(id=tx["id"], protocol_state={**ready.protocol_state, "preparedDestinationTransaction": serialized})
        v = inputs(validate_aleo_claim(c, p, tx))
        signers, signed = v[5], v[6]
        if not isinstance(signers, list) or not isinstance(signed, list) or len(signers) != 5 or len(signed) != 5:
            raise BridgeError("Invalid prepared Verulink attestor arrays")
        supplied, padding = [], False
        for signer, sign in zip(signers, signed):
            if signer == ZERO_ALEO:
                padding = True
            elif padding or not isinstance(signer, str) or not isinstance(sign, str):
                raise BridgeError("Invalid prepared Verulink attestor padding")
            else:
                supplied.append({"signer": signer, "sign": sign})
        positive = verified_signatures(c, p, supplied)
        if positive is None or len(positive) != len(supplied):
            raise BridgeError("Prepared Verulink claim lacks a positive attestor quorum")
        self._save(plan, pending, on_checkpoint)
        submit_serialized(self.bridge, serialized, tx["id"])
        state = dict(ready.protocol_state)
        state.pop("preparedDestinationTransaction", None)
        result = ready.replace(destination_tx_id=tx["id"], status=Status.DESTINATION_CONFIRMING, next_action=None, protocol_state=state)
        self._save(plan, result, on_checkpoint, submitted=True)
        return result
