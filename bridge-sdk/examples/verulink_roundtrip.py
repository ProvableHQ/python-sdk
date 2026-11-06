"""Preview or recover a public ALEO → BSC → Aleo round trip."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from aleo_bridge import Bridge, BridgeError, Ethereum, create_checkpoint
from aleo_bridge.units import format_decimal_amount, parse_decimal_amount

ACK = "I_ACKNOWLEDGE_THIS_SUBMITS_MAINNET_TRANSACTIONS"
BNB_RESERVE = 10**15  # 0.001 BNB budget for claim, approval and burn; not a fee estimate.
ROUTES = ("verulink:aleo/aleo->bsc/aleo", "verulink:bsc/aleo->aleo/aleo")


def require_execution_ack() -> None:
    if os.environ.get("BRIDGE_LIVE_MAINNET_EXECUTE") != ACK:
        raise BridgeError("Execution requires BRIDGE_LIVE_MAINNET_EXECUTE=" + ACK)


def build_bridge(*, execute: bool) -> Bridge:
    from aleo import Aleo, HTTPProvider
    if execute:
        require_execution_ack()
    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_RPC_URL", "https://edge.provable.com/api"), network="mainnet"))
    if execute:
        aleo.default_account = aleo.account.from_private_key(os.environ["ALEO_PRIVATE_KEY"])
    bsc = Ethereum(os.environ.get("BSC_RPC_URL", "https://bsc-dataseed.bnbchain.org"),
                   private_key=os.environ["EVM_PRIVATE_KEY"] if execute else None)
    return Bridge(aleo, evm={"bsc": bsc})


def _save(path: Path, state: dict[str, Any]) -> None:
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".verulink-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def run_roundtrip(bridge: Bridge, *, aleo_address: str, bsc_address: str, amount: str,
                  state_path: Path, execute: bool = False, timeout: float = 1200) -> None:
    """Quote both legs, or continue one journal with submission markers at each wallet boundary.

    The return spends the verified packet amount received on BSC. The same state
    file and arguments resume the journey; an uncertain submission requires its
    transaction hash before continuing.
    """
    def quote(index: int, value: str):
        return bridge.quote(route=ROUTES[index], amount=value,
                            sender=aleo_address if index == 0 else bsc_address,
                            recipient=bsc_address if index == 0 else aleo_address)

    if not execute:
        outbound = quote(0, amount)
        inbound = quote(1, outbound.amount_out)
        print("BSC received estimate:", outbound.amount_out, "ALEO returned estimate:", inbound.amount_out)
        return
    require_execution_ack()
    path = Path(state_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(str(path) + ".lock")
    try:
        fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise BridgeError("Journey lock exists; confirm its owner has stopped before removing it") from exc
    try:
        with os.fdopen(fd, "w") as lock:
            lock.write(str(os.getpid()))
        state = json.loads(path.read_text()) if path.exists() else {
            "version": 1, "aleo": aleo_address, "bsc": bsc_address, "amount": amount,
            "legs": [{"amount": amount}]}
        if (state.get("version") != 1 or state.get("aleo") != aleo_address or state.get("bsc") != bsc_address
                or state.get("amount") != amount or not isinstance(state.get("legs"), list)
                or not 1 <= len(state["legs"]) <= 2):
            raise BridgeError("Saved journey differs; use its original accounts and amount")
        if (bridge.aleo_address() != aleo_address
                or bridge.evm("bsc").conn.require_address().lower() != bsc_address.lower()):
            raise BridgeError("Configured signers differ from the journey addresses")
        if not state["legs"][0].get("source_attempted") and bridge.evm("bsc").conn.w3.eth.get_balance(bsc_address) < BNB_RESERVE:
            raise BridgeError("Fund the BSC account with at least 0.001 BNB before committing ALEO")
        for index in range(2):
            if len(state["legs"]) == index:
                state["legs"].append({"amount": format_decimal_amount(int(state["legs"][0]["received"]), 6)})
            leg = state["legs"][index]
            expected = amount if index == 0 else format_decimal_amount(int(state["legs"][0]["received"]), 6)
            if leg.get("amount") != expected:
                raise BridgeError("Saved leg amount differs from the verified received amount")

            def checkpoint(cp: Any) -> None:
                leg["checkpoint"] = cp.to_dict()
                _save(path, state)

            if not leg.get("checkpoint"):
                if leg.get("source_attempted"):
                    raise BridgeError("Source submission is uncertain; recover its wallet transaction hash before continuing")
                quoted = quote(index, leg["amount"])
                if index == 0:
                    quote(1, quoted.amount_out)  # The estimated return must clear its fees and minimum.
                leg["source_attempted"] = True
                _save(path, state)
                bridge.execute(quoted.plan, on_checkpoint=checkpoint, timeout_seconds=timeout)
            if not leg.get("checkpoint"):
                raise BridgeError("Wallet returned without a checkpoint; reconcile before retrying")
            progress = bridge.recover(leg["checkpoint"])
            sender = aleo_address if index == 0 else bsc_address
            recipient = bsc_address if index == 0 else aleo_address
            if (progress.plan.route_id != ROUTES[index] or progress.plan.amount_atomic != parse_decimal_amount(expected, 6)
                    or not isinstance(progress.plan.sender, str) or progress.plan.sender.lower() != sender.lower()
                    or progress.plan.recipient.lower() != recipient.lower()):
                raise BridgeError("Saved checkpoint differs from the configured journey leg")
            # Recheck completed legs too; the return budget is always backed by chain evidence.
            deadline = time.monotonic() + timeout
            while True:
                print("Leg", index + 1, progress.next, progress.receipt.source_tx_id, progress.receipt.destination_tx_id)
                if progress.next == "done":
                    net = progress.receipt.protocol_state.get("packet", {}).get("amount")
                    if not isinstance(net, str) or not net.isascii() or not net.isdecimal() or int(net) <= 0:
                        raise BridgeError("Completed transfer has no verified packet amount")
                    leg["received"] = net
                    checkpoint(create_checkpoint(progress.plan, progress.receipt, bridge.registry))
                    break
                if progress.next == "failed":
                    raise BridgeError(progress.error or "Verulink transfer failed or is held; retain the checkpoint")
                if time.monotonic() >= deadline:
                    raise BridgeError("Polling timed out; recover the same journey")
                if progress.next == "resume":
                    if not leg["checkpoint"].get("source", {}).get("preparedTransaction"):
                        raise BridgeError("Approval-only BSC recovery requires wallet reconciliation")
                    progress = bridge.resume(progress, on_checkpoint=checkpoint)
                elif progress.next == "complete":
                    if leg.get("claim_attempted") and not leg["checkpoint"].get("destination", {}).get("preparedTransaction"):
                        raise BridgeError("Claim submission is uncertain; add its destination transaction hash before continuing")
                    leg["claim_attempted"] = True
                    _save(path, state)
                    progress = bridge.complete(progress, on_checkpoint=checkpoint)
                else:
                    progress = bridge.wait(progress, timeout_seconds=max(0, deadline - time.monotonic()),
                        on_update=lambda update: checkpoint(create_checkpoint(update.plan, update.receipt, bridge.registry)))
                checkpoint(create_checkpoint(progress.plan, progress.receipt, bridge.registry))
        print("Completed; ALEO returned:", format_decimal_amount(int(state["legs"][1]["received"]), 6), "State:", path)
    finally:
        lock_path.unlink()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aleo-address", required=True)
    parser.add_argument("--bsc-address", required=True)
    parser.add_argument("--amount", default="20", help="Public ALEO sent on the first leg.")
    parser.add_argument("--state", default="~/.aleo-bridge/verulink-roundtrip.json")
    parser.add_argument("--timeout", type=float, default=1200, help="Seconds per leg before returning with its saved checkpoint.")
    parser.add_argument("--execute", action="store_true", help="Submit or recover both MAINNET legs; costs real funds.")
    args = parser.parse_args(argv)
    if args.execute:
        require_execution_ack()
    bridge = build_bridge(execute=args.execute)
    run_roundtrip(bridge, aleo_address=args.aleo_address, bsc_address=args.bsc_address, amount=args.amount,
                  state_path=Path(args.state), execute=args.execute, timeout=args.timeout)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Recover the same journey before submitting again.")
        raise SystemExit(130)
    except Exception as exc:
        # Provider errors can contain credentials; never dump their request or traceback.
        print(type(exc).__name__ + ": journey unfinished. Keep its state file and reconcile any uncertain transaction.")
        raise SystemExit(2)
