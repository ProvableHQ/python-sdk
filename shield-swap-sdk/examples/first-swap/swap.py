"""Trade 1.5 USDCx for ETH on testnet and retain recovery state."""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
from pathlib import Path
import time
from typing import Any

HOME = Path.cwd() / ".shield-first-swap"
ENDPOINT = "https://edge.provable.com/api"


class ExampleError(RuntimeError):
    """Describe a known example failure without including SDK response data."""


def format_amount(amount: int, decimals: int) -> str:
    """Render integer token units exactly, without floating-point rounding."""
    if decimals < 0 or amount < 0:
        raise ExampleError("Invalid output amount or token decimals; inspect the saved journal")
    whole, fraction = divmod(amount, 10**decimals)
    suffix = str(fraction).zfill(decimals).rstrip("0") if decimals else ""
    return f"{whole}.{suffix}" if suffix else str(whole)


def error_details(error: Exception) -> dict[str, str]:
    """Retain actionable example errors while withholding arbitrary SDK messages."""
    return {"error_type": type(error).__name__,
            "action": str(error) if isinstance(error, ExampleError) else
            "Preserve .shield-first-swap; inspect journal locally and use --claim after a submission"}


def save(path: Path, data: dict[str, Any]) -> None:
    """Replace an outcome or submission file after flushing its complete contents."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def finish(dex: Any, home: Path, attempts: int = 40) -> None:
    """Collect the journaled swap and rebuild a result after a restart."""
    for attempt in range(attempts):
        events = dex.journal.events()
        swaps = [event for event in events if event["type"] == "swap"]
        claims = [event for event in events if event["type"] == "claim"]
        for swap in swaps:
            claim = next((c for c in claims if c["swap_id"] == swap["swap_id"]), None)
            if claim is not None and claim["amount_out"] > 0:
                intent = json.loads((home / "submission.json").read_text())
                save(home / "result.json", {
                    "network": "testnet", "address": dex.profile.address,
                    "swap_transaction_id": swap["transaction_id"],
                    "claim_transaction_id": claim["transaction_id"],
                    "received": {"symbol": "ETH",
                                 "amount": format_amount(claim["amount_out"], intent["output_decimals"]),
                                 "amount_base_units": str(claim["amount_out"]),
                                 "decimals": intent["output_decimals"]},
                })
                return
        if not swaps:
            raise ExampleError("Submission outcome unknown; preserve state and inspect the journal before any new trade")
        if attempt + 1 < attempts:
            dex.collect_all()
            time.sleep(15)
    raise ExampleError("Claim not confirmed; preserve state and rerun with --claim")


def run(dex: Any, home: Path, claim_only: bool = False) -> None:
    """Fund and submit one testnet trade, or recover its existing journaled claim."""
    if dex.profile.network != "testnet" or dex.profile.endpoint != ENDPOINT:
        raise ExampleError("This example requires its own testnet profile at the default endpoint")
    marker = home / "submission.json"
    if claim_only:
        if not marker.exists():
            raise ExampleError("No example submission exists")
        finish(dex, home)
        return
    if marker.exists():
        raise ExampleError("A swap was already attempted; inspect result.json or use --claim")
    if dex.journal.events():
        # Onboarding events are harmless; existing trades belong to another run.
        if any(e["type"] in {"swap", "swap_failed", "counters_reserved", "position"}
               for e in dex.journal.events()):
            raise ExampleError("Existing trading journal requires recovery, not another swap")
    onboard = dex.onboard()
    if not onboard.funded:
        raise ExampleError("Funding is not ready; rerun later with the same state")
    tokens = dex.api.get_tokens()
    source = next(t for t in tokens if t.symbol == "USDCx")
    target = next(t for t in tokens if t.symbol == "ETH")
    amount = 15 * 10**source.decimals // 10
    pools = sorted((p for p in dex.api.get_pools()
                    if {p.token0, p.token1} == {source.id, target.id}), key=lambda p: p.key)
    if not pools:
        raise ExampleError("No direct USDCx/ETH pool exists; no swap submitted")
    for attempt in range(40):
        if dex.get_balances().get(source.id, {}).get("private", 0) >= amount:
            break
        if attempt == 39:
            raise ExampleError("Spendable USDCx is not ready; retry later")
        time.sleep(15)
    # Persist intent before a request that could broadcast, even if its response is lost.
    save(marker, {"network": "testnet", "pool_key": pools[0].key,
                  "token_in_id": source.id, "amount_in": str(amount),
                  "output_decimals": target.decimals})
    report = dex.swap_many(pool_key=pools[0].key, token_in_id=source.id,
                           amount_in=amount, count=1, slippage_bps=50)
    if report.failures or len(report.handles) != 1:
        raise ExampleError("Swap submission did not return one handle; preserve the journal and use --claim")
    finish(dex, home)


def main() -> int:
    """Lock private state and save outcomes without logging account or SDK data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim", action="store_true", help="Recover the existing swap without funding or trading again")
    args = parser.parse_args()
    os.umask(0o077)
    logging.disable(logging.CRITICAL)
    HOME.mkdir(mode=0o700, exist_ok=True)
    HOME.chmod(0o700)
    (HOME / ".gitignore").write_text("*\n")
    with (HOME / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 1
        try:
            # These SDK overrides are deliberately unsupported by this fixed testnet example.
            for name in ("SHIELD_SWAP_PRIVATE_KEY_FILE", "SHIELD_SWAP_API_URL",
                         "ALEO_E2E_API_KEY", "ALEO_E2E_CONSUMER_ID"):
                os.environ.pop(name, None)
            from aleo_shield_swap import ShieldSwap
            dex = ShieldSwap.from_profile(HOME, network="testnet", endpoint=ENDPOINT)
            run(dex, HOME, args.claim)
            return 0
        except Exception as error:
            save(HOME / "error.json", error_details(error))
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
