#!/usr/bin/env python3
# bridge-sdk/codegen/gen_context.py
"""Render AGENTS.md from the SDK's docstrings — the anti-drift context page.

Tier 1 = the lifecycle verbs + the conversation pattern; Tier 2 = the protocol
modules and the registry's route table.  Run with no args to rewrite both
copies of AGENTS.md; ``--check`` exits 1 when they are stale (CI); ``--stdout``
prints instead of writing.
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from aleo_bridge import Bridge  # noqa: E402
from aleo_bridge.eth import EthModule  # noqa: E402
from aleo_bridge.freezelist import FreezeList  # noqa: E402
from aleo_bridge.hyperlane import HyperlaneModule  # noqa: E402
from aleo_bridge.registry import DEFAULT_REGISTRY  # noqa: E402
from aleo_bridge.sol import SolModule  # noqa: E402
from aleo_bridge.xreserve import XReserveModule  # noqa: E402

OUTS = [_ROOT / "AGENTS.md", _ROOT / "python" / "aleo_bridge" / "AGENTS.md"]

TIER1 = ["from_env", "from_profile", "status", "quote", "execute", "wait", "recover", "resume",
         "complete", "pending"]
TIER2 = [
    ("hyperlane.transfer_remote", HyperlaneModule.transfer_remote),
    ("hyperlane.quote_gas_payment", HyperlaneModule.quote_gas_payment),
    ("xreserve.burn", XReserveModule.burn),
    ("xreserve.private_mint", XReserveModule.private_mint),
    ("xreserve.get_attestation", XReserveModule.get_attestation),
    ("shield", Bridge.shield),
    ("unshield", Bridge.unshield),
    ("freezelist.exclusion_proof", FreezeList.exclusion_proof),
    ("eth.transfer_remote", EthModule.transfer_remote),
    ("eth.deposit_usdc", EthModule.deposit_usdc),
    ("eth.quote_transfer_remote", EthModule.quote_transfer_remote),
    ("sol.transfer_remote", SolModule.transfer_remote),
    ("sol.quote_transfer_remote", SolModule.quote_transfer_remote),
]

QUICKSTART = """\
```python
from aleo_bridge import Bridge

bridge = Bridge.from_env()                      # BRIDGE_PRIVATE_KEY (+ EVM/Solana keys) from the environment
print(bridge.status())                          # addresses, balances, pending transfers
quote = bridge.quote("ethereum/wbtc", "aleo/wbtc", amount="0.001", recipient=bridge.aleo_address())
print(quote.fees, quote.amount_out)             # show these to the user BEFORE executing
progress = bridge.execute(quote.plan)           # source step; checkpoints saved to the bound store
progress = bridge.wait(progress)                # stops at resume / complete / done / failed
if progress.next == "resume":   progress = bridge.wait(bridge.resume(progress))
if progress.next == "complete": progress = bridge.wait(bridge.complete(progress, secret_nonce=nonce))
assert progress.next == "done", progress.error
```"""

CONVERSATION_PATTERN = """\
## Serving a chatting user (the conversation pattern)

### Keys and identity

1. **NEVER ask the user to paste a private key into the conversation.**  Keys
   come from the environment only: `BRIDGE_PRIVATE_KEY` (Aleo),
   `EVM_PRIVATE_KEY` + `ETHEREUM_RPC_URL`, `SOLANA_PRIVATE_KEY` (+ optional
   `SOLANA_RPC_URL`), set in the user's own shell before the process starts.
   `Bridge.from_profile()` creates an Aleo key on first use and never writes
   EVM/Solana keys to disk.
2. `status()` first in any session: which chains are configured, balances of
   every bridge asset, and the pending transfers in the checkpoint store.  A
   pending transfer is finished with `recover` → `wait`/`resume`/`complete`,
   never by starting a new one.

### Quote first, always

3. **Always `quote` before `execute`** and show the user the route, the fees
   and `amount_out` in human units with symbols ("2 USDC → 2 USDCx; Hyperlane
   hook payment 8.17 ALEO"), never raw atomic units.  Minimums: xReserve
   needs at least 2 USDC in and strictly more than the 2 USDCx withdrawal fee
   out; Hyperlane moves one atomic unit but network fees and the relayer
   payment cost more than that — say so.
4. Only `execute` after the user confirms.  Through the agent tools every
   write requires `confirm=true`; without it the tool returns the quote and
   moves nothing.  A live mainnet execution additionally needs the user's
   own `BRIDGE_LIVE_MAINNET_EXECUTE` acknowledgement — never set it yourself;
   without it, treat any mainnet run as a rehearsal.

### The source step is irreversible

5. Once the deposit / dispatch / burn is broadcast the funds are committed.
   A timeout, an RPC error or a crash after that point is an UNKNOWN outcome,
   not a failure: recover from the last checkpoint (`recover(checkpoint)` or
   `pending()`) — never run `execute` again for the same transfer.  This is
   the funds-safety rule above all others: never resend after an ambiguous
   broadcast.

### What `progress.next` means for the user

| `progress.next` | Status | Tell the user | Do |
| --- | --- | --- | --- |
| `wait` | source confirming, attestation pending, delivery pending | "In flight; I'll keep checking." | `wait(progress)` (or re-check later from the checkpoint) |
| `resume` | `SOURCE_SUBMISSION_PENDING` | "An approval confirmed / a proof was built but the transfer itself was not submitted; I can submit it now." | confirm, then `resume(progress)` |
| `complete` | `DESTINATION_ACTION_REQUIRED` | "Circle attested your deposit; your private mint needs your signature (and the secret nonce)." | confirm, then `complete(progress, secret_nonce=...)` |
| `done` | `COMPLETED` | "Delivered." Report source and destination transaction ids. | nothing |
| `failed` | `FAILED` / `EXPIRED` | Relay `progress.error`; the source step did not commit funds or was rejected. | nothing — a new transfer needs a new quote |

`wait` raising `PollingTimeoutError` is NOT a failure — say the transfer is
still in flight and check again later.

### Private mints and the secret nonce

6. `mint_mode="private"` (USDC → USDCx) commits `(recipient, secret_nonce)` on
   Ethereum.  The same `secret_nonce` is required by `complete`; the SDK
   **never stores** it and checkpoints exclude it (and every other secret).
   Tell the user to keep it (the default `0scalar` needs no storage but adds
   no entropy).  Only the recipient's Aleo key can complete a private mint —
   make sure the recipient IS the configured Aleo address before depositing.
7. Aleo-origin Hyperlane transfers spend PUBLIC balances: `unshield` a private
   record first.  Hyperlane delivers into public balances; `shield` afterwards
   if the user wants privacy.  Private xReserve burns spend records directly.

### While acting

8. Writes are slow (proving + confirmation ≈ a minute or two on Aleo; Circle
   attestation and Hyperlane relay take minutes).  Never re-submit because a
   call seems slow — `status()` / `recover` first.
9. Confirm, act, report ids.  Errors name their own fix — read the exception
   message and do what it says.
"""


def _entry(name: str, fn: object) -> str:
    try:
        sig = str(inspect.signature(fn))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        sig = "(...)"
    doc = inspect.getdoc(fn) or ""
    return f"### `{name}{sig}`\n\n{doc.strip()}\n"


def _route_table() -> list[str]:
    rows = ["| Route id | Protocol | Environment | Availability |", "| --- | --- | --- | --- |"]
    for route in DEFAULT_REGISTRY.routes(include_unavailable=True, environment=None):
        rows.append(f"| `{route.id}` | {route.protocol} | {route.environment} | {route.availability} |")
    return rows


def render() -> str:
    parts = [
        "# aleo-bridge — agent guide",
        "",
        "> GENERATED from SDK docstrings by `codegen/gen_context.py` — do not",
        "> edit by hand; edit the docstrings and regenerate.",
        "",
        "Typed Python client that moves assets between Aleo, Ethereum and Solana",
        "over the reviewed Hyperlane warp routes and Circle xReserve deployments",
        "(`pip install aleo-bridge-sdk[evm,solana]`, imports as `aleo_bridge`).",
        "MCP alternative: `python -m aleo_bridge.mcp` exposes the same lifecycle as",
        "tools; `aleo_bridge.agent.bridge_tools()` gives Claude-shape tool schemas.",
        f"Registry version `{DEFAULT_REGISTRY.version}`.",
        "",
        "## Tier 1 — the lifecycle (quote → execute → wait, then resume / complete as asked)",
        "",
        QUICKSTART,
        "",
    ]
    parts += [_entry(n, getattr(Bridge, n)) for n in TIER1]
    parts += [
        CONVERSATION_PATTERN,
        "",
        "## Tier 2 — the protocol modules (building your own flows)",
        "",
        "Every Aleo write returns an `AleoCall`: nothing touches the network until",
        "`.simulate()` (free), `.prove()` / `.delegate_prepared()` (proved, not",
        "broadcast — checkpoint it), `.submit_prepared()`, `.transact()` (local",
        "proving + broadcast) or `.delegate()` (DPS + broadcast).  EVM and Solana",
        "writes return `EvmCall` / `SolCall` with `.build()` (unsigned) and `.send()`.",
        "The lifecycle verbs above compose these; use them directly only when you",
        "need a single leg.  Confirm-gated writes and the never-resend rule above",
        "apply here too — these are the same broadcasts, just one leg at a time.",
        "",
    ]
    parts += [_entry(name, fn) for name, fn in TIER2]
    parts += ["### Routes in the pinned registry", ""]
    parts += _route_table()
    parts += ["", "`metadata-required` routes are listed but refused by `quote`/`execute`",
              "until their deployments are reviewed upstream.", ""]
    return "\n".join(parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 when AGENTS.md is stale (CI gate)")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    page = render()
    if args.stdout:
        print(page, end="")
        return 0
    if args.check:
        for out in OUTS:
            current = out.read_text() if out.exists() else ""
            if current != page:
                print(f"{out} is stale — run: python codegen/gen_context.py", file=sys.stderr)
                return 1
        return 0
    for out in OUTS:
        out.write_text(page)
        print(f"wrote {out} ({len(page)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
