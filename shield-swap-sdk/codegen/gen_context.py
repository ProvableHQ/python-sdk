#!/usr/bin/env python3
"""Render AGENTS.md from the SDK's docstrings — the anti-drift context page.

Everything an agent needs is derived from code: tier 1 (lifecycle + the
conversation pattern), tier 2 (the development guide's building-block
reference + registration stages).  Run with no args to rewrite AGENTS.md;
``--check`` exits 1 when the committed page is stale (CI); ``--stdout``
prints instead of writing.
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "python"))

from aleo_shield_swap import ShieldSwap, derivations  # noqa: E402
from aleo_shield_swap.api import ApiClient  # noqa: E402
from aleo_shield_swap.lifecycle import REGISTRATION_STAGES  # noqa: E402

OUT = _ROOT / "AGENTS.md"

TIER1 = ["from_profile", "onboard", "status", "get_positions",
         "swap_many", "collect_all"]
TIER2_CLIENT = ["swap", "claim_swap_output", "create_pool", "mint",
                "increase_liquidity", "decrease_liquidity", "collect", "burn",
                "get_pool", "get_slot", "get_swap_output", "get_balances",
                "get_private_balances", "derive_pool_key", "derive_tick_key"]
TIER2_API = ["authenticate", "access_status", "redeem_code",
             "request_airdrop", "get_airdrop_job", "create_api_token",
             "get_pools", "get_tokens", "get_route"]
TIER2_DERIVATIONS = ["blinded_identity_at", "next_blinded_identity"]

QUICKSTART = """\
```python
from aleo_shield_swap import ShieldSwap

dex = ShieldSwap.from_profile()          # key material auto-managed on disk
dex.onboard(invite_code="...")           # first run only; no-op afterwards
pools = dex.api.get_pools()
report = dex.swap_many(pool_key=pools[0].key, token_in_id=pools[0].token0,
                       amount_in=10**6, count=5)
dex.collect_all()                        # any session, any time
```"""

CONVERSATION_PATTERN = """\
## Serving a chatting user (the conversation pattern)

1. Run `status()` first — is the account registered and funded, what does
   it hold, what is pending.  Never onboard an account that `status()`
   shows is already set up.
2. Present tradable pairs from the user's ACTUAL balances crossed with
   `dex.api.get_pools()`, and ask which they want to trade.
3. Proactively recommend minting/liquidity options: pools matching their
   tokens, tick ranges around `get_slot(pool_key)` (see
   `SlotView.tick_range`).  Don't wait for exact parameters.
4. Confirm, act, report ids.

The invite code (first run) is the only thing you should ever need to ask
the user for.  Errors name their own fix — read the exception message and
do what it says.  State (handles, counters, positions) lives in
`~/.shield-swap/`, not in your context: any fresh session re-orients with
`status()`."""


def _entry(name: str, fn: object) -> str:
    try:
        sig = str(inspect.signature(fn))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        sig = "(...)"
    doc = inspect.getdoc(fn) or ""
    return f"### `{name}{sig}`\n\n{doc.strip()}\n"


def render() -> str:
    parts = [
        "# shield-swap — agent guide",
        "",
        "> GENERATED from SDK docstrings by `codegen/gen_context.py` — do not",
        "> edit by hand; edit the docstrings and regenerate.",
        "",
        "Typed Python client for the shield_swap AMM on Aleo",
        "(`pip install shield-swap-sdk`, imports as `aleo_shield_swap`).",
        "MCP alternative: `python -m aleo_shield_swap.mcp` exposes the same",
        "lifecycle as tools.",
        "",
        "## Tier 1 — the lifecycle (six lines end to end)",
        "",
        QUICKSTART,
        "",
    ]
    parts += [_entry(n, getattr(ShieldSwap, n)) for n in TIER1]
    parts += [
        CONVERSATION_PATTERN,
        "",
        "## Tier 2 — the development guide (building your own tools)",
        "",
        "Every write verb returns a prepared `DexCall`: nothing touches the",
        "network until a terminal verb — `.simulate()` (local, free),",
        "`.transact()` (local proving, slow), or `.delegate()` (delegated",
        "proving — the practical path).",
        "",
        "### Registration, unbundled",
        "",
        "`onboard()` is a stage list, and the steps WILL change over time —",
        "introspect `lifecycle.REGISTRATION_STAGES`, never hard-code the",
        "sequence.  Current stages:",
        "",
    ]
    parts += [f"- `{s.name}`" for s in REGISTRATION_STAGES]
    parts += [
        "",
        "Apps that own their onboarding call the same `dex.api` endpoints",
        "the stages use:",
        "",
    ]
    parts += [_entry(f"api.{n}", getattr(ApiClient, n)) for n in TIER2_API]
    parts += [
        "### Counters & blinding",
        "",
        "Blinded identities derive deterministically from (view key, counter,",
        "program).  Counters must NEVER be reused: reserve them via",
        "`dex.journal.reserve_counters(n)` (what `swap_many` does), or probe",
        "on-chain when no journal exists.  Persist `SwapHandle`s — the",
        "blinding factor is the claim secret.",
        "",
    ]
    parts += [_entry(n, getattr(derivations, n)) for n in TIER2_DERIVATIONS]
    parts += ["### Chain verbs", ""]
    parts += [_entry(n, getattr(ShieldSwap, n)) for n in TIER2_CLIENT]
    return "\n".join(parts) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when AGENTS.md is stale (CI gate)")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    page = render()
    if args.stdout:
        print(page, end="")
        return 0
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != page:
            print("AGENTS.md is stale — run: python codegen/gen_context.py",
                  file=sys.stderr)
            return 1
        return 0
    OUT.write_text(page)
    print(f"wrote {OUT} ({len(page)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
