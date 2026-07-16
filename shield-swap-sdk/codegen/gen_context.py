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

# Two copies of one render: the repo root (for people browsing the repo)
# and inside the package (ships in the wheel; `python -m aleo_shield_swap`).
OUTS = [_ROOT / "AGENTS.md", _ROOT / "python" / "aleo_shield_swap" / "AGENTS.md"]

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

### Before doing anything: two questions

1. **Existing account?**  When `status()` shows a brand-new, unregistered
   profile, ask whether the user already has a shield-swap account before
   creating anything — their funds and access live on the old key.
   **NEVER ask the user to paste a private key into the conversation.**
   They supply it out-of-band: `export SHIELD_SWAP_PRIVATE_KEY=...` (or
   `SHIELD_SWAP_PRIVATE_KEY_FILE=path`) in their own shell before the
   profile is first created.
2. **Invite code.**  Access is invite-gated per account; `onboard()` stops
   with `NotRedeemedError` until one is supplied.  Ask the user for their
   code; codes are one-time — never guess or reuse.

### After startup: ask what's next

When onboarding reports funded, STOP and ask the user what they want to do
— never launch into a journey unprompted.  Present the options WITH their
context, in plain language (identities, records, and journals are your
business, not the user's):

1. **Their own playbook.**  Ask whether they have instructions of their
   own — a strategy file, notes, a memory store, output from a previous
   session.  If so, read it and treat it as the plan: their document
   decides what to do, the verbs here describe how each step works.

2. **A suggested journey.**  Frame the setting first — Shield Swap is a
   private exchange on Aleo's test network: trading uses test tokens, and
   what is traded, and by whom, stays hidden on the public chain — then
   offer:

   - *Swap tokens* — trade one token for another.  It settles in two
     steps — placing the trade, then collecting what was bought — and you
     do both, so the proceeds arrive without a separate trip.  The
     natural first move.
   - *Several swaps at once* — place a handful of trades and watch them
     all land (`swap_many`); the busiest way to exercise the exchange.
     First show which trades are possible right now (tokens held x live
     pools) and ask how many — and which — they want.
   - *Open a liquidity position* — instead of trading, become the market:
     deposit a pair of tokens so others can trade against them (`mint`).
     The user picks the price range; while the market price sits inside
     it they earn a cut of every trade passing through.
   - *Add or remove liquidity* — grow a position or take some back out
     (`increase_liquidity`/`decrease_liquidity`); whatever comes out
     becomes earnings to collect.
   - *Collect earnings* — sweep everything the account is owed (tokens
     bought in earlier swaps, fees its liquidity earned) into the wallet
     (`collect_all`); good after any trading session.

3. **Developing a trading application or agent?**  Ask whether they are
   building on Shield Swap — a dApp, a trading bot, a server or agent
   integration — rather than (or besides) trading here.  The chat
   journeys above are one way to use the DEX; consumers also build on
   the SDK directly — route builders to Tier 2 below, which opens with
   the client-choice table (bot/server, agent integration, browser dApp)
   and the integration checklist.

4. **A free-form prompt.**  Whatever they describe, map it onto the
   verbs and journeys above before improvising against the SDK.

### While acting

1. `status()` first in any session — never onboard an account that is
   already set up; state lives in `~/.shield-swap/`, not in your context.
2. Ground every proposal in ACTUAL holdings crossed with
   `dex.api.get_pools()`; recommend tick ranges from `get_slot(pool_key)`
   (see `SlotView.tick_range`) instead of waiting for exact parameters.
3. **Never show raw base units to the user** — render amounts in human
   units with the symbol via the token registry's `decimals`
   ("0.0534 ETH", never "53,369,000,000,000 raw").
4. Writes are slow (delegated proving + confirmation ≈ a minute or two).
   Never re-submit because a call seems slow — check `status()` first.
5. Confirm, act, report ids.  Errors name their own fix — read the
   exception message and do what it says."""


DEVELOPER_OPTIONS = """\
### Building a trading application or agent?

Start by asking what they are building — the client choice follows from
where the signing keys live:

| Building | Stack | Keys live |
| --- | --- | --- |
| Bot / server / CLI / notebook | `aleo-sdk` facade + `shield-swap-sdk` (this package) | A local private key (`ShieldSwap.from_profile()` manages it); delegated proving through the Provable prover — fees covered by default. |
| Agent integration | `aleo_shield_swap.agent` (Claude-shape tool schemas + `dispatch_tool`) or `python -m aleo_shield_swap.mcp` (MCP server), over the same client | Same as the underlying client; the tools bind to it. |
| Browser dApp (wallet-signed) | The TypeScript stack: `@provablehq/shield-swap-sdk` + Veil react hooks — not this package | The user's wallet signs and proves. |

What every integration must handle (each enforced or automated by the
verbs above — this list is the review checklist for code that bypasses
them):

- **Auth is layered**: a bearer credential (24h session JWT from the
  challenge/verify handshake, or a durable `ss_…` API token — data/trading
  endpoints only) AND a one-time invite redemption per account.
- **Dynamic-dispatch imports**: every record-spending write must register
  the involved token programs with the prover (the verbs resolve this via
  the token registry; pass `imports=`/`token_*_program=` to override).
- **Tokens are private records**: spendable balances do not appear in
  public reads; one covering record funds an amount — no aggregation.
- **Amounts obey the no-dust rule** both directions (`amount % scale == 0`);
  quote in canonical decimals, transact in raw base units, display human.
- **A `SwapHandle` is the only key to a swap's output** — persist before
  anything else (the journal does this); claim after finalize with retry.
- **Concurrency needs partitioned blinded-identity counters AND disjoint
  input records** — `swap_many` implements the recipe; copy it, don't
  improvise.

Suggested path for a new integrator: (1) `onboard()` a profile — it
doubles as a test fixture; (2) walk swap → `collect_all()` once with the
Tier 1 verbs so the mechanics are concrete; (3) read the reference below
for the surface your app needs; (4) `tests/integration/` and
`scripts/rehearsal.py` in the repo are working reference implementations
of the full journey."""


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
        DEVELOPER_OPTIONS,
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
        for out in OUTS:
            current = out.read_text() if out.exists() else ""
            if current != page:
                print(f"{out} is stale — run: python codegen/gen_context.py",
                      file=sys.stderr)
                return 1
        return 0
    for out in OUTS:
        out.write_text(page)
        print(f"wrote {out} ({len(page)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
