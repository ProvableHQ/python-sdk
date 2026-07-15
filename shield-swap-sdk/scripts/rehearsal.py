#!/usr/bin/env python3
"""Stress-test rehearsal: the four flows, end to end, via the tier-1 verbs.

Usage: python scripts/rehearsal.py [--code INVITE] [--home DIR]
Needs: network access; ALEO_E2E_API_KEY/ALEO_E2E_CONSUMER_ID (until key
provisioning has an endpoint).  Without --code, an invite is minted with
ALEO_E2E_PRIVATE_KEY (the e2e account can generate codes).

This script deliberately uses ONLY what AGENTS.md documents — if it needs
anything more, that's a finding.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def _mint_invite_code() -> str:
    import aleo
    from aleo_shield_swap import ApiClient

    pk = aleo.testnet.PrivateKey.from_string(os.environ["ALEO_E2E_PRIVATE_KEY"])
    api = ApiClient()
    api.authenticate(str(pk.address), lambda m: str(pk.sign(m.encode())))
    return api._post("/access/generate", {"count": 1})["data"]["codes"][0]


def main() -> int:
    from aleo_shield_swap import ShieldSwap

    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=None)
    ap.add_argument("--code", default=None)
    args = ap.parse_args()
    results: list[tuple[str, str]] = []

    dex = ShieldSwap.from_profile(args.home)
    code = args.code or _mint_invite_code()

    report = dex.onboard(invite_code=code)
    results.append(("startup", "ok" if report.funded else "NOT FUNDED"))

    st = dex.status()
    pools = dex.api.get_pools()
    results.append(("discovery",
                    f"ok: {len(pools)} pools, {len(st.balances)} tokens held, "
                    f"{len(st.open_positions)} positions"))

    batch = dex.swap_many(pool_key=pools[0].key, token_in_id=pools[0].token0,
                          amount_in=10**5, count=3)
    results.append(("swaps", f"{len(batch.handles)} ok, "
                             f"{len(batch.failures)} failed"))

    collected = dex.collect_all()
    results.append(("collection", f"{len(collected.claimed)} claimed, "
                                  f"{len(collected.still_pending)} pending"))
    # Liquidity flow (mint/adjust) joins once ops designates a rehearsal
    # pool with mintable ranges — noted rather than silently skipped.
    results.append(("liquidity", "SKIPPED: no designated rehearsal pool yet"))

    failed = False
    for flow, outcome in results:
        print(f"{flow:12} {outcome}")
        failed |= "NOT" in outcome or outcome.startswith("0 ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
