#!/usr/bin/env python3
"""Stress-test rehearsal: the four flows, end to end, via the tier-1 methods.

Usage: python scripts/rehearsal.py [--code REFERRAL] [--home DIR]
Needs: network access.  ``--code`` is an OPTIONAL referral code to credit
the account that shared it; access never depends on one.  Credentials
self-provision.

This script deliberately uses ONLY what AGENTS.md documents — if it needs
anything more, that's a finding.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def main() -> int:
    from aleo_shield_swap import ShieldSwap

    ap = argparse.ArgumentParser()
    ap.add_argument("--home", default=None)
    ap.add_argument("--code", default=None,
                    help="optional referral code to credit (first onboard only)")
    args = ap.parse_args()
    results: list[tuple[str, str]] = []

    dex = ShieldSwap.from_profile(args.home)

    report = dex.onboard(referral_code=args.code)
    results.append(("startup", "ok" if report.funded else "NOT FUNDED"))

    st = dex.status()
    pools = dex.api.get_pools()
    results.append(("discovery",
                    f"ok: {len(pools)} pools, {len(st.balances)} tokens held, "
                    f"{len(st.open_positions)} positions"))

    # Trade a pool whose tokens the account actually holds; size amounts in
    # raw native units from the token's decimals (~1e-5 of a token).
    held = {tid for tid, v in st.balances.items() if v.get("private", 0) > 0}
    pool = next((p for p in pools if p.token0 in held or p.token1 in held),
                pools[0])
    token_in = pool.token0 if pool.token0 in held else pool.token1
    d0 = pool.token0_info.decimals if pool.token0_info else 9
    d1 = pool.token1_info.decimals if pool.token1_info else 9
    dec_in = d0 if token_in == pool.token0 else d1

    batch = dex.swap_many(pool_key=pool.key, token_in_id=token_in,
                          amount_in=10 ** max(dec_in - 5, 1), count=3)
    results.append(("swaps", f"{len(batch.handles)} ok, "
                             f"{len(batch.failures)} failed"))

    collected = dex.collect_all()
    results.append(("collection", f"{len(collected.claimed)} claimed, "
                                  f"{len(collected.still_pending)} pending"))

    lo, hi = dex.get_slot(pool.key).tick_range(width=4)
    minted = dex.mint(pool_key=pool.key, tick_lower=lo, tick_upper=hi,
                      amount0_desired=10 ** max(d0 - 7, 1),  # raw native units
                      amount1_desired=10 ** max(d1 - 7, 1)).delegate()
    pos = dex._position_state(minted.position_token_id)
    from aleo_shield_swap._core import find_position_plaintext
    import time
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:      # wait for the record to scan
        records = dex._aleo.record_provider.find(
            dex._aleo.default_account, program=dex.program, unspent=True)
        if find_position_plaintext(records, pool.key):
            break
        time.sleep(15)
    dex.decrease_liquidity(pool_key=pool.key,
                           liquidity_to_remove=pos.liquidity // 2).delegate()
    results.append(("liquidity", f"ok: minted {minted.position_token_id[:14]}…, "
                                 f"resized; run collection again for earnings"))

    failed = False
    for flow, outcome in results:
        print(f"{flow:12} {outcome}")
        failed |= "NOT" in outcome or outcome.startswith("0 ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
