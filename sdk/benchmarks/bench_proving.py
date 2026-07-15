#!/usr/bin/env python
"""Proving benchmarks: authorization vs full proof, phase by phase.

Times each stage of the write ladder against a local devnode:

* ``authorize`` — request building + signing (local, no proof)
* ``execute``   — proof-free execution of the transition (Response + Trace)
* ``prepare``   — inclusion-path fetch from the node (network)
* ``prove``     — the SNARK itself (``trace.prove_execution``)

Functions benchmarked:

* ``credits.aleo/transfer_public``
* ``credits.aleo/transfer_private``   (spends a real credits record)
* ``shield_swap_v3.aleo/swap``        (spends a token record, blinded output)
* ``shield_swap_v3.aleo/mint``        (spends two token records)

The first prove of each function synthesizes its proving key (and may
download SNARK parameters into ``~/.aleo/resources``) — that iteration is
reported separately as *cold*; later iterations are *warm*.

Run from a repo checkout (uses shield-swap-sdk's vendored programs and
devnode fixture; requires the ``aleo-devnode`` binary)::

    python sdk/benchmarks/bench_proving.py [--prove-iters N] [--auth-iters N]

Nothing here broadcasts a proven transaction — records are only spent by
the (proofless) setup transactions, so proofs can repeat over the same
inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# The devnode setup must be proofless or it would take ~an hour on its own.
os.environ.setdefault("ALEO_DEVNODE_UNPROVEN", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "shield-swap-sdk" / "python"))
sys.path.insert(0, str(REPO_ROOT / "shield-swap-sdk" / "tests" / "integration"))

import devnode_amm  # noqa: E402  (path bootstrap above)
from devnode_amm import setup_amm_devnode  # noqa: E402


def _pc() -> float:
    return time.perf_counter()


def submit_unproven(ctx, call, account) -> str:
    """Land a DexCall on the devnode prooflessly (setup only)."""
    from aleo import testnet as net

    process = ctx.aleo.process
    auth = call._bound.authorize(account).raw
    root = ctx.state_root()
    execution = net.Execution.from_authorization_unproven(auth, root)
    cost, _ = process.execution_cost(execution)
    fee_auth = process.authorize_fee_public(
        account.private_key, cost, 0, execution.execution_id)
    fee = net.Fee.from_authorization_unproven(fee_auth, root)
    return ctx.submit_and_confirm(
        net.Transaction.from_execution(execution, fee), "bench setup")


def build_calls(ctx):
    """Return [(label, bound_call, signer)] for the four benchmarked verbs."""
    from aleo_shield_swap import ShieldSwap

    aleo = ctx.aleo
    admin, user = ctx.admin, ctx.user

    # A credits record for transfer_private (created prooflessly on-chain so
    # trace.prepare can fetch a real inclusion path).
    tx_id = ctx.execute(
        admin, "credits.aleo", "transfer_public_to_private",
        [str(admin.address), "10000000u64"], "privatize credits")
    credits_record = next(
        r for r in ctx.records_of(admin, tx_id) if "microcredits" in r)

    credits_fns = aleo.programs.get("credits.aleo").functions
    transfer_public = credits_fns.transfer_public(str(user.address), "1000000u64")
    transfer_private = credits_fns.transfer_private(
        credits_record, str(user.address), "1000000u64")

    # Shield-swap: a real pool + real token records, landed prooflessly.
    dex = ShieldSwap(aleo, program=devnode_amm.AMM_PROGRAM)
    submit_unproven(ctx, dex.create_pool(
        token0_id=ctx.token0_field, token1_id=ctx.token1_field,
        fee=3000, initial_tick=0, imports=ctx.imports, account=user), user)
    pool_key = dex.derive_pool_key(ctx.token0_field, ctx.token1_field, 3000)

    swap_record = ctx.privatize_token(user, ctx.token0_program, 50_000_000)
    mint_record0 = ctx.privatize_token(user, ctx.token0_program, 100_000_000)
    mint_record1 = ctx.privatize_token(user, ctx.token1_program, 100_000_000)

    swap = dex.swap(
        pool_key=pool_key, token_in_id=ctx.token0_field, amount_in=10_000_000,
        expected_out=0, slippage_bps=0, token_record=swap_record,
        imports=ctx.imports, account=user)._bound
    mint = dex.mint(
        pool_key=pool_key, tick_lower=-600, tick_upper=600,
        amount0_desired=50_000_000, amount1_desired=50_000_000,
        token0_record=mint_record0, token1_record=mint_record1,
        imports=ctx.imports, account=user)._bound

    return [
        ("credits.aleo/transfer_public", transfer_public, user),
        ("credits.aleo/transfer_private", transfer_private, admin),
        (f"{devnode_amm.AMM_PROGRAM}/swap", swap, user),
        (f"{devnode_amm.AMM_PROGRAM}/mint", mint, user),
    ]


def bench_function(ctx, label, bound, signer, auth_iters, prove_iters):
    from aleo import testnet as net

    process = ctx.aleo.process
    url = str(ctx.aleo._provider.url)
    locator = f"{bound.program_id}/{bound.function_name}"

    auth_times = []
    for _ in range(auth_iters):
        t0 = _pc()
        bound.authorize(signer)
        auth_times.append(_pc() - t0)

    prove_rows = []
    for i in range(prove_iters):
        auth = bound.authorize(signer).raw
        t0 = _pc()
        _, trace = process.execute(auth)
        t_execute = _pc() - t0
        t0 = _pc()
        trace.prepare(net.Query.rest(url))
        t_prepare = _pc() - t0
        t0 = _pc()
        trace.prove_execution(locator)
        t_prove = _pc() - t0
        prove_rows.append(
            {"execute": t_execute, "prepare": t_prepare, "prove": t_prove})
        print(f"    [{label}] prove iter {i + 1}/{prove_iters} "
              f"({'cold' if i == 0 else 'warm'}): execute={t_execute:.2f}s "
              f"prepare={t_prepare:.2f}s prove={t_prove:.2f}s", flush=True)

    return {"label": label, "authorize": auth_times, "proofs": prove_rows}


def summarize(results):
    header = (f"{'function':<38} {'auth (ms)':>10} {'execute (s)':>12} "
              f"{'prepare (s)':>12} {'prove cold (s)':>15} {'prove warm (s)':>15}")
    lines = [header, "-" * len(header)]
    for r in results:
        auth_ms = statistics.mean(r["authorize"]) * 1000
        execute = statistics.mean(p["execute"] for p in r["proofs"])
        prepare = statistics.mean(p["prepare"] for p in r["proofs"])
        cold = r["proofs"][0]["prove"]
        warm_rows = [p["prove"] for p in r["proofs"][1:]]
        warm = statistics.mean(warm_rows) if warm_rows else float("nan")
        lines.append(f"{r['label']:<38} {auth_ms:>10.1f} {execute:>12.2f} "
                     f"{prepare:>12.2f} {cold:>15.2f} {warm:>15.2f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(__doc__ or "proving benchmarks").splitlines()[0])
    parser.add_argument("--auth-iters", type=int, default=5)
    parser.add_argument("--prove-iters", type=int, default=2,
                        help="first iteration is cold (key synthesis)")
    parser.add_argument("--json", type=Path, default=None,
                        help="also write raw timings to this file")
    args = parser.parse_args()

    print("Booting devnode + AMM stack (proofless setup)…", flush=True)
    t0 = _pc()
    ctx = setup_amm_devnode()
    print(f"Setup done in {_pc() - t0:.1f}s", flush=True)

    try:
        results = []
        for label, bound, signer in build_calls(ctx):
            print(f"\n== {label} ==", flush=True)
            results.append(bench_function(
                ctx, label, bound, signer, args.auth_iters, args.prove_iters))
        print("\n" + summarize(results))
        if args.json:
            args.json.write_text(json.dumps(results, indent=2))
            print(f"\nraw timings -> {args.json}")
    finally:
        ctx.stop()


if __name__ == "__main__":
    main()
