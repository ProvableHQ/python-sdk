"""Freeze-list proofs against the REAL testnet: both lists Shield Swap checks
(its own and the USDCx stablecoin's), the published trees, and the SDK's
automatic resolution through ``dex.freezelist``.  Read-only; the write
lifecycles (``test_swap_lifecycle.py``, ``test_liquidity_lifecycle_live.py``)
exercise the same resolver on every swap, claim, mint, increase, rebalance
and collect they submit."""
from __future__ import annotations

import pytest

from aleo.compliance import EMPTY_MERKLE_PROOF_PAIR, EMPTY_TREE_ROOT, ComplianceProofContext, generate_freezelist_proof
from aleo_shield_swap.compliance import SHIELD_SWAP_FREEZELIST

pytestmark = pytest.mark.live

USDCX_LIST = "test_usdcx_freezelist.aleo"
ANY_ADDRESS = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"


@pytest.mark.parametrize("program", [SHIELD_SWAP_FREEZELIST, USDCX_LIST])
def test_published_tree_matches_the_on_chain_root(live_dex, program):
    aleo = live_dex._aleo
    tree = aleo.network.get_compliance_freeze_list(program)
    assert tree is not None and len(tree) >= 3 and (len(tree) + 1) % 2 == 0
    root = live_dex.freezelist.root(program)
    assert tree[-1] == root
    # A witness built from the published tree reconstructs that root.
    proof = generate_freezelist_proof(tree, ANY_ADDRESS, root, "testnet")
    assert proof.startswith("[{ siblings: [") and proof.count("leaf_index") == 2
    if root == EMPTY_TREE_ROOT:
        assert proof == EMPTY_MERKLE_PROOF_PAIR


def test_programs_without_a_list_answer_none(live_dex):
    assert live_dex._aleo.network.get_compliance_freeze_list("credits.aleo") is None


def test_resolver_finds_the_list_behind_each_token_and_caches_it(live_dex):
    fl = live_dex.freezelist
    fl.clear()
    # The USDCx wrapper's underlying stablecoin names its list through its imports.
    assert live_dex._freezelist_for_token("x", "test_usdcx_stablecoin.aleo") == USDCX_LIST
    assert live_dex._freezelist_for_token("x", "credits.aleo") is None
    first = fl.resolve([ComplianceProofContext(SHIELD_SWAP_FREEZELIST, ANY_ADDRESS), ComplianceProofContext(USDCX_LIST, ANY_ADDRESS), ComplianceProofContext(None, ANY_ADDRESS)])
    assert len(first) == 3 and first[2] == EMPTY_MERKLE_PROOF_PAIR
    assert first[0] == fl.exclusion_proof(ANY_ADDRESS, SHIELD_SWAP_FREEZELIST)
    assert first[1] == fl.exclusion_proof(ANY_ADDRESS, "test_usdcx_stablecoin.aleo")
    for program in (SHIELD_SWAP_FREEZELIST, USDCX_LIST):
        entry = fl._cache.current("testnet", program, fl.root(program))
        assert entry is not None and entry.tree[-1] == entry.root


@pytest.mark.slow
def test_usdcx_swap_and_claim_prove_against_both_lists(account_dex):
    """Write tier: a USDCx → ETH swap spends a wrapped input (sender bundle
    against ``test_usdcx_freezelist.aleo``) and its claim proves the signer
    against Shield Swap's list — both bundles built by the SDK."""
    import time

    from aleo_shield_swap.errors import SwapOutputNotFinalizedError
    from .conftest import poll_until, with_retry

    dex = account_dex
    acct = dex._aleo.default_account
    tokens = {t.address: t for t in dex.api.get_tokens()}
    usdcx = next(t for t in tokens.values() if t.symbol == "USDCx")
    pools = [p for p in dex.api.get_pools() if usdcx.address in (p.token0, p.token1)
             and {tokens[p.token0].symbol, tokens[p.token1].symbol} == {"USDCx", "ETH"}]
    assert pools, "no live USDCx/ETH pool"
    pool = max(pools, key=lambda p: dex.get_slot(p.key).liquidity)
    token_out = pool.token1 if pool.token0 == usdcx.address else pool.token0
    amount_in = 10 ** (usdcx.decimals - 2)                       # 0.01 USDCx
    expected = dex._quote_expected_out(token_in_id=usdcx.address, token_out_id=token_out,
                                       amount_in=amount_in, pool_key=pool.key)
    call = dex.swap(pool_key=pool.key, token_in_id=usdcx.address, amount_in=amount_in,
                    expected_out=expected, slippage_bps=100, token_in_program=usdcx.underlying_program)
    assert call._bound.function_name == "swap_from_wrapped"
    handle = with_retry(lambda: call.delegate(acct))
    assert handle.swap_id
    deadline = time.time() + 300
    while True:
        try:
            dex.get_swap_output(handle.swap_id)
            break
        except SwapOutputNotFinalizedError:
            if time.time() > deadline:
                pytest.fail("swap request did not finalize within 5 minutes")
            time.sleep(10)
    out = dex.get_swap_output(handle.swap_id)
    claim = dex.claim_swap_output(handle)
    assert claim._bound.function_name in ("claim_to_arc20_no_refund", "claim_to_arc20_refund_wrapped", "claim_swap_output",
                                          "claim_swap_output_no_refund"), claim._bound.function_name
    result = with_retry(lambda: claim.delegate(acct))
    assert result.amount_out > 0 and result.amount_out == out.amount_out
    assert poll_until(lambda: _absent(dex, handle), 24, 5)
    # Every bundle came from the resolver: both lists are now cached for this client.
    for program in (SHIELD_SWAP_FREEZELIST, USDCX_LIST):
        assert dex.freezelist._cache.current("testnet", program, dex.freezelist.root(program)) is not None


def _absent(dex, handle) -> bool:
    from aleo_shield_swap.errors import SwapOutputNotFinalizedError
    try:
        dex.get_swap_output(handle.swap_id)
        return False
    except SwapOutputNotFinalizedError:
        return True
