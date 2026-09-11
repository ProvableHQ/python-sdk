"""Balances against the real chain + record scanner — veil's
``balances.integration.test.ts``.  Read-only, but needs the account's own
records (scanner credentials self-provision in conftest).  The DEX API is
used only for the public token registry: public balances come from each
token program's on-chain ``balances`` mapping (the API's ``/balances`` route
was retired in 2026-09)."""
from __future__ import annotations

import pytest

from .conftest import account_tier

pytestmark = pytest.mark.live


def _amm_token_programs(dex) -> list[str]:
    return sorted({t.amm_token_program for t in dex.api.get_tokens()
                   if t.amm_token_program})


@account_tier
def test_public_balances_read_a_u128_per_amm_token_program(account_dex):
    addr = str(account_dex._aleo.default_account.address)
    programs = _amm_token_programs(account_dex)
    assert programs
    public = account_dex.get_public_balances(programs, address=addr)
    assert set(public) == set(programs)
    assert all(isinstance(v, int) and v >= 0 for v in public.values())


@account_tier
def test_private_balances_sum_records_per_program(account_dex):
    programs = sorted({t.underlying_program or t.amm_token_program
                       for t in account_dex.api.get_tokens()
                       if t.underlying_program or t.amm_token_program})
    out = account_dex.get_private_balances(programs)
    assert set(out) == set(programs)
    assert all(isinstance(v, int) and v >= 0 for v in out.values())


@account_tier
def test_get_balances_joins_public_and_private(account_dex):
    addr = str(account_dex._aleo.default_account.address)
    tokens = account_dex.api.get_tokens()
    public = account_dex.get_public_balances(_amm_token_programs(account_dex), address=addr)
    public_by_token = {t.address: public.get(t.amm_token_program, 0)
                       for t in tokens if t.amm_token_program}
    balances = account_dex.get_balances()
    assert balances, "the e2e account holds nothing?"
    for token_id, entry in balances.items():
        assert token_id.endswith("field")
        assert entry["total"] == entry["public"] + entry["private"]
        assert entry["public"] == public_by_token.get(token_id, 0)   # matches the direct chain read
        assert entry["public"] >= 0 and entry["private"] >= 0
        assert entry["total"] > 0                                    # unheld tokens are omitted
        assert entry["symbol"]
