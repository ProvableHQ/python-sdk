"""Frozen addresses against the REAL testnet lists.

``FROZEN_ADDRESS_1`` / ``FROZEN_ADDRESS_2`` hold the private keys of two
accounts that are (or will be) frozen on ``shield_swap_freezelist.aleo`` and
``test_usdcx_freezelist.aleo``.  While a list does not yet contain them the
matching checks skip and say so; once it does they prove that

- the published tree lists the address and the resolver refuses to build a
  proof for it (``FrozenAddressError``, before anything is proved or paid),
- a swap signed by a frozen account and a mint naming a frozen recipient are
  refused by the SDK the same way,
- and that the SDK-built bundles authorize locally.  The program compares
  the reconstructed root with its stored root in the *finalize*, on chain,
  so a bad bundle passed through ``compliance_proofs=`` is only rejected
  after the fee is paid — which is why the SDK verifies every witness it
  builds against the root before a call is prepared.

No test here submits a transaction.  Run: ``-m live`` (the e2e account is
needed for the simulate check).
"""
from __future__ import annotations

import os

import pytest

from aleo import testnet
from aleo.compliance import FrozenAddressError, address_to_field_int
from aleo_shield_swap import ShieldSwap
from aleo_shield_swap.compliance import SHIELD_SWAP_FREEZELIST

from .conftest import ENDPOINT, account_tier

pytestmark = pytest.mark.live

USDCX_LIST = "test_usdcx_freezelist.aleo"
USDCX_PROGRAM = "test_usdcx_stablecoin.aleo"
LISTS = (SHIELD_SWAP_FREEZELIST, USDCX_LIST)
KEY_VARS = ("FROZEN_ADDRESS_1", "FROZEN_ADDRESS_2")


def _frozen_keys() -> list[tuple[str, str]]:
    keys = [(name, os.environ.get(name)) for name in KEY_VARS]
    found = [(name, key) for name, key in keys if key]
    if not found:
        pytest.skip("set FROZEN_ADDRESS_1 / FROZEN_ADDRESS_2 (private keys of the frozen test accounts)")
    return found  # type: ignore[return-value]


def _address(private_key: str) -> str:
    return str(testnet.PrivateKey.from_string(private_key).address)


def _is_leaf(dex: ShieldSwap, program: str, address: str) -> bool:
    tree = dex._aleo.network.get_compliance_freeze_list(program)
    assert tree is not None
    leaves = (len(tree) + 1) // 2
    return address_to_field_int(address) in tree[:leaves]


def _frozen_on(dex: ShieldSwap, program: str) -> list[str]:
    return [_address(key) for _, key in _frozen_keys() if _is_leaf(dex, program, _address(key))]


@pytest.fixture(scope="module")
def frozen_dex(live_dex_module):
    """A client whose signer is the first frozen account (reads only; the account needs no funds)."""
    from aleo import Aleo, HTTPProvider
    _, key = _frozen_keys()[0]
    aleo = Aleo(HTTPProvider(ENDPOINT, network="testnet"))
    aleo.default_account = aleo.account.from_private_key(key)
    dex = ShieldSwap(aleo)
    dex.api.authenticate(str(aleo.default_account.address),
                         lambda msg: str(aleo.account.sign(msg.encode(), aleo.default_account)))
    return dex


def test_frozen_addresses_are_listed_and_refused(live_dex):
    seen = []
    for name, key in _frozen_keys():
        address = _address(key)
        for program in LISTS:
            listed = _is_leaf(live_dex, program, address)
            seen.append((name, program, listed))
            print(f"  {name} {address[:12]}… on {program}: {'FROZEN' if listed else 'not listed'}")
            if listed:
                with pytest.raises(FrozenAddressError, match="is on the freeze list"):
                    live_dex.freezelist.exclusion_proof(address, program)
            else:
                proof = live_dex.freezelist.exclusion_proof(address, program)
                assert proof.count("leaf_index") == 2
    if not any(listed for _, _, listed in seen):
        pytest.skip("neither list contains the frozen test accounts yet")


def test_frozen_signer_swap_is_refused_before_proving(frozen_dex, live_dex):
    signer = str(frozen_dex._aleo.default_account.address)
    if not _is_leaf(live_dex, USDCX_LIST, signer):
        pytest.skip(f"{signer[:12]}… is not on {USDCX_LIST} yet")
    tokens = {t.address: t for t in frozen_dex.api.get_tokens()}
    usdcx = next(t for t in tokens.values() if t.symbol == "USDCx")
    pool = next(p for p in frozen_dex.api.get_pools() if usdcx.address in (p.token0, p.token1))
    token_out = pool.token1 if pool.token0 == usdcx.address else pool.token0
    amount_in = 10 ** (usdcx.decimals - 2)
    expected = frozen_dex._quote_expected_out(token_in_id=usdcx.address, token_out_id=token_out,
                                              amount_in=amount_in, pool_key=pool.key)
    # A record literal stands in for a scan: the refusal comes before the record is ever used.
    record = f"{{ owner: {signer}.private, amount: {amount_in}u128.private, _nonce: 0group.public }}"
    with pytest.raises(FrozenAddressError, match="is on the freeze list"):
        frozen_dex.swap(pool_key=pool.key, token_in_id=usdcx.address, amount_in=amount_in,
                        expected_out=expected, slippage_bps=100, token_in_program=usdcx.underlying_program,
                        token_record=record, track=False)


@account_tier
def test_mint_naming_a_frozen_recipient_is_refused_before_proving(account_dex):
    frozen = _frozen_on(account_dex, SHIELD_SWAP_FREEZELIST)
    if not frozen:
        pytest.skip(f"no frozen test account on {SHIELD_SWAP_FREEZELIST} yet")
    held = {tid: v["private"] for tid, v in account_dex.get_balances().items()}
    pool = next(p for p in account_dex.api.get_pools() if held.get(p.token0, 0) and held.get(p.token1, 0))
    slot = account_dex.get_slot(pool.key)
    lo, hi = slot.tick_range(width=5)
    with pytest.raises(FrozenAddressError, match="is on the freeze list"):
        account_dex.mint(pool_key=pool.key, tick_lower=lo, tick_upper=hi, amount0_desired=1, amount1_desired=1,
                         recipient=frozen[0])
    with pytest.raises(FrozenAddressError, match="is on the freeze list"):
        account_dex.mint(pool_key=pool.key, tick_lower=lo, tick_upper=hi, amount0_desired=1, amount1_desired=1,
                         withdrawal=frozen[0])


@account_tier
def test_sdk_built_bundles_authorize_locally(account_dex):
    """``simulate()`` runs the transitions off-chain: the bundles the SDK builds
    for a wrapped-input swap pass that stage.  (A bundle that does not match
    the stored root passes it too — the root is checked in the finalize — so
    this is a check of the SDK's own output, not a substitute for the chain.)"""
    tokens = {t.address: t for t in account_dex.api.get_tokens()}
    usdcx = next(t for t in tokens.values() if t.symbol == "USDCx")
    pool = next(p for p in account_dex.api.get_pools() if usdcx.address in (p.token0, p.token1))
    token_out = pool.token1 if pool.token0 == usdcx.address else pool.token0
    amount_in = 10 ** (usdcx.decimals - 2)
    expected = account_dex._quote_expected_out(token_in_id=usdcx.address, token_out_id=token_out,
                                               amount_in=amount_in, pool_key=pool.key)
    call = account_dex.swap(pool_key=pool.key, token_in_id=usdcx.address, amount_in=amount_in,
                            expected_out=expected, slippage_bps=100, token_in_program=usdcx.underlying_program,
                            track=False)
    assert call._bound.function_name == "swap_from_wrapped"
    call.simulate()
    signer = str(account_dex._aleo.default_account.address)
    explicit = account_dex.swap(pool_key=pool.key, token_in_id=usdcx.address, amount_in=amount_in,
                                expected_out=expected, slippage_bps=100, token_in_program=usdcx.underlying_program,
                                track=False,
                                compliance_proofs=(account_dex.freezelist.exclusion_proof(signer, USDCX_PROGRAM),))
    explicit.simulate()
