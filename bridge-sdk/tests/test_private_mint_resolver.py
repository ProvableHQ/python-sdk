"""``resolve_secret_nonce``: the one place every verb turns ``secret_nonce=None`` into a scalar.

An explicit nonce wins and touches no store. Otherwise the bridge derives for its own account —
reserving at execute, looking up (or re-deriving) at resume/complete — and refuses, naming both
fixes, when it cannot."""
import aleo
import pytest

from aleo_bridge import private_mint as pm
from aleo_bridge.errors import ConfigurationError
from tests.fakes.fake_bridge import FakeBridge

ACCOUNT = aleo.mainnet.Account.from_private_key(aleo.mainnet.PrivateKey.random())
ME = str(ACCOUNT.address)
OTHER = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
PROGRAM = pm.DEFAULT_WRAPPER_PROGRAM


def _bridge(store=True, account=True):
    b = FakeBridge()
    if account:
        b.aleo.default_account = ACCOUNT
    b.private_mint_identities = pm.MemoryPrivateMintIdentityStore() if store else None
    return b


def test_explicit_nonce_wins_and_touches_no_store():
    b = _bridge()
    assert pm.resolve_secret_nonce(b, recipient=ME, program=PROGRAM, secret_nonce="7scalar") == "7scalar"
    assert b.private_mint_identities.load() == []


def test_execute_reserves_and_later_phases_find_the_same_scalar():
    b = _bridge()
    nonce = pm.resolve_secret_nonce(b, recipient=ME, program=PROGRAM)
    commitment = pm.derive_address_commitment(ME, nonce, "mainnet")
    assert pm.resolve_secret_nonce(b, recipient=ME, program=PROGRAM, address_commitment=commitment) == nonce
    assert len(b.private_mint_identities.load()) == 1


def test_recovery_scans_when_the_store_is_empty():
    b = _bridge()
    nonce = pm.derive_secret_nonce(ACCOUNT.view_key, 5, "mainnet")
    commitment = pm.derive_address_commitment(ME, nonce, "mainnet")
    assert pm.resolve_secret_nonce(b, recipient=ME, program=PROGRAM, address_commitment=commitment) == nonce


@pytest.mark.parametrize("store,account,recipient", [(False, True, ME), (True, False, ME), (True, True, OTHER)])
def test_underivable_cases_fail_naming_both_fixes(store, account, recipient):
    b = _bridge(store=store, account=account)
    with pytest.raises(ConfigurationError, match="secret_nonce") as info:
        pm.resolve_secret_nonce(b, recipient=recipient, program=PROGRAM)
    assert "private_mint_identities" in str(info.value)
    assert pm.can_derive(b, recipient) is False


def test_quote_placeholder_only_when_nothing_can_derive():
    assert pm.resolve_secret_nonce(_bridge(store=False), recipient=ME, program=PROGRAM, placeholder=True) == "0scalar"
    b = _bridge()
    expected = pm.derive_secret_nonce(ACCOUNT.view_key, 0, "mainnet")
    assert pm.resolve_secret_nonce(b, recipient=ME, program=PROGRAM, placeholder=True) == expected
    assert b.private_mint_identities.load() == []


def test_stored_identity_for_another_recipient_is_refused():
    b = _bridge()
    identity = pm.reserve_identity(b.private_mint_identities, view_key=ACCOUNT.view_key, recipient=OTHER,
                                   network="mainnet")
    with pytest.raises(ConfigurationError, match="recipient"):
        pm.resolve_secret_nonce(b, recipient=ME, program=PROGRAM, address_commitment=identity.address_commitment)


def test_default_store_rides_with_a_file_checkpoint_store(tmp_path):
    from aleo_bridge.checkpoint import FileCheckpointStore
    store = pm.default_identity_store(FileCheckpointStore(tmp_path))
    assert isinstance(store, pm.FilePrivateMintIdentityStore)
    assert store.path == tmp_path / ".private-mint-identities.json"
    assert pm.default_identity_store(None) is None and pm.default_identity_store(object()) is None


def test_lookup_recognises_a_deposit_made_under_the_zero_scalar_default():
    """Tier-2 compatibility: ``eth.deposit_usdc`` still defaults to ``0scalar``; a later look-up for that
    commitment answers ``0scalar`` at once — with no store, no account, and without scanning."""
    zero = pm.derive_address_commitment(ME, "0scalar", "mainnet")
    assert pm.resolve_secret_nonce(_bridge(store=False, account=False), recipient=ME, program=PROGRAM,
                                   address_commitment=zero) == "0scalar"
    b = _bridge()
    assert pm.resolve_secret_nonce(b, recipient=ME, program=PROGRAM, address_commitment=zero) == "0scalar"
    assert b.private_mint_identities.load() == []
