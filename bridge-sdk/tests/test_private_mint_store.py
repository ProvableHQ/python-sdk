"""The private mint identity store: counters never recycle, files are 0600 and atomic, a second
instance (or process) on the same path continues the sequence, and a lost file is recovered by
re-deriving counters from zero."""
import json
import stat
import threading

import pytest

from aleo_bridge import private_mint as pm
from aleo_bridge.errors import ConfigurationError

VIEW_KEY = "AViewKey1sqm952gJj1tmAWySYDQvSv2NmfnyEMvU6a9ZBCuyG7PN"
RECIPIENT = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
KW = dict(view_key=VIEW_KEY, recipient=RECIPIENT, network="mainnet")


def _file_store(tmp_path):
    return pm.FilePrivateMintIdentityStore(tmp_path / ".private-mint-identities.json")


def test_reservation_is_monotonic_and_saved(tmp_path):
    store = _file_store(tmp_path)
    first, second = pm.reserve_identity(store, **KW), pm.reserve_identity(store, **KW)
    assert (first.counter, second.counter) == (0, 1)
    assert first.secret_nonce == pm.derive_secret_nonce(VIEW_KEY, 0, "mainnet")
    assert [i.address_commitment for i in store.load()] == [first.address_commitment, second.address_commitment]
    assert pm.find_identity(store, second.address_commitment) == second
    assert pm.find_identity(store, "00" * 32) is None


def test_file_store_mode_and_companions(tmp_path):
    store = _file_store(tmp_path)
    pm.reserve_identity(store, **KW)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.counter_path.stat().st_mode) == 0o600
    assert store.counter_path.read_text().strip() == "1"
    data = json.loads(store.path.read_text())
    assert data["identities"][0]["secretNonce"].endswith("scalar")
    assert not any(p.name.endswith(".tmp") for p in tmp_path.iterdir())


def test_counter_survives_losing_the_identities_file(tmp_path):
    store = _file_store(tmp_path)
    pm.reserve_identity(store, **KW)
    pm.reserve_identity(store, **KW)
    store.path.unlink()
    assert pm.reserve_identity(store, **KW).counter == 2


def test_counter_is_consumed_before_a_failed_derivation(tmp_path):
    store = _file_store(tmp_path)
    with pytest.raises(ConfigurationError):
        pm.reserve_identity(store, view_key="AViewKey1nope", recipient=RECIPIENT, network="mainnet")
    assert pm.reserve_identity(store, **KW).counter == 1


def test_a_second_instance_on_the_same_path_continues_the_sequence(tmp_path):
    a, b = _file_store(tmp_path), _file_store(tmp_path)
    pm.reserve_identity(a, **KW)
    assert pm.reserve_identity(b, **KW).counter == 1
    assert len(a.load()) == 2


def test_concurrent_reservations_get_distinct_counters(tmp_path):
    store = _file_store(tmp_path)
    out: list[pm.PrivateMintIdentity] = []
    threads = [threading.Thread(target=lambda: out.append(pm.reserve_identity(store, **KW))) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(i.counter for i in out) == list(range(8)) and len(store.load()) == 8


def test_peek_does_not_consume(tmp_path):
    store = _file_store(tmp_path)
    assert pm.peek_secret_nonce(store, **KW) == pm.derive_secret_nonce(VIEW_KEY, 0, "mainnet")
    assert pm.reserve_identity(store, **KW).counter == 0


def test_recover_rederives_after_the_file_is_lost(tmp_path):
    store = _file_store(tmp_path)
    lost = pm.reserve_identity(store, **KW)
    pm.reserve_identity(store, **KW)
    store.path.unlink()
    store.counter_path.unlink()
    found = pm.recover_identity(store, address_commitment=lost.address_commitment, **KW)
    assert found == lost and pm.find_identity(store, lost.address_commitment) == lost


def test_recover_stops_at_the_scan_limit_with_an_actionable_error(tmp_path):
    store = _file_store(tmp_path)
    with pytest.raises(ConfigurationError, match=r"3 counters.*secret_nonce"):
        pm.recover_identity(store, address_commitment="ab" * 32, scan_limit=3, **KW)


def test_recover_rejects_a_stored_identity_for_another_recipient(tmp_path):
    store = _file_store(tmp_path)
    other = "aleo132x69f77mz7cx2f4s5wykktj73tj8cy7smdvxrgwja9qcmyuysxst42d9m"
    identity = pm.reserve_identity(store, view_key=VIEW_KEY, recipient=other, network="mainnet")
    with pytest.raises(ConfigurationError, match="recipient"):
        pm.recover_identity(store, address_commitment=identity.address_commitment, **KW)


@pytest.mark.parametrize("corrupt", ["not json", '{"identities": [{"counter": -1}]}'])
def test_malformed_identities_file_raises_instead_of_resetting(tmp_path, corrupt):
    store = _file_store(tmp_path)
    store.path.write_text(corrupt)
    with pytest.raises(ConfigurationError, match="restore"):
        pm.reserve_identity(store, **KW)


def test_malformed_counter_file_raises(tmp_path):
    store = _file_store(tmp_path)
    store.counter_path.write_text("x")
    with pytest.raises(ConfigurationError, match="restore"):
        pm.reserve_identity(store, **KW)


def test_counter_space_exhaustion(tmp_path):
    store = _file_store(tmp_path)
    store.counter_path.write_text(str(pm.MAX_COUNTER + 1))
    with pytest.raises(ConfigurationError, match="exhausted"):
        pm.reserve_identity(store, **KW)


def test_memory_store_is_explicit_only():
    store = pm.MemoryPrivateMintIdentityStore()
    assert pm.reserve_identity(store, **KW).counter == 0 and pm.reserve_identity(store, **KW).counter == 1


def test_recovery_scan_is_fast_enough_to_use_the_default_limit(tmp_path):
    """Hoisted invariants: 2,000 counters in well under the time one would take re-parsing everything."""
    import time
    store = _file_store(tmp_path)
    started = time.perf_counter()
    with pytest.raises(ConfigurationError, match="2000 counters"):
        pm.recover_identity(store, address_commitment="ab" * 32, scan_limit=2000, **KW)
    assert time.perf_counter() - started < 5.0
