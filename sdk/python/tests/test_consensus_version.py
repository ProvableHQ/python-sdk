"""Consensus / Varuna version lookups against snarkvm's activation tables.

Heights are snarkVM 4.12.0's per-network ``CONSENSUS_VERSION_HEIGHTS``:
mainnet V21 (Varuna V3) at 22_437_000 and V22 at 22_962_000; testnet V21 at
20_234_000 and V22 at 20_415_000.

Invariant behind the no-height default: snarkvm parks *unscheduled* versions
at ``u32::MAX`` in the table, and ``CONSENSUS_VERSION(u32::MAX)`` matches
them, so the bindings resolve the default one block below that
(``u32::MAX - 1``) to land on the newest version with a real activation
height.  In 4.12.0 every version on both networks is scheduled, so the
default and the ``u32::MAX - 1`` lookup must agree; the next time snarkvm
parks a version at ``u32::MAX``, add a test here pinning the default BELOW
it (4.11.0 had testnet V21 parked and the default was 20 / Varuna 2).
"""

import pytest

import aleo.mainnet as mainnet

MAINNET_V21_HEIGHT = 22_437_000
MAINNET_V22_HEIGHT = 22_962_000
TESTNET_V21_HEIGHT = 20_234_000
TESTNET_V22_HEIGHT = 20_415_000

# The fallback height the bindings use for "no height given".
NEWEST_SCHEDULED_LOOKUP = 2**32 - 2


def test_mainnet_consensus_version_follows_table():
    assert mainnet.consensus_version(0) == 1
    assert mainnet.consensus_version(MAINNET_V21_HEIGHT - 1) == 20
    assert mainnet.consensus_version(MAINNET_V21_HEIGHT) == 21
    assert mainnet.consensus_version(MAINNET_V22_HEIGHT - 1) == 21
    assert mainnet.consensus_version(MAINNET_V22_HEIGHT) == 22


def test_mainnet_varuna_version_follows_consensus():
    assert mainnet.varuna_version(0) == 1
    assert mainnet.varuna_version(MAINNET_V21_HEIGHT - 1) == 2
    assert mainnet.varuna_version(MAINNET_V21_HEIGHT) == 3
    # V22 changes program limits and opcode costs, not the proof system.
    assert mainnet.varuna_version(MAINNET_V22_HEIGHT - 1) == 3
    assert mainnet.varuna_version(MAINNET_V22_HEIGHT) == 3


def test_mainnet_default_is_newest_scheduled_version():
    # No height → the newest version with a real activation height, never
    # one snarkvm has parked at u32::MAX (see the module docstring).
    assert mainnet.consensus_version() == 22
    assert mainnet.varuna_version() == 3
    assert mainnet.varuna_version(None) == mainnet.varuna_version()
    assert mainnet.consensus_version() == mainnet.consensus_version(
        NEWEST_SCHEDULED_LOOKUP
    )


def test_testnet_consensus_version_follows_table():
    testnet = pytest.importorskip("aleo.testnet")
    assert testnet.consensus_version(0) == 1
    assert testnet.consensus_version(TESTNET_V21_HEIGHT - 1) == 20
    assert testnet.consensus_version(TESTNET_V21_HEIGHT) == 21
    assert testnet.consensus_version(TESTNET_V22_HEIGHT - 1) == 21
    assert testnet.consensus_version(TESTNET_V22_HEIGHT) == 22


def test_testnet_varuna_version_follows_consensus():
    testnet = pytest.importorskip("aleo.testnet")
    assert testnet.varuna_version(0) == 1
    assert testnet.varuna_version(TESTNET_V21_HEIGHT - 1) == 2
    assert testnet.varuna_version(TESTNET_V21_HEIGHT) == 3
    assert testnet.varuna_version(TESTNET_V22_HEIGHT - 1) == 3
    assert testnet.varuna_version(TESTNET_V22_HEIGHT) == 3


def test_testnet_default_is_newest_scheduled_version():
    testnet = pytest.importorskip("aleo.testnet")
    assert testnet.consensus_version() == 22
    assert testnet.varuna_version() == 3
    assert testnet.varuna_version(None) == testnet.varuna_version()
    assert testnet.consensus_version() == testnet.consensus_version(
        NEWEST_SCHEDULED_LOOKUP
    )
