"""Consensus / Varuna version lookups against snarkvm's activation tables.

Heights are snarkVM 4.11.0's per-network ``CONSENSUS_VERSION_HEIGHTS``:
mainnet V21 (Varuna V3) at 22_437_000; testnet has not scheduled V21.
"""

import pytest

import aleo.mainnet as mainnet

MAINNET_V21_HEIGHT = 22_437_000


def test_mainnet_consensus_version_follows_table():
    assert mainnet.consensus_version(0) == 1
    assert mainnet.consensus_version(MAINNET_V21_HEIGHT - 1) == 20
    assert mainnet.consensus_version(MAINNET_V21_HEIGHT) == 21


def test_mainnet_varuna_version_follows_consensus():
    assert mainnet.varuna_version(0) == 1
    assert mainnet.varuna_version(MAINNET_V21_HEIGHT - 1) == 2
    assert mainnet.varuna_version(MAINNET_V21_HEIGHT) == 3


def test_mainnet_default_is_newest_scheduled_version():
    # No height → the newest version with a real activation height, never
    # one snarkvm has parked at u32::MAX.
    assert mainnet.consensus_version() == 21
    assert mainnet.varuna_version() == 3
    assert mainnet.varuna_version(None) == mainnet.varuna_version()


def test_testnet_default_stays_on_varuna_v2():
    testnet = pytest.importorskip("aleo.testnet")
    assert testnet.consensus_version() == 20
    assert testnet.varuna_version() == 2
    assert testnet.varuna_version(0) == 1
