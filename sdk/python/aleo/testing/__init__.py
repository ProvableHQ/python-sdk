"""Test-support utilities for the Aleo SDK.

Public helpers for spinning up a local `aleo-devnode`_ and finding records
against any node, so integration tests get eth-tester-style ergonomics:
deterministic pre-funded accounts, manual block production, and record
discovery without a hosted scanning service.

.. _aleo-devnode: https://github.com/ProvableHQ/aleo-devnode
"""
from __future__ import annotations

from .devnode import (
    Devnode as Devnode,
    DEVNODE_PRIVATE_KEY as DEVNODE_PRIVATE_KEY,
    DEFAULT_ACCOUNTS as DEFAULT_ACCOUNTS,
)
from .local_scanner import LocalRecordScanner as LocalRecordScanner

__all__ = [
    "Devnode",
    "DEVNODE_PRIVATE_KEY",
    "DEFAULT_ACCOUNTS",
    "LocalRecordScanner",
]
