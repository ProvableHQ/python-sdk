"""Test-support utilities for the Aleo SDK.

Public helpers for spinning up a local `aleo-devnode`_ — deterministic
pre-funded accounts and manual block production — so integration tests get
eth-tester-style ergonomics.

.. _aleo-devnode: https://github.com/ProvableHQ/aleo-devnode
"""
from __future__ import annotations

from .devnode import (
    Devnode as Devnode,
    DEVNODE_PRIVATE_KEY as DEVNODE_PRIVATE_KEY,
    DEFAULT_ACCOUNTS as DEFAULT_ACCOUNTS,
)

__all__ = [
    "Devnode",
    "DEVNODE_PRIVATE_KEY",
    "DEFAULT_ACCOUNTS",
]
