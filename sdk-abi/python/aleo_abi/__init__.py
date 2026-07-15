# Copyright (C) 2019-2026 Provable Inc.
# SPDX-License-Identifier: GPL-3.0-or-later
"""ABI generation and compatibility checking for Aleo programs."""

from ._aleo_abi import generate_abi, check_compatibility

__all__ = ["generate_abi", "check_compatibility"]
