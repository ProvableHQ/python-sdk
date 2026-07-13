"""Devnode tier — deliberately skipped.

Hermetic write tests via the facade's Devnode utility are blocked: the
devnode emits pre-V8 record versions (consensus-version skew), so
record-spending shield-swap verbs cannot run against it until the
Rust-side consensus-version fix lands.  This module keeps the gap visible
in test output rather than silent.
"""
import pytest

pytest.skip("devnode record-version skew — record-spending verbs blocked; "
            "see the facade's devnode docs", allow_module_level=True)
