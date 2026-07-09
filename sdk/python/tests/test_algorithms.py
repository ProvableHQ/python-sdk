# Copyright (C) 2019-2023 Aleo Systems Inc.
# This file is part of the Aleo SDK library.

# The Aleo SDK library is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# The Aleo SDK library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with the Aleo SDK library. If not, see <https://www.gnu.org/licenses/>.

"""
Known-answer tests for hash algorithm classes (BHP, Pedersen, Poseidon).

Source of KAT vectors:
  github.com/ProvableHQ/sdk@543b41e0a7e3d6a8a49ae8d0870809dc92b88684
  sdk/tests/data/algebra.ts  +  sdk/tests/algorithm.test.ts
"""

import json
import os
import pytest

from aleo.mainnet import (
    BHP256, BHP512, BHP768, BHP1024,
    Field,
    Network,
    Pedersen64, Pedersen128,
    Poseidon2, Poseidon4, Poseidon8,
    Scalar,
)

# ---------------------------------------------------------------------------
# Load KAT vectors
# ---------------------------------------------------------------------------

_VECTORS_PATH = os.path.join(os.path.dirname(__file__), "vectors", "algorithms.json")
with open(_VECTORS_PATH) as _f:
    _V = json.load(_f)

# ---------------------------------------------------------------------------
# Shared test inputs (mirrors algorithm.test.ts)
# ---------------------------------------------------------------------------

# Field generator Fg and powers Fg², Fg³, Fg⁴
_Fg = Field.from_string(_V["field_generator"])
_F2 = _Fg * _Fg
_F3 = _F2 * _Fg
_F4 = _F3 * _Fg
_SFg = Scalar.from_string(_V["scalar_generator"])

# BHP / Pedersen input: concatenated to_bits_le of [Fg, F2, F3, F4]
_FIELD_BITS: list[bool] = (
    _Fg.to_bits_le()
    + _F2.to_bits_le()
    + _F3.to_bits_le()
    + _F4.to_bits_le()
)

# Poseidon input: list of field elements [Fg, F2, F3, F4]
_FIELD_ARRAY: list[Field] = [_Fg, _F2, _F3, _F4]

# Pedersen input: two u32(1) LE → [T, F×31, T, F×31] (64 bits total)
_PEDERSEN_BITS: list[bool] = (
    [True] + [False] * 31   # u32(1) LE
    + [True] + [False] * 31  # u32(1) LE
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _str(obj) -> str:
    return str(obj)


# ---------------------------------------------------------------------------
# BHP256
# ---------------------------------------------------------------------------

class TestBHP256:
    def setup_method(self):
        self.hasher = BHP256()

    def test_hash(self):
        result = self.hasher.hash(_FIELD_BITS)
        assert _str(result) == _V["bhp256"]["hash"]

    def test_hash_to_group(self):
        result = self.hasher.hash_to_group(_FIELD_BITS)
        assert _str(result) == _V["bhp256"]["hash_to_group"]

    def test_commit(self):
        result = self.hasher.commit(_FIELD_BITS, _SFg)
        assert _str(result) == _V["bhp256"]["commit"]

    def test_commit_to_group(self):
        result = self.hasher.commit_to_group(_FIELD_BITS, _SFg)
        assert _str(result) == _V["bhp256"]["commit_to_group"]

    def test_custom_domain_differs(self):
        custom = BHP256.setup("custom_domain_xyz")
        default_hash = _str(self.hasher.hash(_FIELD_BITS))
        custom_hash = _str(custom.hash(_FIELD_BITS))
        assert default_hash != custom_hash


# ---------------------------------------------------------------------------
# BHP512
# ---------------------------------------------------------------------------

class TestBHP512:
    def setup_method(self):
        self.hasher = BHP512()

    def test_hash(self):
        result = self.hasher.hash(_FIELD_BITS)
        assert _str(result) == _V["bhp512"]["hash"]

    def test_hash_to_group(self):
        result = self.hasher.hash_to_group(_FIELD_BITS)
        assert _str(result) == _V["bhp512"]["hash_to_group"]

    def test_commit(self):
        result = self.hasher.commit(_FIELD_BITS, _SFg)
        assert _str(result) == _V["bhp512"]["commit"]

    def test_commit_to_group(self):
        result = self.hasher.commit_to_group(_FIELD_BITS, _SFg)
        assert _str(result) == _V["bhp512"]["commit_to_group"]

    def test_custom_domain_differs(self):
        custom = BHP512.setup("custom_domain_xyz")
        assert _str(self.hasher.hash(_FIELD_BITS)) != _str(custom.hash(_FIELD_BITS))


# ---------------------------------------------------------------------------
# BHP768
# ---------------------------------------------------------------------------

class TestBHP768:
    def setup_method(self):
        self.hasher = BHP768()

    def test_hash(self):
        result = self.hasher.hash(_FIELD_BITS)
        assert _str(result) == _V["bhp768"]["hash"]

    def test_hash_to_group(self):
        result = self.hasher.hash_to_group(_FIELD_BITS)
        assert _str(result) == _V["bhp768"]["hash_to_group"]

    def test_commit(self):
        result = self.hasher.commit(_FIELD_BITS, _SFg)
        assert _str(result) == _V["bhp768"]["commit"]

    def test_commit_to_group(self):
        result = self.hasher.commit_to_group(_FIELD_BITS, _SFg)
        assert _str(result) == _V["bhp768"]["commit_to_group"]

    def test_custom_domain_differs(self):
        custom = BHP768.setup("custom_domain_xyz")
        assert _str(self.hasher.hash(_FIELD_BITS)) != _str(custom.hash(_FIELD_BITS))


# ---------------------------------------------------------------------------
# BHP1024
# ---------------------------------------------------------------------------

class TestBHP1024:
    def setup_method(self):
        self.hasher = BHP1024()

    def test_hash(self):
        result = self.hasher.hash(_FIELD_BITS)
        assert _str(result) == _V["bhp1024"]["hash"]

    def test_hash_to_group(self):
        result = self.hasher.hash_to_group(_FIELD_BITS)
        assert _str(result) == _V["bhp1024"]["hash_to_group"]

    def test_commit(self):
        result = self.hasher.commit(_FIELD_BITS, _SFg)
        assert _str(result) == _V["bhp1024"]["commit"]

    def test_commit_to_group(self):
        result = self.hasher.commit_to_group(_FIELD_BITS, _SFg)
        assert _str(result) == _V["bhp1024"]["commit_to_group"]

    def test_custom_domain_differs(self):
        custom = BHP1024.setup("custom_domain_xyz")
        assert _str(self.hasher.hash(_FIELD_BITS)) != _str(custom.hash(_FIELD_BITS))


# ---------------------------------------------------------------------------
# Pedersen64
# ---------------------------------------------------------------------------

class TestPedersen64:
    def setup_method(self):
        self.hasher = Pedersen64()

    def test_hash(self):
        result = self.hasher.hash(_PEDERSEN_BITS)
        assert _str(result) == _V["pedersen64"]["hash"]

    def test_commit(self):
        result = self.hasher.commit(_PEDERSEN_BITS, _SFg)
        assert _str(result) == _V["pedersen64"]["commit"]

    def test_commit_to_group(self):
        result = self.hasher.commit_to_group(_PEDERSEN_BITS, _SFg)
        assert _str(result) == _V["pedersen64"]["commit_to_group"]

    def test_custom_domain_differs(self):
        custom = Pedersen64.setup("custom_domain_xyz")
        assert _str(self.hasher.hash(_PEDERSEN_BITS)) != _str(custom.hash(_PEDERSEN_BITS))


# ---------------------------------------------------------------------------
# Pedersen128
# ---------------------------------------------------------------------------

class TestPedersen128:
    def setup_method(self):
        self.hasher = Pedersen128()

    def test_hash(self):
        result = self.hasher.hash(_PEDERSEN_BITS)
        assert _str(result) == _V["pedersen128"]["hash"]

    def test_commit(self):
        result = self.hasher.commit(_PEDERSEN_BITS, _SFg)
        assert _str(result) == _V["pedersen128"]["commit"]

    def test_commit_to_group(self):
        result = self.hasher.commit_to_group(_PEDERSEN_BITS, _SFg)
        assert _str(result) == _V["pedersen128"]["commit_to_group"]

    def test_custom_domain_differs(self):
        custom = Pedersen128.setup("custom_domain_xyz")
        assert _str(self.hasher.hash(_PEDERSEN_BITS)) != _str(custom.hash(_PEDERSEN_BITS))


# ---------------------------------------------------------------------------
# Poseidon2
# ---------------------------------------------------------------------------

class TestPoseidon2:
    def setup_method(self):
        self.hasher = Poseidon2()

    def test_hash(self):
        result = self.hasher.hash(list(_FIELD_ARRAY))
        assert _str(result) == _V["poseidon2"]["hash"]

    def test_hash_to_scalar(self):
        result = self.hasher.hash_to_scalar(list(_FIELD_ARRAY))
        assert _str(result) == _V["poseidon2"]["hash_to_scalar"]

    def test_hash_to_group(self):
        result = self.hasher.hash_to_group(list(_FIELD_ARRAY))
        assert _str(result) == _V["poseidon2"]["hash_to_group"]

    def test_hash_many(self):
        results = self.hasher.hash_many(list(_FIELD_ARRAY), 2)
        assert [_str(r) for r in results] == _V["poseidon2"]["hash_many_2"]

    def test_custom_domain_differs(self):
        custom = Poseidon2.setup("custom_domain_xyz")
        assert _str(self.hasher.hash(list(_FIELD_ARRAY))) != _str(custom.hash(list(_FIELD_ARRAY)))

    def test_equals_network_hash_psd2(self):
        """Poseidon2.hash must match Network.hash_psd2 (cross-check)."""
        expected = Network.hash_psd2(_FIELD_ARRAY)
        result = self.hasher.hash(list(_FIELD_ARRAY))
        assert _str(result) == _str(expected)


# ---------------------------------------------------------------------------
# Poseidon4
# ---------------------------------------------------------------------------

class TestPoseidon4:
    def setup_method(self):
        self.hasher = Poseidon4()

    def test_hash(self):
        result = self.hasher.hash(list(_FIELD_ARRAY))
        assert _str(result) == _V["poseidon4"]["hash"]

    def test_hash_to_scalar(self):
        result = self.hasher.hash_to_scalar(list(_FIELD_ARRAY))
        assert _str(result) == _V["poseidon4"]["hash_to_scalar"]

    def test_hash_to_group(self):
        result = self.hasher.hash_to_group(list(_FIELD_ARRAY))
        assert _str(result) == _V["poseidon4"]["hash_to_group"]

    def test_hash_many(self):
        results = self.hasher.hash_many(list(_FIELD_ARRAY), 2)
        assert [_str(r) for r in results] == _V["poseidon4"]["hash_many_2"]

    def test_custom_domain_differs(self):
        custom = Poseidon4.setup("custom_domain_xyz")
        assert _str(self.hasher.hash(list(_FIELD_ARRAY))) != _str(custom.hash(list(_FIELD_ARRAY)))


# ---------------------------------------------------------------------------
# Poseidon8
# ---------------------------------------------------------------------------

class TestPoseidon8:
    def setup_method(self):
        self.hasher = Poseidon8()

    def test_hash(self):
        result = self.hasher.hash(list(_FIELD_ARRAY))
        assert _str(result) == _V["poseidon8"]["hash"]

    def test_hash_to_scalar(self):
        result = self.hasher.hash_to_scalar(list(_FIELD_ARRAY))
        assert _str(result) == _V["poseidon8"]["hash_to_scalar"]

    def test_hash_to_group(self):
        result = self.hasher.hash_to_group(list(_FIELD_ARRAY))
        assert _str(result) == _V["poseidon8"]["hash_to_group"]

    def test_hash_many(self):
        results = self.hasher.hash_many(list(_FIELD_ARRAY), 2)
        assert [_str(r) for r in results] == _V["poseidon8"]["hash_many_2"]

    def test_custom_domain_differs(self):
        custom = Poseidon8.setup("custom_domain_xyz")
        assert _str(self.hasher.hash(list(_FIELD_ARRAY))) != _str(custom.hash(list(_FIELD_ARRAY)))
