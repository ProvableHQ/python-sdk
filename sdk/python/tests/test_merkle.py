# Copyright (C) 2019-2026 Provable Inc.
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

"""Tests for :class:`aleo.MerkleExclusionProof`.

The known answers come from deployed state rather than from the TypeScript
reference, so the port is pinned to what the chain actually verifies:

- ``EMPTY_TREE_ROOT`` is ``shield_swap_freezelist.aleo``'s initial
  ``freeze_list_root[1u8]``, read from testnet 2026-08-06.
- The address/field pairs are ``SealanceMerkleTree``'s own docstring examples
  (``ProvableHQ/sdk@mainnet:sdk/src/integrations/sealance/merkle-tree.ts``).
"""

import re

import pytest

from aleo import MerkleExclusionProof
from aleo.mainnet import Address, Plaintext, Poseidon4


#: ``Poseidon4([1field, 0field, 0field])`` — the root of a two-leaf empty tree.
EMPTY_TREE_ROOT = (
    "3642222252059314292809609689035560016959342421640560347114299934615987159853field"
)

ADDR_A = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
ADDR_B = "aleo1s3ws5tra87fjycnjrwsjcrnw2qxr8jfqqdugnf0xzqqw29q9m5pqem2u4t"
ZERO_ADDRESS = "aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc"

#: Nine freeze-list members in ascending field order.  Taking the first ``n``
#: gives member counts either side of a power of two, which is what exercises
#: the zero padding.
MEMBERS = [
    "aleo10p6fdm4054p50e7ykx5f4ec5hy0gt5pwnm0xyj8rr5vx6pva5cqsyx5k5y",
    "aleo13smp4u3ctasl5xnsrg9slcagsk384kvsk3enu9z0n3qsjcwjgcps4623cu",
    "aleo1z0c4vn2erd0908020slz5muqm459c5kkyle2xy4jc3n5xh2w6uzq508wr7",
    "aleo1zh8kzhr0eydshme7gjdj9ynf47pv6xg3al8mylx32yl4770t3vrqld2y2a",
    "aleo15ls2v3ga6cxx34ddeqsm5f0ksrvwjaj60e8pztdyxu2fl6lpfyyqu93feh",
    "aleo1wg0jwg4x9a2fmwhu2dkwdv2jd9j4vzmzp5gldjeyyzkmhj5lyy9q95m824",
    "aleo1l6qyu9jl34ged8he4sfglcfuxjpupyyulvkjt2qa2qwj6s93ag9snnc929",
    "aleo1fvx6rtmd6swdgwllgua6ygjjvej73nqj5uakzj9fe2sal8xe4cxsgr7lyq",
    "aleo1fr5nxtkyflk83zq3jxr3w6tsa7akaj0eslvy83suu6pwut03958sjwvkdx",
]
#: Falls between ``MEMBERS[1]`` and ``MEMBERS[2]`` — the bracketed case.
TARGET_INSIDE = "aleo178tq4f3qpcggwgt4l0xdashky3k2tun2lfm9lu6dt0y7h309fsps4hszf8"
#: Below every member — the below-first case.
TARGET_BELOW = "aleo1x9m7n0vx0rd8am8qdr2f5m8u2e2zwzmj53yeh5jszl57yp5lqqqqjm8wh3"
#: Above every member — the above-last case.
TARGET_ABOVE = "aleo1c9pnkwzja5m5dj93dg80gjyale8ep0n8zgrwl8cpt0r0cu874gfqyzrcgf"


@pytest.fixture
def merkle() -> MerkleExclusionProof:
    return MerkleExclusionProof()


# ── A independent reimplementation of the on-chain verifier ──────────────────
#
# Transcribed from ``verify_merkle_non_inclusion`` and
# ``calculate_merkle_root_and_depth`` in
# ``ProvableHQ/amm-v3@development:src/main.leo`` (b66d4f2), which matches the
# bytecode deployed as ``shield_swap.aleo``.  It hashes via its own Poseidon4
# calls rather than through the class under test, so it can disagree with the
# implementation — which is the point.


def _hash3(prefix: str, left: int, right: int) -> int:
    plaintext = Plaintext.from_string(f"[{prefix},{left}field,{right}field]")
    return int(str(Poseidon4().hash(plaintext.to_fields())).removesuffix("field"))


def parse_proof_literal(literal: str) -> list[tuple[list[int], int]]:
    """``[{siblings: [...], leaf_index: Nu32}, ...]`` back into Python."""
    parsed = []
    for block in re.findall(r"\{[^{}]*\}", literal):
        siblings = [int(s) for s in re.findall(r"(\d+)field", block)]
        leaf_index = int(re.search(r"leaf_index:\s*(\d+)u32", block).group(1))
        parsed.append((siblings, leaf_index))
    return parsed


def leo_root_and_depth(siblings: list[int], leaf_index: int,
                       max_depth: int = 15) -> tuple[int, int]:
    root = _hash3("1field", *(
        (siblings[0], siblings[1]) if leaf_index % 2 == 0
        else (siblings[1], siblings[0])))
    for i in range(2, max_depth + 1):
        if siblings[i] == 0:
            return root, i - 1
        pair = ((root, siblings[i]) if (leaf_index // 2 ** (i - 1)) % 2 == 0
                else (siblings[i], root))
        root = _hash3("0field", *pair)
    return root, max_depth


def leo_verify_non_inclusion(literal: str, address: str,
                             max_depth: int = 15) -> int:
    """Return the proven root, or raise ``AssertionError`` as the contract aborts."""
    proofs = parse_proof_literal(literal)
    assert len(proofs) == 2, "the verifier takes exactly two paths"
    (sib0, idx0), (sib1, idx1) = proofs

    root0, depth0 = leo_root_and_depth(sib0, idx0, max_depth)
    root1, depth1 = leo_root_and_depth(sib1, idx1, max_depth)
    assert root0 == root1, "the two paths must share a root"
    assert depth0 == depth1, "the two paths must share a depth"

    value = int(str(Address.from_string(address).to_field()).removesuffix("field"))
    last_leaf_index = 2 ** depth0 - 1
    if idx0 == idx1:
        if idx0 == 0:
            assert value < sib0[0], "below-first: value must precede leaf 0"
        else:
            assert idx0 == last_leaf_index, "above-last: must be the final leaf"
            assert value > sib0[0], "above-last: value must follow the final leaf"
    else:
        assert value > sib0[0], "bracketed: value must follow the left leaf"
        assert value < sib1[0], "bracketed: value must precede the right leaf"
        assert idx1 <= last_leaf_index, "bracketed: right leaf out of range"
        assert idx0 + 1 == idx1, "bracketed: leaves must be adjacent"
    return root0


def synthetic_tree(merkle: MerkleExclusionProof, num_leaves: int) -> list[int]:
    """A tree of *num_leaves* whose leaves are the fields ``10, 20, 30, ...``.

    Built directly from field literals rather than addresses so a test can
    place a target value precisely between two members.
    """
    leaves = [f"{(i + 1) * 10}field" for i in range(num_leaves)]
    return merkle.build_tree(leaves)


def test_empty_freezelist_root_matches_the_deployed_root(merkle):
    """An empty address list must reproduce the freezelist's on-chain root."""
    tree = merkle.build_tree(merkle.leaves_from_addresses([]))

    assert merkle.root(tree) == EMPTY_TREE_ROOT


@pytest.mark.parametrize("depth", range(1, 16))
def test_sibling_path_always_fills_the_proof_array(merkle, depth):
    """Every proof carries ``max_depth + 1`` siblings, whatever the tree depth.

    The reference implementation pads to ``depth`` instead, yielding 15 slots
    for every tree below the maximum and 16 only at depth 15.  The verifying
    struct is ``[field; 16]``, so a short path is rejected outright.

    The path's width depends only on the tree's size, so the sweep uses stand-in
    node values; :func:`test_sibling_path_of_a_hashed_tree_fills_the_array`
    covers a genuinely hashed tree.
    """
    num_leaves = 2 ** depth
    tree = list(range(1, 2 * num_leaves))

    path = merkle.sibling_path(tree, 0)

    assert len(path.siblings) == merkle.max_depth + 1


def test_sibling_path_of_a_hashed_tree_fills_the_array(merkle):
    """The width also holds for a real Poseidon-hashed tree."""
    tree = synthetic_tree(merkle, 8)

    path = merkle.sibling_path(tree, 3)

    assert len(path.siblings) == 16


#: Eight members fill a tree exactly, so there is no padding and leaf 0 is a
#: real address.  That is the only shape in which the verifier's below-first
#: case can arise.
UNPADDED = MEMBERS[:8]


def test_leaf_indices_bracket_an_absent_address(merkle):
    """An address inside the range brackets the two leaves either side of it."""
    tree = merkle.build_tree(merkle.leaves_from_addresses(UNPADDED))

    assert merkle.leaf_indices(tree, TARGET_INSIDE) == (1, 2)


def test_leaf_indices_below_every_member_collapse_to_the_first_leaf(merkle):
    """Below an unpadded list, both paths point at leaf 0."""
    tree = merkle.build_tree(merkle.leaves_from_addresses(UNPADDED))

    assert merkle.leaf_indices(tree, TARGET_BELOW) == (0, 0)


def test_leaf_indices_above_every_member_collapse_to_the_last_leaf(merkle):
    """Above the whole list, both paths point at the final leaf."""
    tree = merkle.build_tree(merkle.leaves_from_addresses(UNPADDED))

    assert merkle.leaf_indices(tree, TARGET_ABOVE) == (7, 7)


def test_leaf_indices_bracket_against_padding_below_the_first_member(merkle):
    """With padding, a below-everything address brackets the last zero leaf.

    Nine members pad to sixteen leaves, so leaf 6 is ``0field`` and leaf 7 is
    the smallest real address.  Every address sorts above zero, so this is the
    ordinary bracketed case rather than the below-first one.
    """
    tree = merkle.build_tree(merkle.leaves_from_addresses(MEMBERS))

    assert merkle.leaf_indices(tree, TARGET_BELOW) == (6, 7)


def test_leaf_indices_reject_an_address_that_is_on_the_list(merkle):
    """A member cannot be proven absent, so ask for it and get an error.

    The reference brackets with ``<=``, which hands back indices whose proof
    fails the verifier's strict inequality — but only after the caller has paid
    to prove and broadcast it.
    """
    tree = merkle.build_tree(merkle.leaves_from_addresses(MEMBERS))

    with pytest.raises(ValueError, match="on the list"):
        merkle.leaf_indices(tree, MEMBERS[2])


@pytest.mark.parametrize("target,case", [
    (TARGET_INSIDE, "bracketed"),
    (TARGET_BELOW, "below-first"),
    (TARGET_ABOVE, "above-last"),
])
def test_exclusion_proof_satisfies_the_contract_verifier(merkle, target, case):
    """Every generated proof is accepted by the transcribed on-chain verifier.

    This is what pins the port to the contract rather than to the TypeScript:
    the oracle hashes independently, so a wrong ordering, index bit, or domain
    separator shows up here.
    """
    tree = merkle.build_tree(merkle.leaves_from_addresses(UNPADDED))

    literal = merkle.exclusion_proof(tree, target)

    assert leo_verify_non_inclusion(literal, target) == tree[-1], case


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5, 7, 9])
@pytest.mark.parametrize("target", [TARGET_INSIDE, TARGET_BELOW, TARGET_ABOVE])
def test_exclusion_proof_survives_zero_padded_trees(merkle, count, target):
    """Member counts that are not powers of two pad with leading ``0field``."""
    tree = merkle.build_tree(merkle.leaves_from_addresses(MEMBERS[:count]))

    literal = merkle.exclusion_proof(tree, target)

    assert leo_verify_non_inclusion(literal, target) == tree[-1]


def test_exclusion_proof_literal_matches_the_struct(merkle):
    """The literal is two ``MerkleProof`` structs with full 16-slot arrays."""
    tree = merkle.build_tree(merkle.leaves_from_addresses(MEMBERS))

    literal = merkle.exclusion_proof(tree, TARGET_INSIDE)

    proofs = parse_proof_literal(literal)
    assert len(proofs) == 2
    assert all(len(siblings) == 16 for siblings, _ in proofs)
    assert literal.startswith("[{") and literal.endswith("}]")


def test_leaves_capacity_matches_the_depth(merkle):
    """A depth-``d`` tree holds ``2 ** d`` addresses, not ``2 ** (d - 1)``."""
    assert len(MerkleExclusionProof(max_depth=2).leaves_from_addresses(MEMBERS[:4])) == 4
    with pytest.raises(ValueError, match="caps at 2"):
        MerkleExclusionProof(max_depth=1).leaves_from_addresses(MEMBERS[:4])


def test_tree_from_nodes_reads_the_freeze_list_endpoint_shape(merkle):
    """The endpoint serves the whole tree as decimal strings, root last."""
    tree = merkle.tree_from_nodes(
        ["0", "0", EMPTY_TREE_ROOT.removesuffix("field")])

    assert merkle.root(tree) == EMPTY_TREE_ROOT


def test_tree_from_nodes_round_trips_a_built_tree(merkle):
    """A served tree proves exactly like one built locally from addresses."""
    built = merkle.build_tree(merkle.leaves_from_addresses(MEMBERS))

    served = merkle.tree_from_nodes([str(node) for node in built])

    assert served == built
    assert leo_verify_non_inclusion(
        merkle.exclusion_proof(served, TARGET_INSIDE), TARGET_INSIDE) == built[-1]


def test_zero_addresses_are_not_members(merkle):
    """The zero address is padding, so it never becomes a leaf of its own."""
    with_zero = merkle.leaves_from_addresses([ZERO_ADDRESS, *MEMBERS[:3]])

    assert with_zero == merkle.leaves_from_addresses(MEMBERS[:3])


def test_sibling_path_width_follows_a_custom_max_depth(merkle):
    """``max_depth`` is the single knob: the array is ``max_depth + 1`` wide."""
    shallow = MerkleExclusionProof(max_depth=10)
    tree = synthetic_tree(shallow, 8)

    assert len(shallow.sibling_path(tree, 0).siblings) == 11
    assert len(merkle.sibling_path(tree, 0, max_depth=4).siblings) == 5
