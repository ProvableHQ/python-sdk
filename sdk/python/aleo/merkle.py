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

"""Merkle exclusion (non-inclusion) proofs over sorted address trees.

Programs following the Sealance compliance architecture keep a freeze list as a
sorted Merkle tree of addresses and require callers to prove their address is
*absent* from it.  This module builds those proofs client-side.

It is program-agnostic: it knows about sorted address trees and nothing about
any particular freeze list, so it serves ``shield_swap_freezelist.aleo``, the
compliance stablecoins, and anything else built the same way.

Ported from the wasm SDK's ``SealanceMerkleTree``
(``ProvableHQ/sdk@mainnet:sdk/src/integrations/sealance/merkle-tree.ts``).
"""
from __future__ import annotations

from typing import Any, NamedTuple, Sequence

from ._client_common import DEFAULT_NETWORK

__all__ = ["MerkleExclusionProof", "SiblingPath"]


class SiblingPath(NamedTuple):
    """One authentication path, as the verifier's ``MerkleProof`` struct.

    ``siblings[0]`` is the leaf itself and ``siblings[1]`` its leaf-layer
    sibling; the rest are the sibling at each level above, zero-padded to fill
    the struct's fixed-size array.
    """

    siblings: list[int]
    leaf_index: int

#: The all-zero address, which the tree carries as padding rather than as a
#: member.  Verifiers compare against it, so it is never a real leaf.
ZERO_ADDRESS = "aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc"

#: Domain separators.  Leaf-layer pairs hash under ``1field``, every layer above
#: under ``0field``, which is what keeps a leaf from being replayed as a node.
_LEAF_PREFIX = "1field"
_NODE_PREFIX = "0field"


class MerkleExclusionProof:
    """Builds Merkle non-inclusion proofs against a sorted address tree.

    Args:
        max_depth: Maximum tree depth, matching the verifying program's
            ``MAX_MERKLE_TREE_DEPTH``.  This is the single knob the rest of the
            shape derives from: the tree holds up to ``2 ** max_depth`` leaves
            and a proof carries ``max_depth + 1`` siblings (one leaf slot plus
            one per level).  Defaults to 15, the shield_swap value.
        network: Which extension module supplies ``Poseidon4`` and ``Address``.
    """

    def __init__(self, *, max_depth: int = 15,
                 network: str = DEFAULT_NETWORK) -> None:
        if max_depth < 1:
            raise ValueError(f"max_depth must be at least 1, got {max_depth}")
        self.max_depth = max_depth
        self._network = network
        # Built on first use and reused: a full-depth tree is 65_535 hashes,
        # and Poseidon setup is not free.
        self._hasher: Any = None

    def __repr__(self) -> str:
        return (f"MerkleExclusionProof(max_depth={self.max_depth}, "
                f"network={self._network!r})")

    def _net(self) -> Any:
        """The network extension module supplying the hash and address types."""
        try:
            if self._network == "testnet":
                from . import testnet as _mod  # type: ignore[attr-defined]
            else:
                from . import mainnet as _mod  # type: ignore[attr-defined]
        except ImportError:
            raise ImportError(
                f"aleo {self._network} module not available"
            ) from None
        return _mod

    # ── Tree construction ────────────────────────────────────────────────

    def leaves_from_addresses(self, addresses: Sequence[str], *,
                              max_depth: int | None = None) -> list[str]:
        """Addresses as sorted, zero-padded leaf literals.

        Zero addresses are dropped, the rest are converted to fields and sorted
        ascending, then ``0field`` padding is *prepended* to reach a power of
        two.  Prepending is what keeps the greatest real address at the right
        boundary, which is the invariant the above-last case relies on.

        Args:
            addresses: The member addresses, in any order.
            max_depth: Overrides the instance's depth for this call.

        Returns:
            ``2 ** k`` field literals, ordered padding-first then ascending.

        Raises:
            ValueError: If there are more addresses than ``2 ** max_depth``.
        """
        depth = self.max_depth if max_depth is None else max_depth
        capacity = 2 ** depth

        members = [a for a in addresses if a != ZERO_ADDRESS]
        if len(members) > capacity:
            raise ValueError(
                f"freeze list holds {len(members)} addresses, but a depth-"
                f"{depth} tree caps at {capacity}"
            )

        num_leaves = 2 if len(members) <= 1 else 1 << (len(members) - 1).bit_length()

        fields = sorted(self.address_to_field(a) for a in members)
        padding = ["0field"] * (num_leaves - len(fields))
        return padding + [f"{f}field" for f in fields]

    def build_tree(self, leaves: Sequence[str]) -> list[int]:
        """Hash *leaves* bottom-up into a flat tree array.

        Args:
            leaves: An even number of field literals, as
                :meth:`leaves_from_addresses` returns.

        Returns:
            Every node as an int, level by level: leaves first, then each layer
            above, ending with the root.

        Raises:
            ValueError: If *leaves* is empty or has an odd length.
        """
        if not leaves:
            raise ValueError("leaves cannot be empty")
        if len(leaves) % 2 != 0:
            raise ValueError(
                f"leaves must have an even length, got {len(leaves)}")

        current = list(leaves)
        tree = list(current)
        while len(current) > 1:
            # The leaf layer is the only one whose width equals the input's.
            prefix = _LEAF_PREFIX if len(current) == len(leaves) else _NODE_PREFIX
            current = [
                self._hash_pair(prefix, current[i], current[i + 1])
                for i in range(0, len(current), 2)
            ]
            tree.extend(current)
        return [int(node.removesuffix("field")) for node in tree]

    def tree_from_nodes(self, nodes: Sequence[str]) -> list[int]:
        """A pre-built tree from its serialized nodes.

        Freeze-list services publish the whole tree — leaves, then each layer
        above, root last — as decimal strings, so the common path needs no
        hashing at all.

        Args:
            nodes: Every node as a decimal string, in tree order.  A ``field``
                suffix is tolerated.

        Returns:
            The same nodes as ints, ready for :meth:`exclusion_proof`.

        Raises:
            ValueError: If *nodes* is empty, is not a whole tree
                (``2 * leaves - 1`` nodes), or holds a non-numeric entry.
        """
        if not nodes:
            raise ValueError("a tree needs at least one node")
        try:
            tree = [int(str(node).strip().removesuffix("field")) for node in nodes]
        except ValueError as exc:
            raise ValueError(f"tree holds a non-numeric node: {exc}") from None

        # A complete binary tree over 2**k leaves has 2**(k+1) - 1 nodes.
        if (len(tree) + 1) & len(tree):
            raise ValueError(
                f"{len(tree)} nodes is not a complete tree — expected "
                f"2 * leaves - 1"
            )
        return tree

    def root(self, tree: Sequence[int]) -> str:
        """The tree's root as a field literal."""
        if not tree:
            raise ValueError("tree cannot be empty")
        return f"{tree[-1]}field"

    # ── Proofs ───────────────────────────────────────────────────────────

    def leaf_indices(self, tree: Sequence[int], address: str) -> tuple[int, int]:
        """The two leaves whose paths together prove *address* is absent.

        Args:
            tree: A flat tree as :meth:`build_tree` returns.
            address: The address to exclude.

        Returns:
            ``(left, right)``.  Normally these bracket the address as adjacent
            leaves.  They collapse to a single index at the boundaries: both
            ``0`` when the address sorts below every leaf, and both the last
            index when it sorts above every leaf — the two special cases the
            verifier checks separately.

        Raises:
            ValueError: If *address* is a member.  A member cannot be proven
                absent, and returning indices anyway would produce a proof that
                fails only after the caller has paid to prove and broadcast it.
        """
        num_leaves = (len(tree) + 1) // 2
        leaves = list(tree[:num_leaves])
        value = self.address_to_field(address)

        if value in leaves:
            raise ValueError(
                f"{address} is on the list (leaf {leaves.index(value)}), so it "
                f"cannot be proven excluded"
            )

        right = next((i for i, leaf in enumerate(leaves) if value <= leaf), -1)
        if right == -1:
            return num_leaves - 1, num_leaves - 1
        if right == 0:
            return 0, 0
        return right - 1, right

    def sibling_path(self, tree: Sequence[int], leaf_index: int, *,
                     max_depth: int | None = None) -> SiblingPath:
        """The authentication path for *leaf_index*.

        Args:
            tree: A flat tree as :meth:`build_tree` returns.
            leaf_index: Which leaf to authenticate.
            max_depth: Overrides the instance's depth for this call.

        Returns:
            A :class:`SiblingPath` holding exactly ``max_depth + 1`` siblings.
            The array is padded to its full width rather than to the tree's own
            depth: the verifier reads a fixed-size array and treats the trailing
            zeros as "no more levels", so a short path fails to typecheck and a
            long one cannot be represented.

        Raises:
            IndexError: If *leaf_index* is outside the tree's leaf layer.
            ValueError: If the tree is deeper than ``max_depth`` allows.
        """
        depth = self.max_depth if max_depth is None else max_depth
        width = depth + 1

        num_leaves = (len(tree) + 1) // 2
        if not 0 <= leaf_index < num_leaves:
            raise IndexError(
                f"leaf_index {leaf_index} is outside a tree of {num_leaves} "
                f"leaves"
            )

        siblings = [tree[leaf_index]]
        index = leaf_index
        parent_index = num_leaves
        level = 1
        while parent_index < len(tree):
            sibling = index + 1 if index % 2 == 0 else index - 1
            siblings.append(tree[sibling])
            index = parent_index + leaf_index // (2 ** level)
            parent_index += num_leaves // (2 ** level)
            level += 1

        if len(siblings) > width:
            raise ValueError(
                f"tree of {num_leaves} leaves needs {len(siblings)} proof "
                f"slots, but max_depth={depth} allows {width}"
            )
        siblings.extend([0] * (width - len(siblings)))
        return SiblingPath(siblings, leaf_index)

    def format_proof(self, paths: Sequence[SiblingPath]) -> str:
        """Authentication paths as the Aleo array-of-struct literal."""
        structs = ", ".join(
            "{siblings: [" + ", ".join(f"{s}field" for s in path.siblings)
            + f"], leaf_index: {path.leaf_index}u32}}"
            for path in paths
        )
        return f"[{structs}]"

    def exclusion_proof(self, tree: Sequence[int], address: str, *,
                        max_depth: int | None = None) -> str:
        """A ready-to-pass ``[MerkleProof; 2]`` literal proving *address* is absent.

        Args:
            tree: A flat tree as :meth:`build_tree` or :meth:`tree_from_nodes`
                returns.
            address: The address to prove absent.
            max_depth: Overrides the instance's depth for this call.

        Returns:
            The literal to pass wherever the program takes ``[MerkleProof; 2]``.

        Raises:
            ValueError: If *address* is on the list.
        """
        left, right = self.leaf_indices(tree, address)
        return self.format_proof([
            self.sibling_path(tree, left, max_depth=max_depth),
            self.sibling_path(tree, right, max_depth=max_depth),
        ])

    # ── Primitives ───────────────────────────────────────────────────────

    def address_to_field(self, address: str) -> int:
        """An Aleo address as its field element, matching Leo's ``as field``."""
        net = self._net()
        return int(str(net.Address.from_string(address).to_field())
                   .removesuffix("field"))

    def _hash_pair(self, prefix: str, left: str, right: str) -> str:
        """``Poseidon4`` over ``[prefix, left, right]`` as a field literal.

        The operands go through ``Plaintext`` so they hash as a 3-element array,
        which is what the Leo side does.  Hashing a bare list of fields, or
        going through ``to_fields_raw``, gives a different and wrong answer.
        """
        net = self._net()
        if self._hasher is None:
            self._hasher = net.Poseidon4()
        plaintext = net.Plaintext.from_string(f"[{prefix},{left},{right}]")
        return str(self._hasher.hash(plaintext.to_fields()))
