"""Sealance compliance tree — Merkle exclusion proofs for ARC-22 (USDCx) private transfers and burns.

Port of ``sdk/src/integrations/sealance/merkle-tree.ts``. Leaves are frozen addresses as little-endian
field ints, sorted ascending and front-padded with ``0field`` to a power of two (minimum two). The leaf
level hashes ``Poseidon4([1field, l, r])``, inner levels ``Poseidon4([0field, l, r])`` — with the
three-element array packed through ``Plaintext.to_fields()``, exactly as the Leo program does. A sibling
path starts with the leaf itself, then one sibling per level, zero-padded to ``depth`` entries.
The deployed ``MerkleProof`` struct is ``[field; 16]`` + ``u32``, so proofs use ``PROOF_SIBLINGS = 16``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from . import encoding as enc
from .errors import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover
    from .client import Bridge

ZERO_ADDRESS = "aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq3ljyzc"
DEFAULT_DEPTH = 15                    # TS getSiblingPath default; tree capacity 2**(depth-1) leaves
PROOF_SIBLINGS = 16                   # struct MerkleProof { siblings: [field; 16u32], leaf_index: u32 }
FREEZE_LIST_MAPPING = "freeze_list"                    # u32 => address
FREEZE_LIST_LAST_INDEX_MAPPING = "freeze_list_last_index"   # bool => u32, keyed "true"
EMPTY_TREE_ROOT = 3642222252059314292809609689035560016959342421640560347114299934615987159853
_EMPTY_PROOF = "{ siblings: [" + ", ".join(["0field"] * PROOF_SIBLINGS) + "], leaf_index: 1u32 }"
EMPTY_MERKLE_PROOF_PAIR = f"[{_EMPTY_PROOF}, {_EMPTY_PROOF}]"


def address_to_field_int(address: str) -> int:
    """bech32m payload bytes read little-endian — the field element a Leo program sees for an address."""
    return int.from_bytes(enc.aleo_address_to_bytes32(address), "little")


def hash_two(prefix: str, left: str, right: str, network: str) -> str:
    """``Poseidon4(Plaintext("[prefix,left,right]").to_fields())`` as a ``…field`` literal."""
    net = enc.network_module(network)
    plaintext = net.Plaintext.from_string(f"[{prefix},{left},{right}]")
    return str(net.Poseidon4().hash(plaintext.to_fields()))


def generate_leaves(addresses: list[str], depth: int = DEFAULT_DEPTH) -> list[str]:
    """Frozen addresses → sorted ``…field`` leaves, zero address dropped, front-padded to a power of two (min 2)."""
    live = [a for a in addresses if a != ZERO_ADDRESS]
    max_leaves = 2 ** (depth - 1)
    if len(live) > max_leaves:
        raise ConfigurationError(f"Leaves limit exceeded. Max: {max_leaves}, provided: {len(live)}")
    count = 2 if len(live) <= 1 else 2 ** math.ceil(math.log2(len(live)))
    fields = sorted(address_to_field_int(a) for a in live)
    return ["0field"] * (count - len(fields)) + [f"{f}field" for f in fields]


def build_tree(leaves: list[str], network: str) -> list[int]:
    """Bottom-up tree as ints: leaves first, root last (the layout the TS SDK and compliance API use)."""
    if not leaves:
        raise ConfigurationError("Leaves array cannot be empty")
    if len(leaves) % 2:
        raise ConfigurationError("Leaves array must have even number of elements")
    tree = list(leaves)
    level = list(leaves)
    while len(level) > 1:
        prefix = "1field" if len(level) == len(leaves) else "0field"
        level = [hash_two(prefix, level[i], level[i + 1], network) for i in range(0, len(level), 2)]
        tree.extend(level)
    return [int(node[: -len("field")]) for node in tree]


def leaf_indices(tree: list[int], address: str) -> tuple[int, int]:
    """(left, right) leaf indices bracketing *address* for a non-inclusion proof (TS ``getLeafIndices``)."""
    count = (len(tree) + 1) // 2
    target = address_to_field_int(address)
    leaves = tree[:count]
    right = next((i for i, leaf in enumerate(leaves) if target <= leaf), -1)
    left = right - 1
    if right == -1:
        right = left = count - 1
    if right == 0:
        left = 0
    return left, right


def sibling_path(tree: list[int], index: int, depth: int = DEFAULT_DEPTH) -> list[int]:
    """Leaf, then the sibling at each level, zero-padded to *depth* entries (TS ``getSiblingPath``)."""
    count = (len(tree) + 1) // 2
    path = [tree[index]]
    node, parent, level = index, count, 1
    while parent < len(tree):
        sibling = node + 1 if node % 2 == 0 else node - 1
        path.append(tree[sibling])
        node = parent + index // 2 ** level
        parent += count // 2 ** level
        level += 1
    while len(path) < depth:
        path.append(0)
    return path


def format_merkle_proof(left: tuple[list[int], int], right: tuple[list[int], int]) -> str:
    """``[MerkleProof; 2]`` literal with veil's spacing: ``{ siblings: [a, b], leaf_index: Nu32 }``."""
    parts = []
    for siblings, index in (left, right):
        parts.append("{ siblings: [" + ", ".join(f"{s}field" for s in siblings) + f"], leaf_index: {index}u32 }}")
    return "[" + ", ".join(parts) + "]"


class FreezeList:
    """``bridge.freezelist`` — reads a compliant token's frozen addresses and proves an address is not among them."""

    def __init__(self, bridge: "Bridge") -> None:
        self._bridge = bridge

    def leaves(self, program: str) -> list[str]:
        """Frozen addresses from ``program``'s ``freeze_list`` mapping (indices 0..last inclusive); ``[]`` when none."""
        last = self._bridge.mapping_value(program, FREEZE_LIST_LAST_INDEX_MAPPING, "true")
        if last is None:
            return []
        try:
            count = int(last.removesuffix("u32"))
        except ValueError as exc:
            raise ConfigurationError(f"{program}/{FREEZE_LIST_LAST_INDEX_MAPPING} returned {last!r}, expected a u32") from exc
        addresses = []
        for index in range(count + 1):
            value = self._bridge.mapping_value(program, FREEZE_LIST_MAPPING, f"{index}u32")
            if value and value != ZERO_ADDRESS:
                addresses.append(value)
        return addresses

    def tree(self, program: str) -> list[int]:
        return build_tree(generate_leaves(self.leaves(program)), self._bridge.network)

    def exclusion_proof(self, address: str, program: str) -> str:
        """``[MerkleProof; 2]`` proving *address* is not frozen on *program*; veil's empty pair when the list is empty."""
        leaves = self.leaves(program)
        if not leaves:
            return EMPTY_MERKLE_PROOF_PAIR
        tree = build_tree(generate_leaves(leaves), self._bridge.network)
        count = (len(tree) + 1) // 2
        target = address_to_field_int(address)
        if target in tree[:count]:
            raise ConfigurationError(f"{address} is on the {program} freeze list; no exclusion proof exists for it")
        left, right = leaf_indices(tree, address)
        return format_merkle_proof((sibling_path(tree, left, PROOF_SIBLINGS), left),
                                   (sibling_path(tree, right, PROOF_SIBLINGS), right))


__all__ = ["DEFAULT_DEPTH", "EMPTY_MERKLE_PROOF_PAIR", "EMPTY_TREE_ROOT", "FREEZE_LIST_LAST_INDEX_MAPPING",
           "FREEZE_LIST_MAPPING", "PROOF_SIBLINGS", "ZERO_ADDRESS", "FreezeList", "address_to_field_int",
           "build_tree", "format_merkle_proof", "generate_leaves", "hash_two", "leaf_indices", "sibling_path"]
