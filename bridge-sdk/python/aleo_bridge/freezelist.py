"""Freeze-list exclusion proofs for ARC-22 (USDCx) private transfers and burns.

The tree math and the per-client proof cache live in :mod:`aleo.compliance`;
this module keeps the bridge's names for them and the ``bridge.freezelist``
module that reads a token's list straight from its mappings.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from aleo.compliance import (  # noqa: F401 — re-exported under the bridge's historical names
    CURRENT_ROOT_KEY,
    DEFAULT_DEPTH,
    EMPTY_MERKLE_PROOF_PAIR,
    EMPTY_TREE_ROOT,
    FREEZE_LIST_INDEX_MAPPING,
    FREEZE_LIST_LAST_INDEX_MAPPING,
    FREEZE_LIST_MAPPING,
    FREEZE_LIST_ROOT_MAPPING,
    KNOWN_TOKEN_FREEZELISTS,
    PROOF_SIBLINGS,
    ZERO_ADDRESS,
    FreezeListError,
    FreezeListProofs,
    FrozenAddressError,
    MerkleProofInput,
    MerkleProofPair,
    ComplianceProofs,
    address_to_field_int,
    build_tree,
    format_merkle_proof,
    generate_leaves,
    hash_two,
    leaf_indices,
    sibling_path,
)
from aleo.compliance import literal_compliance_proof as _literal_compliance_proof

from .errors import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover
    from .client import Bridge

#: Token program -> its freeze-list program, for when ``Bridge.program(token).imports`` is unavailable.
FREEZE_LIST_PROGRAMS = {program: freezelist for program, freezelist in KNOWN_TOKEN_FREEZELISTS.items() if freezelist}


def literal_compliance_proof(compliance_proofs: ComplianceProofs | tuple[ComplianceProofs] | None, merkle_proof: str | None) -> str | None:
    """The single ``[MerkleProof; 2]`` literal a bridge call was given, or ``None`` to compute one.

    Raises:
        ConfigurationError: Both arguments were given, or the value is not a proof bundle.
    """
    try:
        return _literal_compliance_proof(compliance_proofs, merkle_proof)
    except FreezeListError as exc:
        raise ConfigurationError(str(exc)) from exc


class FreezeList:
    """``bridge.freezelist`` — prove that an address is not on a compliant token's freeze list.

    ``exclusion_proof`` is what every private USDCx unshield and burn needs. It
    reads the list's current root, fetches the published tree only when that
    root changed since the last call, and reuses verified witnesses while it
    is unchanged. While the list is empty the fixed empty-tree pair is
    returned. ``leaves`` and ``tree`` read the list entry by entry from the
    program's mappings instead, for callers that want to inspect it.
    """

    def __init__(self, bridge: "Bridge") -> None:
        self._bridge = bridge
        self._proofs = FreezeListProofs(bridge.aleo)

    def clear_cache(self) -> None:
        """Forget every cached tree and witness; the next proof reads the chain again."""
        self._proofs.clear()

    def freeze_list_program(self, token_program: str) -> str:
        """The freeze-list program backing *token_program* (usually a token program that imports it).

        Raises:
            ConfigurationError: *token_program* names no freeze list.
        """
        if token_program.endswith("freezelist.aleo"):
            return token_program
        from aleo.facade.errors import AleoError, ProgramNotFound
        try:
            imports = self._bridge.program(token_program).imports
        except (ProgramNotFound, AleoError):
            imports = []
        for dep in imports:
            if str(dep).endswith("freezelist.aleo"):
                return str(dep)
        fallback = FREEZE_LIST_PROGRAMS.get(token_program)
        if fallback:
            return fallback
        raise ConfigurationError(f"{token_program} has no freeze-list program; pass compliance_proofs explicitly")

    def leaves(self, program: str) -> list[str]:
        """Frozen addresses from *program*'s freeze-list ``freeze_list_index`` mapping (0..last inclusive),
        with the zero-address sentinel dropped; ``[]`` when the list is empty/unreadable."""
        fl_program = self.freeze_list_program(program)
        last = self._bridge.mapping_value(fl_program, FREEZE_LIST_LAST_INDEX_MAPPING, "true")
        if last is None:
            return []
        try:
            count = int(last.removesuffix("u32"))
        except ValueError as exc:
            raise ConfigurationError(f"{fl_program}/{FREEZE_LIST_LAST_INDEX_MAPPING} returned {last!r}, expected a u32") from exc
        addresses: list[str] = []
        for index in range(count + 1):
            value = self._bridge.mapping_value(fl_program, FREEZE_LIST_INDEX_MAPPING, f"{index}u32")
            if value and value != ZERO_ADDRESS:
                addresses.append(value)
        return addresses

    def _verified_tree(self, fl_program: str, leaves: list[str]) -> list[int]:
        tree = build_tree(generate_leaves(leaves), self._bridge.network)
        on_chain_root = self._bridge.mapping_value(fl_program, FREEZE_LIST_ROOT_MAPPING, CURRENT_ROOT_KEY)
        if on_chain_root is None:
            raise ConfigurationError(
                f"{fl_program}/{FREEZE_LIST_ROOT_MAPPING}[{CURRENT_ROOT_KEY}] is unreadable; pass compliance_proofs= explicitly")
        computed_root = f"{tree[-1]}field"
        if computed_root != on_chain_root:
            raise ConfigurationError(
                f"computed freeze-list root {computed_root} != on-chain root {on_chain_root} for {fl_program}; "
                "refusing to build a proof")
        return tree

    def tree(self, program: str) -> list[int]:
        """Merkle tree over *program*'s frozen addresses (read from its mappings), verified against the on-chain root."""
        return self._verified_tree(self.freeze_list_program(program), self.leaves(program))

    def exclusion_proof(self, address: str, program: str) -> str:
        """``[MerkleProof; 2]`` proving *address* is not frozen on *program*; the empty pair when the list is empty.

        Reads the current root every call and the published tree only when
        the root changed. Raises ``ConfigurationError`` when the root is
        unreadable, the published tree does not match it, or *address* is on
        the list.
        """
        fl_program = self.freeze_list_program(program)
        try:
            return self._proofs.exclusion_proof(address, fl_program)
        except FreezeListError as exc:
            raise ConfigurationError(str(exc)) from exc


__all__ = ["CURRENT_ROOT_KEY", "DEFAULT_DEPTH", "EMPTY_MERKLE_PROOF_PAIR", "EMPTY_TREE_ROOT",
           "FREEZE_LIST_INDEX_MAPPING", "FREEZE_LIST_LAST_INDEX_MAPPING", "FREEZE_LIST_MAPPING",
           "FREEZE_LIST_PROGRAMS", "FREEZE_LIST_ROOT_MAPPING", "PROOF_SIBLINGS", "ZERO_ADDRESS", "FreezeList",
           "FreezeListError", "FrozenAddressError", "MerkleProofInput", "MerkleProofPair", "ComplianceProofs",
           "address_to_field_int", "build_tree", "format_merkle_proof", "generate_leaves", "hash_two",
           "leaf_indices", "literal_compliance_proof", "sibling_path"]
