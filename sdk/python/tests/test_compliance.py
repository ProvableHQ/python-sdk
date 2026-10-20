"""aleo.compliance — freeze-list trees, exclusion proofs, and the per-client proof cache.

Tree vectors come from ProvableHQ/sdk ``sdk/src/integrations/sealance/merkle-tree.ts``
docstrings (moved here from the bridge package's test_freezelist.py). Resolution tests
run against a fake facade that scripts the root mapping and the compliance endpoint.
"""
from __future__ import annotations

from typing import Any

import pytest

from aleo import compliance as fl
from aleo.compliance import (
    EMPTY_MERKLE_PROOF_PAIR, EMPTY_TREE_ROOT, AsyncFreezeListProofs, FreezeListError, FreezeListProofs,
    FrozenAddressError, MerkleProof, MerkleProofPair, ProofSlot, bundle_literal, exclusion_proof_from_tree,
    freeze_list_program, one_bundle,
)

A = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
B = "aleo1s3ws5tra87fjycnjrwsjcrnw2qxr8jfqqdugnf0xzqqw29q9m5pqem2u4t"
A_FIELD = 3501665755452795161867664882580888971213780722176652848275908626939553697821
B_FIELD = 1295133970529764960316948294624974168921228814652993007266766481909235735940
RECIPIENT = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
C = "aleo15mstsvdtzqf5nw8rfzx8mrllwxt907amfpt8nx3p8cskj4wd3uxq4uywn9"
C_FIELD = 5681830473930530271878304807765952613665599463594380247149021024878193074342
LOW = "aleo13rmacw9jk943wwg4s6t5yxcycvl3ud356l3cyzgav8hw9fkcjsqsgt4vy3"     # sorts below B
MID = "aleo1240fsvz2dhmj0cdtt8mc0yc8um9fmu236rqcl2qnlj9703hd2vpsdwyrtf"     # sorts between B and A
ZERO = fl.ZERO_ADDRESS
LIST = "usdcx_freezelist.aleo"
EMPTY_ONE = "{ siblings: [" + ", ".join(["0field"] * 16) + "], leaf_index: 1u32 }"
NET = "mainnet"


# ── Tree math (vectors) ───────────────────────────────────────────────────────

def test_empty_pair_literal_matches_veil():
    assert EMPTY_MERKLE_PROOF_PAIR == f"[{EMPTY_ONE}, {EMPTY_ONE}]"
    assert EMPTY_MERKLE_PROOF_PAIR.count("0field") == 32 and EMPTY_MERKLE_PROOF_PAIR.count("leaf_index: 1u32") == 2
    assert fl.PROOF_SIBLINGS == 16 and fl.DEFAULT_DEPTH == 15


def test_address_to_field_int_matches_ts_docstrings():
    assert fl.address_to_field_int(A) == A_FIELD
    assert fl.address_to_field_int(B) == B_FIELD
    assert fl.address_to_field_int(ZERO) == 0
    with pytest.raises(FreezeListError, match="Not an Aleo address"):
        fl.address_to_field_int("aleo1nope")


def test_hash_two_uses_poseidon4_over_packed_array():
    from aleo import mainnet as net
    expected = str(net.Poseidon4().hash(net.Plaintext.from_string("[1field,3field,4field]").to_fields()))
    assert fl.hash_two("1field", "3field", "4field", NET) == expected
    assert fl.hash_two("0field", "3field", "4field", NET) != expected      # prefix matters (leaf vs node)


def test_empty_tree_root_pinned():
    # Pinned 2026-09-03 from the live compliance tree ([0, 0, root]); investigate rather than re-pin on failure.
    assert fl.build_tree(["0field", "0field"], NET) == [0, 0, EMPTY_TREE_ROOT]


def test_generate_leaves_sorts_pads_and_filters_zero():
    assert fl.generate_leaves([A, B, B]) == ["0field", f"{B_FIELD}field", f"{B_FIELD}field", f"{A_FIELD}field"]
    assert fl.generate_leaves([ZERO, ZERO, A]) == ["0field", f"{A_FIELD}field"]
    assert fl.generate_leaves([]) == ["0field", "0field"]
    assert len(fl.generate_leaves([A] * 5)) == 8 and fl.generate_leaves([A] * 5)[:3] == ["0field"] * 3
    with pytest.raises(FreezeListError, match="Leaves limit exceeded"):
        fl.generate_leaves([A] * (2 ** 14 + 1), 15)


def test_build_tree_shapes_and_errors():
    two = fl.build_tree(["1field", "2field"], NET)
    four = fl.build_tree(["1field", "2field", "3field", "4field"], NET)
    assert len(two) == 3 and two[:2] == [1, 2]
    assert len(four) == 7 and four[:4] == [1, 2, 3, 4]
    assert f"{four[4]}field" == fl.hash_two("1field", "1field", "2field", NET)
    assert f"{four[6]}field" == fl.hash_two("0field", f"{four[4]}field", f"{four[5]}field", NET)
    with pytest.raises(FreezeListError, match="cannot be empty"):
        fl.build_tree([], NET)
    with pytest.raises(FreezeListError, match="even number"):
        fl.build_tree(["1field", "2field", "3field"], NET)


def test_leaf_indices_and_sibling_path():
    tree = fl.build_tree(fl.generate_leaves([A]), NET)          # leaves [0, A_FIELD]
    assert fl.leaf_indices(tree, RECIPIENT) == (1, 1)
    assert fl.leaf_indices(tree, A) == (0, 1)
    four = fl.build_tree(["1field", "2field", "3field", "4field"], NET)
    path = fl.sibling_path(four, 1, 15)
    assert len(path) == 15 and path[:3] == [2, 1, four[5]] and path[3:] == [0] * 12
    assert len(fl.sibling_path(four, 1, 16)) == 16
    assert fl.format_merkle_proof(([2, 1] + [0] * 14, 1), ([3, 4] + [0] * 14, 2)).startswith(
        "[{ siblings: [2field, 1field, 0field")


def test_pure_exclusion_proof_of_empty_tree_equals_veil_literal():
    tree = fl.build_tree(["0field", "0field"], NET)
    left, right = fl.leaf_indices(tree, RECIPIENT)
    assert (left, right) == (1, 1)
    assert fl.format_merkle_proof((fl.sibling_path(tree, left, 16), left),
                                  (fl.sibling_path(tree, right, 16), right)) == EMPTY_MERKLE_PROOF_PAIR


# ── Proof values ──────────────────────────────────────────────────────────────

def test_merkle_proof_round_trips_through_its_literal():
    one = MerkleProof(tuple([A_FIELD, 0] + [0] * 14), 1)
    assert str(one) == "{ siblings: [" + f"{A_FIELD}field, 0field, " + ", ".join(["0field"] * 14) + "], leaf_index: 1u32 }"
    assert MerkleProof.from_string(str(one)) == one
    pair = MerkleProofPair(one, one)
    assert MerkleProofPair.from_string(str(pair)) == pair
    assert MerkleProofPair.from_string(EMPTY_MERKLE_PROOF_PAIR) == MerkleProofPair(
        MerkleProof((0,) * 16, 1), MerkleProof((0,) * 16, 1))
    with pytest.raises(FreezeListError, match="16 siblings"):
        MerkleProof((1, 2), 0)
    with pytest.raises(FreezeListError, match="Not a MerkleProof"):
        MerkleProof.from_string("[nope]")
    with pytest.raises(FreezeListError, match="two proofs"):
        MerkleProofPair.from_string(f"[{EMPTY_ONE}]")


def test_bundle_literal_accepts_every_supported_form():
    one = MerkleProof((0,) * 16, 1)
    assert bundle_literal(EMPTY_MERKLE_PROOF_PAIR) == EMPTY_MERKLE_PROOF_PAIR
    assert bundle_literal(f"  {EMPTY_MERKLE_PROOF_PAIR} ") == EMPTY_MERKLE_PROOF_PAIR
    assert bundle_literal(MerkleProofPair(one, one)) == EMPTY_MERKLE_PROOF_PAIR
    assert bundle_literal((one, one)) == EMPTY_MERKLE_PROOF_PAIR
    with pytest.raises(FreezeListError, match="encoded"):
        bundle_literal("not a literal")
    with pytest.raises(FreezeListError, match="MerkleProofPair"):
        bundle_literal((one,))  # type: ignore[arg-type]


def test_one_bundle_handles_new_old_and_conflicting_arguments():
    one = MerkleProof((0,) * 16, 1)
    assert one_bundle(None) is None
    assert one_bundle(EMPTY_MERKLE_PROOF_PAIR) == EMPTY_MERKLE_PROOF_PAIR
    assert one_bundle((EMPTY_MERKLE_PROOF_PAIR,)) == EMPTY_MERKLE_PROOF_PAIR
    assert one_bundle((one, one)) == EMPTY_MERKLE_PROOF_PAIR            # a bare path tuple is one bundle
    assert one_bundle(((one, one),)) == EMPTY_MERKLE_PROOF_PAIR          # a one-tuple of a path tuple too
    assert one_bundle(None, EMPTY_MERKLE_PROOF_PAIR) == EMPTY_MERKLE_PROOF_PAIR
    with pytest.raises(FreezeListError, match="not both"):
        one_bundle(EMPTY_MERKLE_PROOF_PAIR, EMPTY_MERKLE_PROOF_PAIR)
    with pytest.raises(FreezeListError, match="exactly one"):
        one_bundle((EMPTY_MERKLE_PROOF_PAIR, EMPTY_MERKLE_PROOF_PAIR))
    with pytest.raises(FreezeListError, match="merkle_proof= takes the encoded"):
        one_bundle(None, (one, one))


# ── Witnesses against a populated tree ───────────────────────────────────────

def _populated() -> tuple[list[int], int]:
    tree = fl.build_tree(fl.generate_leaves([A, B, C]), NET)    # leaves [0, B, A, C] (B < A < C)
    assert tree[:4] == [0, B_FIELD, A_FIELD, C_FIELD]
    return tree, tree[-1]


def test_exclusion_proof_from_tree_builds_verified_witnesses():
    tree, root = _populated()
    literal = exclusion_proof_from_tree(tree, RECIPIENT, root, NET)
    pair = MerkleProofPair.from_string(literal)
    # RECIPIENT sorts above every leaf, so both paths open the last leaf (C).
    assert pair.left.leaf_index == 3 and pair.right.leaf_index == 3
    assert pair.left.siblings[0] == C_FIELD and pair.left.siblings[1] == A_FIELD and pair.left.siblings[2] == tree[4]
    assert pair.left.siblings[3:] == (0,) * 13
    assert fl._reconstruct_root(pair.left, NET) == root
    # LOW sorts between the zero pad and B: the pair opens leaves 0 and 1.
    low = MerkleProofPair.from_string(exclusion_proof_from_tree(tree, LOW, root, NET))
    assert (low.left.leaf_index, low.right.leaf_index) == (0, 1)
    assert low.left.siblings[0] == 0 and low.right.siblings[0] == B_FIELD
    # MID sorts between B and A: leaves 1 and 2.
    mid = MerkleProofPair.from_string(exclusion_proof_from_tree(tree, MID, root, NET))
    assert (mid.left.leaf_index, mid.right.leaf_index) == (1, 2)
    # The zero address equals the padding leaf and is reported as frozen, like any leaf value.
    with pytest.raises(FrozenAddressError):
        exclusion_proof_from_tree(tree, ZERO, root, NET)


def test_exclusion_proof_from_tree_empty_list_is_the_fixed_pair():
    assert exclusion_proof_from_tree([], RECIPIENT, EMPTY_TREE_ROOT, NET) == EMPTY_MERKLE_PROOF_PAIR
    assert exclusion_proof_from_tree([0, 0, EMPTY_TREE_ROOT], RECIPIENT, EMPTY_TREE_ROOT, NET) == EMPTY_MERKLE_PROOF_PAIR


def test_exclusion_proof_from_tree_rejects_bad_input():
    tree, root = _populated()
    with pytest.raises(FrozenAddressError, match="on the freeze list"):
        exclusion_proof_from_tree(tree, A, root, NET)
    with pytest.raises(FreezeListError, match="does not match the on-chain root"):
        exclusion_proof_from_tree(tree, RECIPIENT, root + 1, NET)
    with pytest.raises(FreezeListError, match="complete binary tree"):
        exclusion_proof_from_tree(tree[:-1], RECIPIENT, root, NET)
    unsorted = [0, A_FIELD, B_FIELD, C_FIELD, *tree[4:]]
    with pytest.raises(FreezeListError, match="sorted"):
        exclusion_proof_from_tree(unsorted, RECIPIENT, root, NET)
    corrupt = list(tree)
    corrupt[4] += 1                                               # an inner node that does not hash from its leaves
    with pytest.raises(FreezeListError, match="does not reconstruct"):
        exclusion_proof_from_tree(corrupt, RECIPIENT, root, NET)


# ── Fake facade ───────────────────────────────────────────────────────────────

class _Program:
    def __init__(self, imports: list[str]) -> None:
        self.imports = imports


class _Programs:
    def __init__(self, fake: "FakeAleo") -> None:
        self._fake = fake

    def get(self, program_id: str) -> _Program:
        from aleo.facade.errors import ProgramNotFound
        self._fake.fetched.append(program_id)
        if program_id not in self._fake.imports:
            raise ProgramNotFound(program_id)
        return _Program(self._fake.imports[program_id])


class _Network:
    def __init__(self, fake: "FakeAleo") -> None:
        self._fake = fake

    def get_program_mapping_value(self, program_id: str, mapping: str, key: str) -> Any:
        self._fake.mapping_reads.append((program_id, mapping, key))
        return self._fake.mappings.get(program_id, {}).get(mapping, {}).get(key)

    def get_program_freeze_list(self, program_id: str) -> list[int] | None:
        self._fake.tree_reads.append(program_id)
        tree = self._fake.trees.get(program_id)
        if isinstance(tree, Exception):
            raise tree
        return tree


class FakeAleo:
    network_name = NET

    def __init__(self) -> None:
        self.mappings: dict[str, dict[str, dict[str, str]]] = {}
        self.trees: dict[str, Any] = {}
        self.imports: dict[str, list[str]] = {}
        self.fetched: list[str] = []
        self.mapping_reads: list[tuple[str, str, str]] = []
        self.tree_reads: list[str] = []
        self.programs = _Programs(self)
        self.network = _Network(self)

    def set_root(self, program: str, root: int | str) -> None:
        value = f"{root}field" if isinstance(root, int) else root
        self.mappings.setdefault(program, {}).setdefault("freeze_list_root", {})["1u8"] = value


class _AsyncPrograms(_Programs):
    async def get(self, program_id: str) -> _Program:  # type: ignore[override]
        return super().get(program_id)


class _AsyncNetwork(_Network):
    async def get_program_mapping_value(self, program_id: str, mapping: str, key: str) -> Any:  # type: ignore[override]
        return super().get_program_mapping_value(program_id, mapping, key)

    async def get_program_freeze_list(self, program_id: str) -> list[int] | None:  # type: ignore[override]
        return super().get_program_freeze_list(program_id)


class FakeAsyncAleo(FakeAleo):
    def __init__(self) -> None:
        super().__init__()
        self.programs = _AsyncPrograms(self)
        self.network = _AsyncNetwork(self)


# ── Which list a program checks ──────────────────────────────────────────────

def test_freeze_list_program_resolution_order():
    fake = FakeAleo()
    assert freeze_list_program(fake, LIST) == LIST                      # already a list
    assert freeze_list_program(fake, "usdcx_stablecoin.aleo") == LIST   # known table, no fetch
    assert freeze_list_program(fake, "credits.aleo") is None            # ignores the proof
    assert fake.fetched == []
    fake.imports["shield_swap_arc20_wrapped_usdcx.aleo"] = ["merkle_tree.aleo", "test_usdcx_freezelist.aleo"]
    assert freeze_list_program(fake, "shield_swap_arc20_wrapped_usdcx.aleo") == "test_usdcx_freezelist.aleo"
    with pytest.raises(FreezeListError, match="names no freeze-list program"):
        freeze_list_program(fake, "arc20_wbtc.aleo")


def test_freeze_list_program_propagates_unrelated_exceptions():
    fake = FakeAleo()

    def boom(program_id: str) -> _Program:
        raise RuntimeError("rpc hiccup")

    fake.programs.get = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="rpc hiccup"):
        freeze_list_program(fake, "arc20_wbtc.aleo")


# ── Resolution and caching ────────────────────────────────────────────────────

def test_empty_root_short_circuits_to_the_empty_pair_without_fetching_the_tree():
    fake = FakeAleo()
    fake.set_root(LIST, EMPTY_TREE_ROOT)
    proofs = FreezeListProofs(fake)
    assert proofs.exclusion_proof(RECIPIENT, "usdcx_stablecoin.aleo") == EMPTY_MERKLE_PROOF_PAIR
    assert proofs.exclusion_proof(RECIPIENT, "credits.aleo") == EMPTY_MERKLE_PROOF_PAIR
    assert fake.tree_reads == []
    assert fake.mapping_reads == [(LIST, "freeze_list_root", "1u8")]


def test_unreadable_or_invalid_root_is_fatal():
    fake = FakeAleo()
    proofs = FreezeListProofs(fake)
    with pytest.raises(FreezeListError, match=r"freeze_list_root\[1u8\] is unreadable"):
        proofs.exclusion_proof(RECIPIENT, LIST)
    fake.set_root(LIST, "garbage")
    with pytest.raises(FreezeListError, match="Invalid current freeze-list root"):
        proofs.exclusion_proof(RECIPIENT, LIST)


def test_populated_list_is_proved_from_the_published_tree_and_cached_per_root():
    fake = FakeAleo()
    tree, root = _populated()
    fake.set_root(LIST, root)
    fake.trees[LIST] = tree
    proofs = FreezeListProofs(fake)
    first = proofs.exclusion_proof(RECIPIENT, LIST)
    assert first == exclusion_proof_from_tree(tree, RECIPIENT, root, NET)
    assert proofs.exclusion_proof(RECIPIENT, LIST) == first
    assert proofs.exclusion_proof(LOW, LIST) != first                # another subject, same tree
    assert fake.tree_reads == [LIST]                                 # one fetch while the root is unchanged
    assert len(fake.mapping_reads) == 3                              # the root is re-read every call
    with pytest.raises(FrozenAddressError):
        proofs.exclusion_proof(A, LIST)
    # The list changes: new root → the old tree is discarded and fetched again.
    bigger = fl.build_tree(fl.generate_leaves([A, B, RECIPIENT]), NET)
    fake.set_root(LIST, bigger[-1])
    fake.trees[LIST] = bigger
    with pytest.raises(FrozenAddressError):
        proofs.exclusion_proof(RECIPIENT, LIST)
    assert fake.tree_reads == [LIST, LIST]
    assert proofs.exclusion_proof(LOW, LIST) == exclusion_proof_from_tree(bigger, LOW, bigger[-1], NET)
    proofs.clear()
    proofs.exclusion_proof(LOW, LIST)
    assert fake.tree_reads == [LIST, LIST, LIST]


def test_stale_published_tree_is_not_cached():
    fake = FakeAleo()
    tree, root = _populated()
    fake.set_root(LIST, root + 1)                                    # chain moved, indexer still serves the old tree
    fake.trees[LIST] = tree
    proofs = FreezeListProofs(fake)
    with pytest.raises(FreezeListError, match="does not match the on-chain root"):
        proofs.exclusion_proof(RECIPIENT, LIST)
    assert len(proofs._cache) == 0
    fake.trees[LIST] = FreezeListError("indexer down")
    with pytest.raises(FreezeListError, match="indexer down"):
        proofs.exclusion_proof(RECIPIENT, LIST)
    fake.trees[LIST] = None
    with pytest.raises(FreezeListError, match="publishes no compliance tree"):
        proofs.exclusion_proof(RECIPIENT, LIST)
    assert len(proofs._cache) == 0


def test_resolve_reads_each_list_root_once_per_call_and_fills_placeholders():
    fake = FakeAleo()
    tree, root = _populated()
    fake.set_root(LIST, root)
    fake.trees[LIST] = tree
    fake.set_root("shield_swap_freezelist.aleo", EMPTY_TREE_ROOT)
    proofs = FreezeListProofs(fake)
    out = proofs.resolve([ProofSlot("shield_swap_freezelist.aleo", RECIPIENT), None, ProofSlot(None, RECIPIENT),
                          ProofSlot(LIST, RECIPIENT), ProofSlot(LIST, LOW), ProofSlot("shield_swap_freezelist.aleo", LOW)])
    assert out[0] == out[1] == out[2] == out[5] == EMPTY_MERKLE_PROOF_PAIR
    assert out[3] == exclusion_proof_from_tree(tree, RECIPIENT, root, NET)
    assert out[4] == exclusion_proof_from_tree(tree, LOW, root, NET)
    assert fake.mapping_reads == [("shield_swap_freezelist.aleo", "freeze_list_root", "1u8"), (LIST, "freeze_list_root", "1u8")]


def test_cache_is_bounded():
    fake = FakeAleo()
    proofs = FreezeListProofs(fake)
    for i in range(fl.MAX_CACHED_LISTS + 2):
        fake.set_root(f"l{i}_freezelist.aleo", EMPTY_TREE_ROOT)
        proofs.exclusion_proof(RECIPIENT, f"l{i}_freezelist.aleo")
    assert len(proofs._cache) == fl.MAX_CACHED_LISTS
    tree, root = _populated()
    fake.set_root(LIST, root)
    fake.trees[LIST] = tree
    entry = proofs._entry(LIST, root)
    for i in range(fl.MAX_CACHED_ADDRESSES + 5):
        entry.proofs[f"addr{i}"] = "x"
        if len(entry.proofs) > fl.MAX_CACHED_ADDRESSES:
            entry.proofs.popitem(last=False)
    proofs.exclusion_proof(RECIPIENT, LIST)
    assert len(entry.proofs) == fl.MAX_CACHED_ADDRESSES


@pytest.mark.asyncio
async def test_async_resolver_matches_sync():
    fake = FakeAsyncAleo()
    tree, root = _populated()
    fake.set_root(LIST, root)
    fake.trees[LIST] = tree
    fake.set_root("shield_swap_freezelist.aleo", EMPTY_TREE_ROOT)
    fake.imports["shield_swap_arc20_wrapped_usdcx.aleo"] = ["test_usdcx_freezelist.aleo"]
    proofs = AsyncFreezeListProofs(fake)
    assert await proofs.exclusion_proof(RECIPIENT, "credits.aleo") == EMPTY_MERKLE_PROOF_PAIR
    assert await proofs.exclusion_proof(RECIPIENT, "usdcx_stablecoin.aleo") == exclusion_proof_from_tree(tree, RECIPIENT, root, NET)
    assert await proofs.exclusion_proof(RECIPIENT, LIST) == exclusion_proof_from_tree(tree, RECIPIENT, root, NET)
    assert fake.tree_reads == [LIST]
    out = await proofs.resolve([ProofSlot("shield_swap_freezelist.aleo", RECIPIENT), ProofSlot(LIST, LOW), None])
    assert out[0] == out[2] == EMPTY_MERKLE_PROOF_PAIR and out[1] == exclusion_proof_from_tree(tree, LOW, root, NET)
    assert await fl.async_freeze_list_program(fake, "shield_swap_arc20_wrapped_usdcx.aleo") == "test_usdcx_freezelist.aleo"
    with pytest.raises(FrozenAddressError):
        await proofs.exclusion_proof(A, LIST)
    fake.set_root(LIST, root + 1)
    with pytest.raises(FreezeListError, match="does not match"):
        await proofs.exclusion_proof(RECIPIENT, LIST)
    assert proofs._cache.current(NET, LIST, root + 1) is None        # the stale tree was not kept
    assert len(proofs._cache) == 1                                   # the shield_swap list entry survives
