import pytest

from aleo_bridge import freezelist as fl
from aleo_bridge.errors import ConfigurationError

# Vectors from sdk/src/integrations/sealance/merkle-tree.ts docstrings
A = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
B = "aleo1s3ws5tra87fjycnjrwsjcrnw2qxr8jfqqdugnf0xzqqw29q9m5pqem2u4t"
A_FIELD = 3501665755452795161867664882580888971213780722176652848275908626939553697821
B_FIELD = 1295133970529764960316948294624974168921228814652993007266766481909235735940
RECIPIENT = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
ZERO = fl.ZERO_ADDRESS
PROGRAM = "usdcx_stablecoin.aleo"
EMPTY_ONE = "{ siblings: [" + ", ".join(["0field"] * 16) + "], leaf_index: 1u32 }"


def test_empty_pair_literal_matches_veil():
    assert fl.EMPTY_MERKLE_PROOF_PAIR == f"[{EMPTY_ONE}, {EMPTY_ONE}]"
    assert fl.EMPTY_MERKLE_PROOF_PAIR.count("0field") == 32 and fl.EMPTY_MERKLE_PROOF_PAIR.count("leaf_index: 1u32") == 2
    assert fl.PROOF_SIBLINGS == 16 and fl.DEFAULT_DEPTH == 15


def test_address_to_field_int_matches_ts_docstrings():
    assert fl.address_to_field_int(A) == A_FIELD
    assert fl.address_to_field_int(B) == B_FIELD
    assert fl.address_to_field_int(ZERO) == 0


def test_hash_two_uses_poseidon4_over_packed_array():
    from aleo import mainnet as net
    expected = str(net.Poseidon4().hash(net.Plaintext.from_string("[1field,3field,4field]").to_fields()))
    assert fl.hash_two("1field", "3field", "4field", "mainnet") == expected
    assert expected.endswith("field") and expected != "0field"
    assert fl.hash_two("0field", "3field", "4field", "mainnet") != expected      # prefix matters (leaf vs node)


def test_empty_tree_root_pinned():
    # Pinned 2026-09-03 from the live compliance tree ([0, 0, root]); investigate rather than re-pin on failure.
    assert fl.build_tree(["0field", "0field"], "mainnet") == [0, 0, fl.EMPTY_TREE_ROOT]
    assert fl.EMPTY_TREE_ROOT == 3642222252059314292809609689035560016959342421640560347114299934615987159853


def test_generate_leaves_sorts_pads_and_filters_zero():
    assert fl.generate_leaves([A, B, B]) == ["0field", f"{B_FIELD}field", f"{B_FIELD}field", f"{A_FIELD}field"]
    assert fl.generate_leaves([ZERO, ZERO, A]) == ["0field", f"{A_FIELD}field"]
    assert fl.generate_leaves([]) == ["0field", "0field"]
    assert len(fl.generate_leaves([A] * 5)) == 8 and fl.generate_leaves([A] * 5)[:3] == ["0field"] * 3
    with pytest.raises(ConfigurationError, match="Leaves limit exceeded"):
        fl.generate_leaves([A] * (2 ** 14 + 1), 15)


def test_build_tree_shapes_and_errors():
    two = fl.build_tree(["1field", "2field"], "mainnet")
    four = fl.build_tree(["1field", "2field", "3field", "4field"], "mainnet")
    assert len(two) == 3 and two[:2] == [1, 2]
    assert len(four) == 7 and four[:4] == [1, 2, 3, 4]
    assert f"{four[4]}field" == fl.hash_two("1field", "1field", "2field", "mainnet")   # leaf level uses 1field
    assert f"{four[6]}field" == fl.hash_two("0field", f"{four[4]}field", f"{four[5]}field", "mainnet")  # inner level uses 0field
    with pytest.raises(ConfigurationError, match="cannot be empty"):
        fl.build_tree([], "mainnet")
    with pytest.raises(ConfigurationError, match="even number"):
        fl.build_tree(["1field", "2field", "3field"], "mainnet")


def test_leaf_indices_and_sibling_path():
    tree = fl.build_tree(fl.generate_leaves([A]), "mainnet")          # leaves [0, A_FIELD]
    assert fl.leaf_indices(tree, RECIPIENT) == (1, 1)                 # RECIPIENT's field > A_FIELD → clamps to the last leaf
    assert fl.leaf_indices(tree, A) == (0, 1)                         # equal to leaf 1 → (0, 1)
    four = fl.build_tree(["1field", "2field", "3field", "4field"], "mainnet")
    path = fl.sibling_path(four, 1, 15)
    assert len(path) == 15 and path[:3] == [2, 1, four[5]] and path[3:] == [0] * 12   # leaf, sibling, uncle, zero-padding
    assert len(fl.sibling_path(four, 1, 16)) == 16
    assert fl.format_merkle_proof(([2, 1], 1), ([3, 4], 2)) == \
        "[{ siblings: [2field, 1field], leaf_index: 1u32 }, { siblings: [3field, 4field], leaf_index: 2u32 }]"


def test_pure_exclusion_proof_of_empty_tree_equals_veil_literal():
    tree = fl.build_tree(["0field", "0field"], "mainnet")
    left, right = fl.leaf_indices(tree, RECIPIENT)
    assert (left, right) == (1, 1)
    proof = fl.format_merkle_proof((fl.sibling_path(tree, left, fl.PROOF_SIBLINGS), left),
                                   (fl.sibling_path(tree, right, fl.PROOF_SIBLINGS), right))
    assert proof == fl.EMPTY_MERKLE_PROOF_PAIR


def test_freezelist_reads_mappings_and_builds_proof(bridge):
    mappings = bridge.aleo.mappings[PROGRAM]
    assert bridge.freezelist.leaves(PROGRAM) == [] and bridge.freezelist.exclusion_proof(RECIPIENT, PROGRAM) == fl.EMPTY_MERKLE_PROOF_PAIR
    mappings["freeze_list_last_index"] = {"true": "1u32"}
    mappings["freeze_list"] = {"0u32": A, "1u32": ZERO}
    assert bridge.freezelist.leaves(PROGRAM) == [A]                    # zero address filtered
    assert bridge.freezelist.tree(PROGRAM)[:2] == [0, A_FIELD]
    one = "{ siblings: [" + f"{A_FIELD}field, 0field, " + ", ".join(["0field"] * 14) + "], leaf_index: 1u32 }"
    assert bridge.freezelist.exclusion_proof(RECIPIENT, PROGRAM) == f"[{one}, {one}]"
    with pytest.raises(ConfigurationError, match="freeze list"):
        bridge.freezelist.exclusion_proof(A, PROGRAM)


def test_freezelist_last_index_parsing_is_not_rstrip(bridge):
    mappings = bridge.aleo.mappings[PROGRAM]
    mappings["freeze_list_last_index"] = {"true": "12u32"}
    mappings["freeze_list"] = {f"{i}u32": ZERO for i in range(12)} | {"12u32": B}
    assert bridge.freezelist.leaves(PROGRAM) == [B]
