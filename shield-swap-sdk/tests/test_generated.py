"""Sanity tests for the committed aleo.codegen output (_generated.py)."""
from aleo_shield_swap import _generated as g


def test_program_id_is_new_core():
    assert g.PROGRAM_ID == "shield_swap.aleo"


def test_u256_roundtrip():
    u = g.U256__8JquwLopp8(hi=1, lo=17013693014354590797691252010145372)
    text = u.to_plaintext()
    assert text == "{ hi: 1u128, lo: 17013693014354590797691252010145372u128 }"
    assert g.U256__8JquwLopp8.from_plaintext(text) == u


def test_merkle_proof_roundtrip():
    p = g.MerkleProof(siblings=["0field"] * 16, leaf_index=1)
    assert g.MerkleProof.from_plaintext(p.to_plaintext()) == p


def test_pool_state_has_no_scales():
    names = set(g.PoolState.__dataclass_fields__)
    assert names == {"token0", "token1", "fee", "enabled"}


def test_slot_decode_with_u256_prices():
    slot = g.Slot.from_plaintext(
        "{ tick: 4055i32, tick_spacing: 60u32, "
        "sqrt_price: { hi: 1u128, lo: 22526123159817891330747538u128 }, "
        "fee_protocol: 0u8, liquidity: 183051202759u128, "
        "fee_growth_global0_x_128: { hi: 0u128, lo: 0u128 }, "
        "fee_growth_global1_x_128: { hi: 0u128, lo: 0u128 }, "
        "max_liquidity_per_tick: 1000u128, protocol_fees0: 0u128, "
        "protocol_fees1: 0u128, next_init_below: 3960i32, next_init_above: 4080i32 }")
    assert slot.tick == 4055 and slot.tick_spacing == 60
    assert slot.sqrt_price == g.U256__8JquwLopp8(hi=1, lo=22526123159817891330747538)
    assert "fee_residual0_x_64" not in g.Slot.__dataclass_fields__


def test_position_nft_has_withdrawal():
    assert "withdrawal" in g.PositionNFT.__dataclass_fields__


def test_mapping_decoder_table_covers_key_mappings():
    for name in ("slots", "pools", "swap_outputs", "fee_tiers",
                 "used_blinded_addresses", "from_wrapper_token_id"):
        assert name in g.MAPPING_VALUE_DECODERS, name


def test_abi_constant_carries_key_types():
    assert g.ABI["program"] == "shield_swap.aleo"
    slots = next(m for m in g.ABI["mappings"] if m["name"] == "slots")
    assert slots["key"] == {"Primitive": "Field"}


def test_removed_functions_gone():
    names = {fn["name"] for fn in g.ABI["functions"]}
    assert "claim_multi_hop_output" not in names
    assert "set_token_decimals" not in names
    assert {"swap", "claim_swap_output", "mint", "collect", "allow_token"} <= names


def test_mint_request_encodes():
    req_kwargs = {}
    for f in g.MintPositionRequest.__dataclass_fields__.values():
        if f.type is bool:
            req_kwargs[f.name] = True
        elif f.type is int:
            req_kwargs[f.name] = 1
        else:
            req_kwargs[f.name] = "1field"
    text = g.MintPositionRequest(**req_kwargs).to_plaintext()
    assert text.startswith("{ ") and text.endswith(" }")
    assert g.MintPositionRequest.from_plaintext(text) == g.MintPositionRequest(**req_kwargs)


def test_merkle_proof_rejects_the_wrong_number_of_siblings():
    # `[field; 16]` both ways: a 15-sibling proof is not a MerkleProof.
    import pytest
    short = "{ siblings: [" + ", ".join(["0field"] * 15) + "], leaf_index: 1u32 }"
    with pytest.raises(ValueError, match="length 16"):
        g.MerkleProof.from_plaintext(short)
    with pytest.raises(ValueError, match="length 16"):
        g.MerkleProof(siblings=["0field"] * 15, leaf_index=1).to_plaintext()
