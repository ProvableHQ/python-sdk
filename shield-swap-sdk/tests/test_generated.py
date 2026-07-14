"""Sanity tests for the committed aleo.codegen output (_generated.py)."""
from aleo_shield_swap import _generated as g


def test_program_id():
    assert g.PROGRAM_ID == "shield_swap_v3.aleo"


def test_slot_decode():
    slot = g.Slot.from_plaintext(
        "{ tick: 4055i32, tick_spacing: 60i32, sqrt_price: 22526123159817891330747538u128, "
        "fee_protocol: 0u8, liquidity: 183051202759u128, fee_growth_global0_x_64: 0u128, "
        "fee_growth_global1_x_64: 0u128, fee_residual0_x_64: 0u128, fee_residual1_x_64: 0u128, "
        "max_liquidity_per_tick: 1000u128, protocol_fees0: 0u128, protocol_fees1: 0u128, "
        "next_init_below: 3960i32, next_init_above: 4080i32 }")
    assert slot.tick == 4055 and slot.tick_spacing == 60
    assert slot.sqrt_price == 22526123159817891330747538


def test_mapping_decoder_table_covers_key_mappings():
    for name in ("slots", "pools", "swap_outputs", "fee_tiers", "used_blinded_addresses"):
        assert name in g.MAPPING_VALUE_DECODERS, name


def test_abi_constant_carries_key_types():
    assert g.ABI["program"] == "shield_swap_v3.aleo"
    slots = next(m for m in g.ABI["mappings"] if m["name"] == "slots")
    assert slots["key"] == {"Primitive": "Field"}


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
