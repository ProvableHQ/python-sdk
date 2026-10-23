"""Program interface parsing, asset resolution, and record decoding.

_source: ProvableHQ/veil packages/core/test/inventory/assetDefaults.test.ts and
inventory.test.ts ("scanner-backed inventory"), PR #172 @ 2e324d5. Wallet
grant cases are not ported: the Python SDK only sees decrypted plaintext.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from aleo.inventory import (
    InventoryError, RecordAsset, RecordJoinProgram, TokenJoinRouter, decode_record, inventory_record_id,
    parse_program_interface, validate_asset, validate_record_join_program, validate_token_join_router,
)
from aleo.inventory.join_programs import (
    join_program_candidates, supported_join_program, validate_record_join_function, validate_token_join_function,
)

FIXTURES = Path(__file__).parent / "fixtures"
OWNER = "aleo1owner"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def program_source(program: str, record: str, fields: str, amount_type: str = "u64") -> str:
    return (
        f"program {program};\nrecord {record}:\n    owner as address.private;\n{fields}\n"
        f"function join:\n    input r0 as {record}.record;\n    input r1 as {record}.record;\n    output r2 as {record}.record;\n"
        f"function split:\n    input r0 as {record}.record;\n    input r1 as {amount_type}.private;\n"
        f"    output r2 as {record}.record;\n    output r3 as {record}.record;\n"
    )


def owned(program: str, plaintext: str, *, record_name: str | None = None, spent: bool = False, nonce: int = 1) -> dict[str, object]:
    return {"program_name": program, "record_name": record_name, "spent": spent, "tag": f"{nonce}field",
            "commitment": f"{nonce}field", "record_plaintext": plaintext}


def token_plaintext(amount: int, nonce: int, extra: str = "", field: str = "amount", width: str = "u128") -> str:
    return f"{{ owner: {OWNER}.private, {field}: {amount}{width}.private, {extra}_nonce: {nonce}group.public }}"


# ── interface parser ─────────────────────────────────────────────────────────

def test_parses_credits_interface_from_the_compiled_program() -> None:
    from aleo import mainnet
    interface = parse_program_interface(str(mainnet.Program.credits().source))
    assert interface.id == "credits.aleo"
    join = interface.functions["join"]
    assert [p.kind for p in join.inputs] == ["record", "record"] and join.inputs[0].type == "credits"
    split = interface.functions["split"]
    assert split.inputs[1].kind == "plaintext" and split.inputs[1].type == "u64" and split.inputs[1].visibility == "private"
    assert len(split.outputs) == 2
    assert interface.records["credits"].field("microcredits") is not None
    assert interface.records["credits"].field("owner") is not None
    # Finalize blocks declare inputs too; they must not leak into functions.
    assert all(p.kind != "plaintext" or p.visibility != "public" or f.name != "join" for f in interface.functions.values() for p in f.inputs)


def test_parses_the_dynamic_router_and_external_record_types() -> None:
    router = parse_program_interface(fixture("main_aj_arc20_2_15.aleo"))
    assert router.id == "main_aj_arc20_2_15.aleo"
    join_3 = router.functions["join_3"]
    assert join_3.inputs[0].kind == "plaintext" and join_3.inputs[0].type == "identifier" and join_3.inputs[0].visibility == "public"
    assert [p.kind for p in join_3.inputs[1:]] == ["dynamic_record"] * 3
    assert join_3.outputs[0].kind == "dynamic_record"
    static = parse_program_interface(
        "import credits.aleo;\nprogram autojoin_credits_2_10.aleo;\nfunction join_2:\n"
        "    input r0 as credits.aleo/credits.record;\n    input r1 as credits.aleo/credits.record;\n"
        "    output r2 as credits.aleo/credits.record;\n"
    )
    assert static.imports == ("credits.aleo",)
    assert static.functions["join_2"].inputs[0].kind == "external_record"
    assert static.functions["join_2"].inputs[0].type == "credits.aleo/credits"


# ── asset resolution ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("program,record,field,width,standard", [
    ("test_arc20_eth.aleo", "Token", "amount", "u128", "arc20"),
    ("test_usdcx_stablecoin.aleo", "Token", "amount", "u128", "arc22"),
])
def test_reads_deployed_tokens_without_a_standard(program: str, record: str, field: str, width: str, standard: str) -> None:
    capabilities = validate_asset(RecordAsset(program), fixture(program))
    assert capabilities.asset.record_name == record
    assert capabilities.asset.amount_field == field
    assert capabilities.asset.amount_type == width
    assert capabilities.asset.standard == standard


def test_credits_needs_no_source_and_rejects_overrides() -> None:
    capabilities = validate_asset(RecordAsset("credits.aleo"), None)
    assert capabilities.asset.record_name == "credits" and capabilities.asset.amount_field == "microcredits"
    with pytest.raises(InventoryError, match="credits record"):
        validate_asset(RecordAsset("credits.aleo", record_name="Token"), None)
    with pytest.raises(InventoryError, match="credits standard"):
        validate_asset(RecordAsset("credits.aleo", standard="arc20"), None)
    with pytest.raises(InventoryError, match="not deployed"):
        validate_asset(RecordAsset("missing.aleo"), None)


@pytest.mark.parametrize("fields,field,want", [
    ("    amount as u64.private;\n    microcredits as u64.private;", "amount", "u64"),
    ("    microcredits as u64.private;", "microcredits", "u64"),
])
def test_prefers_amount_and_falls_back_to_microcredits(fields: str, field: str, want: str) -> None:
    capabilities = validate_asset(RecordAsset("custom.aleo"), program_source("custom.aleo", "Balance", fields))
    assert capabilities.asset.record_name == "Balance"
    assert capabilities.asset.amount_field == field
    assert capabilities.asset.amount_type == want
    assert capabilities.asset.standard is None


def test_explicit_overrides_apply_and_a_missing_override_is_an_error() -> None:
    source = program_source("custom.aleo", "Balance", "    value as u64.private;\n    amount as u64.private;")
    asset = RecordAsset("custom.aleo", record_name="Balance", amount_field="value")
    assert validate_asset(asset, source).asset.amount_field == "value"
    with pytest.raises(InventoryError, match="amount field"):
        validate_asset(RecordAsset("custom.aleo", amount_field="missing"), source)


def test_selects_the_native_join_record_even_when_an_unrelated_token_record_exists() -> None:
    source = program_source("custom.aleo", "Balance", "    amount as u64.private;") + \
        "\nrecord Token:\n    owner as address.private;\n    amount as u128.private;\n"
    assert validate_asset(RecordAsset("custom.aleo"), source).asset.record_name == "Balance"


@pytest.mark.parametrize("declaration", ["token_id as field", "sender as address", "recipient as address"])
def test_excludes_records_declaring_compliance_or_registry_fields(declaration: str) -> None:
    source = program_source("custom.aleo", "Balance", f"    amount as u64.private;\n    {declaration}.private;")
    capabilities = validate_asset(RecordAsset("custom.aleo"), source)
    assert capabilities.excluded is True
    record = owned("custom.aleo", token_plaintext(5, 1, field="amount", width="u64"))
    assert decode_record(OWNER, RecordAsset("custom.aleo"), record, capabilities) is None


def test_rejects_an_incompatible_split_signature() -> None:
    source = program_source("custom.aleo", "Balance", "    amount as u64.private;", amount_type="u128")
    with pytest.raises(InventoryError, match="incompatible"):
        validate_asset(RecordAsset("custom.aleo"), source)


# ── record decoding ──────────────────────────────────────────────────────────

ARC20 = RecordAsset("test_arc20_eth.aleo")


def arc20_capabilities():  # type: ignore[no-untyped-def]
    return validate_asset(ARC20, fixture("test_arc20_eth.aleo"))


def test_decodes_owned_token_records_and_skips_others() -> None:
    capabilities = arc20_capabilities()
    record = decode_record(OWNER, ARC20, owned(ARC20.program, token_plaintext(10, 1)), capabilities)
    assert record is not None and record.amount == 10 and record.id == "1group" and record.commitment == "1field"
    assert decode_record(OWNER, ARC20, owned("other.aleo", token_plaintext(10, 2)), capabilities) is None
    assert decode_record(OWNER, ARC20, owned(ARC20.program, token_plaintext(10, 3), spent=True), capabilities) is None
    assert decode_record("aleo1other", ARC20, owned(ARC20.program, token_plaintext(10, 4)), capabilities) is None
    assert decode_record(OWNER, ARC20, owned(ARC20.program, token_plaintext(20, 5, "recipient_bound: true.private, ")), capabilities) is None
    compliance = owned(ARC20.program, token_plaintext(30, 6, f"sender: aleo1sender.private, recipient: {OWNER}.private, "), record_name="ComplianceRecord")
    assert decode_record(OWNER, ARC20, compliance, capabilities) is None
    assert decode_record(OWNER, ARC20, owned(ARC20.program, token_plaintext(0, 7)), capabilities) is None
    assert decode_record(OWNER, ARC20, owned(ARC20.program, token_plaintext(10, 8, width="u64")), capabilities) is None


def test_refuses_records_without_plaintext_instead_of_reporting_empty_inventory() -> None:
    capabilities = arc20_capabilities()
    with pytest.raises(InventoryError, match="plaintext"):
        decode_record(OWNER, ARC20, {"program_name": ARC20.program, "record_name": "Token"}, capabilities)
    with pytest.raises(InventoryError, match="plaintext"):
        inventory_record_id({"record_ciphertext": "record1..."})


def test_custom_u32_amount_field_decodes_with_its_declared_width() -> None:
    source = program_source("custom.aleo", "Balance", "    quantity as u32.private;", amount_type="u32")
    asset = RecordAsset("custom.aleo", record_name="Balance", amount_field="quantity")
    capabilities = validate_asset(asset, source)
    record = owned("custom.aleo", f"{{ owner: {OWNER}.private, quantity: 100u32.private, _nonce: 1group.public }}")
    decoded = decode_record(OWNER, asset, record, capabilities)
    assert decoded is not None and decoded.amount == 100


# ── join helpers ─────────────────────────────────────────────────────────────

def static_helper(program: str, asset: str, record: str, low: int, high: int) -> str:
    out = f"import {asset};\nprogram {program};\n"
    for n in range(low, high + 1):
        out += f"function join_{n}:\n" + "".join(f"    input r{j} as {asset}/{record}.record;\n" for j in range(n))
        out += f"    output r{n} as {asset}/{record}.record;\n\n"
    return out


def test_validates_helper_and_router_metadata() -> None:
    validate_record_join_program(RecordJoinProgram("autojoin_credits_2_10.aleo", "records", (3,)))
    with pytest.raises(InventoryError, match="helper"):
        validate_record_join_program(RecordJoinProgram("x.aleo", "records", (17,)))
    with pytest.raises(InventoryError, match="helper"):
        validate_record_join_program(RecordJoinProgram("x.aleo", "program_identifier_then_records", (16,)))
    assert validate_token_join_router(TokenJoinRouter("main_aj_arc20_2_15.aleo")) == 15
    with pytest.raises(InventoryError, match="2 to 15"):
        validate_token_join_router(TokenJoinRouter("main_aj_arc20_2_15.aleo", 16))


def test_candidates_follow_network_program_and_standard() -> None:
    credits = validate_asset(RecordAsset("credits.aleo"), None).asset
    usdcx = validate_asset(RecordAsset("test_usdcx_stablecoin.aleo"), fixture("test_usdcx_stablecoin.aleo")).asset
    eth = arc20_capabilities().asset
    assert [h.program for h in join_program_candidates("testnet", credits, None)] == [
        "autojoin_credits_2_10.aleo", "autojoin_credits_11_14.aleo", "autojoin_credits_15_16.aleo"]
    assert [h.program for h in join_program_candidates("testnet", usdcx, None)][0] == "aj_test_usdcx_stablecoin_2_10.aleo"
    assert [h.program for h in join_program_candidates("mainnet", eth, None)] == ["main_aj_arc20_2_15.aleo"]
    assert join_program_candidates("devnet", credits, None) == ()
    assert join_program_candidates("testnet", credits, False) == ()
    # An explicit router handles tokens; credits still get their helpers.
    assert join_program_candidates("testnet", eth, TokenJoinRouter("test_aj_arc20_2_15.aleo")) == ()
    assert join_program_candidates("testnet", credits, TokenJoinRouter("test_aj_arc20_2_15.aleo")) != ()


def test_helper_signature_checks_record_type_and_count() -> None:
    credits = validate_asset(RecordAsset("credits.aleo"), None).asset
    helper = RecordJoinProgram("autojoin_credits_2_10.aleo", "records", tuple(range(2, 11)))
    good = parse_program_interface(static_helper(helper.program, "credits.aleo", "credits", 2, 10))
    assert supported_join_program(helper, good, credits) == helper
    validate_record_join_function(helper, good, credits, 3)
    wrong = parse_program_interface(static_helper(helper.program, "other.aleo", "credits", 2, 10))
    assert supported_join_program(helper, wrong, credits) is None
    with pytest.raises(InventoryError, match="incompatible"):
        validate_record_join_function(helper, wrong, credits, 3)
    with pytest.raises(InventoryError, match="limits"):
        validate_record_join_function(helper, good, credits, 11)


def test_router_signature_checks_identifier_visibility_and_arity() -> None:
    router = TokenJoinRouter("test_aj_arc20_2_15.aleo")
    source = fixture("main_aj_arc20_2_15.aleo").replace("main_aj_arc20_2_15.aleo", router.program)
    validate_token_join_function(router, parse_program_interface(source), 4)
    with pytest.raises(InventoryError, match="incompatible"):
        validate_token_join_function(router, parse_program_interface(source.replace("identifier.public", "identifier.private")), 4)
    with pytest.raises(InventoryError, match="2 to 15"):
        validate_token_join_function(router, parse_program_interface(source), 16)
    eth = arc20_capabilities().asset
    dynamic = RecordJoinProgram(router.program, "program_identifier_then_records", tuple(range(2, 16)))
    assert supported_join_program(dynamic, parse_program_interface(source), eth) == dynamic
