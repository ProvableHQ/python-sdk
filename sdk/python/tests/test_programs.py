from pathlib import Path

from aleo.mainnet import Program, Process, PrivateKey, Identifier, ProgramID, Value, Authorization

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_source(name: str) -> str:
    return (FIXTURES / name).read_text()


def fixture_program(name: str) -> Program:
    return Program.from_source(fixture_source(name))


def test_credits_program_parse():
    p = Program.credits()
    assert str(p.id) == "credits.aleo"
    assert Identifier.from_string("transfer_public") in p.functions


def test_authorize_transfer_public():
    process = Process.load()
    pk = PrivateKey.random()
    recipient = str(pk.address)
    auth = process.authorize(
        pk,
        ProgramID.from_string("credits.aleo"),
        Identifier.from_string("transfer_public"),
        [Value.parse(f"{recipient}"), Value.parse("10u64")],
    )
    assert auth is not None
    # round-trip the authorization
    assert str(Authorization.from_json(auth.to_json())) == str(auth)


def test_function_inputs_external_struct_id():
    # the template's transfer_private takes [MerkleProof; 2u32] where MerkleProof
    # is imported from freezelist.aleo; struct_id must be the struct name, not the
    # program name
    p = fixture_program("compliant_token_template.aleo")
    array_input = p.get_function_inputs("transfer_private")[3]
    assert array_input["type"] == "array"
    assert array_input["element_type"]["struct_id"] == "MerkleProof"


class TestIsArc20:
    def test_compliant_program(self):
        assert fixture_program("arc20_token.aleo").is_arc20()

    def test_arc20_is_not_arc22(self):
        # transfer signatures differ between the two standards
        assert not fixture_program("arc20_token.aleo").is_arc22()

    def test_deployed_testnet_token(self):
        # https://testnet.explorer.provable.com/program/test_arc20_eth.aleo
        p = fixture_program("test_arc20_eth.aleo")
        assert p.is_arc20()
        assert not p.is_arc22()

    def test_credits_is_not_arc20(self):
        assert not Program.credits().is_arc20()

    def test_wrong_amount_type(self):
        src = fixture_source("arc20_token.aleo").replace("u128", "u64")
        assert not Program.from_source(src).is_arc20()

    def test_missing_function(self):
        src = fixture_source("arc20_token.aleo").replace("function join:", "function join_tokens:")
        assert not Program.from_source(src).is_arc20()

    def test_missing_all_views(self):
        src = fixture_source("arc20_token.aleo").split("view balance_of:")[0]
        assert not Program.from_source(src).is_arc20()

    def test_missing_one_view(self):
        src = fixture_source("arc20_token.aleo").replace("view symbol:", "view symbol_of:")
        assert not Program.from_source(src).is_arc20()

    def test_wrong_view_signature(self):
        src = fixture_source("arc20_token.aleo").replace(
            "view balance_of:\n    input r0 as address.public;",
            "view balance_of:\n    input r0 as field.public;",
        )
        assert not Program.from_source(src).is_arc20()

    def test_foreign_futures(self):
        # future outputs pointing at another program do not satisfy the interface
        src = fixture_source("arc20_token.aleo").replace("arc20_token.aleo/", "credits.aleo/")
        assert not Program.from_source(src).is_arc20()

    def test_public_record_owner(self):
        src = fixture_source("arc20_token.aleo").replace(
            "owner as address.private", "owner as address.public"
        )
        assert not Program.from_source(src).is_arc20()

    def test_wrong_record_name(self):
        src = fixture_source("arc20_token.aleo").replace("Token", "Coupon")
        assert not Program.from_source(src).is_arc20()


class TestIsArc22:
    def test_compliant_program(self):
        # the extra `token_id` entry on the Token record is permitted by the `..`
        assert fixture_program("arc22_token.aleo").is_arc22()

    def test_arc22_is_not_arc20(self):
        assert not fixture_program("arc22_token.aleo").is_arc20()

    def test_leo_compiled_template(self):
        # Leo-compiled template importing MerkleProof from freezelist.aleo
        p = fixture_program("compliant_token_template.aleo")
        assert p.is_arc22()
        assert not p.is_arc20()

    def test_deployed_testnet_near_miss(self):
        # matches all IARC22 function/record signatures but declares no view
        # functions, so it is not compliant
        # https://api.provable.com/v2/testnet/programs/test_usdcx_stablecoin.aleo
        p = fixture_program("test_usdcx_stablecoin.aleo")
        assert not p.is_arc22()
        assert not p.is_arc20()

    def test_credits_is_not_arc22(self):
        assert not Program.credits().is_arc22()

    def test_wrong_merkle_proof_shape(self):
        src = fixture_source("arc22_token.aleo").replace("[field; 16u32]", "[field; 8u32]")
        assert not Program.from_source(src).is_arc22()

    def test_missing_compliance_record(self):
        src = fixture_source("arc22_token.aleo").replace("ComplianceRecord", "AuditRecord")
        assert not Program.from_source(src).is_arc22()
