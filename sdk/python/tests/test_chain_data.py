"""Tests for chain-data introspection: Transition, Execution, Authorization, Transaction.

Transition fixtures and KATs are vendored verbatim from the wasm SDK test suite
(wasm/src/ledger/transition.rs). The execution fixture (vectors/execution.json)
is a locally proven mainnet credits.aleo/transfer_public execution.

All tests are offline. The Authorization tests use the offline authorize
pattern (Process.authorize needs no network; first-ever run downloads a small
~3 MB SRS chunk to ~/.aleo).
"""

import json

import pytest

from aleo.mainnet import (
    Authorization,
    Execution,
    Field,
    Identifier,
    PrivateKey,
    Process,
    ProgramID,
    Transaction,
    Transition,
    Value,
)
from conftest import VECTORS

# ---------------------------------------------------------------------------
# Vendored KATs from wasm/src/ledger/transition.rs
# ---------------------------------------------------------------------------

INPUT_RECORD_SERIAL_NUMBER = "4569194627311410524427044648350523511369013276760031398859310110870190258038field"
INPUT_RECORD_TAG = "4584393733726099907383249165298083023636530416018938077800083356406243497342field"
OUTPUT_CHECKSUM = "17461704767783030875142836237730678349755524657182224909428747180538982740field"
OUTPUT_RECORD = "record1qyqspwnlv6gfxx05yj7aw7z2dl44gyh06jrvgf42jux0dep33cy7jlsvqsrxzmt0w4h8ggcqqgqsqwdwr889h9fhnyclazs8yt26t6r5ua4qk7yksj7p40rz9846mzgrpp6x76m9de0kjezrqqpqyq9sj8x3qdmz6nal4j470a0wwcray54lffe3ya5u2zlpeq45lg4up3na8gul0vgrn3eced6dka4ax2ja85xzds4pmqf8csrn8ku5cv3qz8m90p6x2unwv9k97ct4w35x7unf0fshg6t0de0hyet3w45hyetyyvqqyqgq8djhghnte2w86qsdjaumy4zcux2fxszm3ej2956af8cpl2w95g9pqct4w35x7unf0fjkghm4de6xjmprqqpqzqxd6c782j0ny65ed2ckzp3vlx7cv8drslasq8kqpdzmjeyzal38qemw38x0axnz5t53fj6ttavh8l4jlfjdryc6mesd4w6uvpmzfsqqjvtu0xd"
OUTPUT_RECORD_COMMITMENT = "3771264214823666953346974490700157125043441681812666085949968314967709800215field"
TRANSITION = r'{"id":"au1naeu56spz0x0zct003sa8qgpzndy6nxj8rrcm7n0fehy9llcl5yscflt0k","program":"token_registry.aleo","function":"burn_private","inputs":[{"type":"record","id":"4569194627311410524427044648350523511369013276760031398859310110870190258038field","tag":"4584393733726099907383249165298083023636530416018938077800083356406243497342field"},{"type":"public","id":"4155661860779318196369465902681808025430867777096367712868886959018716227815field","value":"2853086u128"}],"outputs":[{"type":"record","id":"3771264214823666953346974490700157125043441681812666085949968314967709800215field","checksum":"17461704767783030875142836237730678349755524657182224909428747180538982740field","value":"record1qyqspwnlv6gfxx05yj7aw7z2dl44gyh06jrvgf42jux0dep33cy7jlsvqsrxzmt0w4h8ggcqqgqsqwdwr889h9fhnyclazs8yt26t6r5ua4qk7yksj7p40rz9846mzgrpp6x76m9de0kjezrqqpqyq9sj8x3qdmz6nal4j470a0wwcray54lffe3ya5u2zlpeq45lg4up3na8gul0vgrn3eced6dka4ax2ja85xzds4pmqf8csrn8ku5cv3qz8m90p6x2unwv9k97ct4w35x7unf0fshg6t0de0hyet3w45hyetyyvqqyqgq8djhghnte2w86qsdjaumy4zcux2fxszm3ej2956af8cpl2w95g9pqct4w35x7unf0fjkghm4de6xjmprqqpqzqxd6c782j0ny65ed2ckzp3vlx7cv8drslasq8kqpdzmjeyzal38qemw38x0axnz5t53fj6ttavh8l4jlfjdryc6mesd4w6uvpmzfsqqjvtu0xd","sender_ciphertext":null},{"type":"future","id":"2177527202823505610844479366424698260670813913152550670302738921219693374616field","value":"{\n  program_id: token_registry.aleo,\n  function_name: burn_private,\n  arguments: [\n    3443843282313283355522573239085696902919850365217539366784739393210722344986field,\n    2853086u128,\n    aleo1tjkv7vquk6yldxz53ecwsy5csnun43rfaknpkjc97v5223dlnyxsglv7nm,\n    5783861720504029593520331872442756678068735468923730684279741068753131773333field\n  ]\n}"}],"tpk":"8426225807947287980879824833030089440060785195861154519084544916641544071836group","tcm":"3226339871444496417979841037237975848011574524309845233165930705339306709897field","scm":"6845182532650964173356391969005331370591444046632036068754797772530920467754field"}'
TEST_PRIVATE_KEY = "APrivateKey1zkp6rE5FSWGD3jxrsAT64aZutFs3w6xvF8uQzGZKJEKsN8j"
EXPECTED_TVK = "4386935145534748320784836619728244316439880324135120862336274251207085504468field"

# Mainnet decryption fixtures (hello_hello.aleo/main), signer key below.
TRANSITION_MAINNET = r'{"id":"au1mguuz0dh20f78802m4z0py7n08xhl0pz60llzck63mhl8pc8l5xqxpwgtn","program":"hello_hello.aleo","function":"main","inputs":[{"type":"public","id":"6393584049543470937057043098611271993206122889317039351966319038535020834557field","value": "1u32"},{"type":"private","id":"8207446256045172951742235001162005156507562935942883128759030124682934277495field","value":"ciphertext1qyqqgz9qnupeld9vr4vuwp6yrpmhgtkvmgag5m7mmrruw0r6je666qgqdswk3"}],"outputs":[{"type":"private","id":"127469473292952941321346770257126666363371158501875622169294663492714835110field","value":"ciphertext1qyqyapkjuxm9dcslgyjf7hkr2k3dek500z40gjspnwvll0uawj23vzgggc405"}],"tpk":"7647553513996966044119163122930125808381703910407273818947266861843062002251group","tcm":"4479413938380109857414238205380483440836495997450846894155088299187217672609field","scm":"6461007226176477784737642021400489186736987671609840640950580467598882134642field"}'
TRANSITION_MAINNET_DECRYPTED = r'{"id":"au1jl2ur42sj7hwe4r0alv6gnklqxj0fszrvu3q82gjcls5x6q9pyzqdgmu2k","program":"hello_hello.aleo","function":"main","inputs":[{"type":"public","id":"6393584049543470937057043098611271993206122889317039351966319038535020834557field","value":"1u32"},{"type":"public","id":"8207446256045172951742235001162005156507562935942883128759030124682934277495field","value":"2u32"}],"outputs":[{"type":"public","id":"127469473292952941321346770257126666363371158501875622169294663492714835110field","value":"3u32"}],"tpk":"7647553513996966044119163122930125808381703910407273818947266861843062002251group","tcm":"4479413938380109857414238205380483440836495997450846894155088299187217672609field","scm":"6461007226176477784737642021400489186736987671609840640950580467598882134642field"}'
DECRYPTION_PRIVATE_KEY = "APrivateKey1zkp8CZNn3yeCseEtxuVPbDCwSyhGW6yZKUYKfgXmcpoGPWH"


@pytest.fixture(scope="module")
def transition() -> Transition:
    return Transition.from_json(TRANSITION)


@pytest.fixture(scope="module")
def execution() -> Execution:
    return Execution.from_json((VECTORS / "execution.json").read_text())


@pytest.fixture(scope="module")
def process() -> Process:
    return Process.load()


def _authorize_transfer_public(process: Process, private_key: PrivateKey) -> Authorization:
    return process.authorize(
        private_key,
        ProgramID.from_string("credits.aleo"),
        Identifier.from_string("transfer_public"),
        [Value.parse(str(private_key.address)), Value.parse("10u64")],
    )


@pytest.fixture(scope="module")
def authorization(process) -> Authorization:
    return _authorize_transfer_public(process, PrivateKey.random())


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


class TestTransition:
    def test_from_json_roundtrip(self, transition):
        # serde drops null optional fields (e.g. "sender_ciphertext": null),
        # so compare parsed structures with nulls stripped, plus a bytes-level
        # roundtrip which is exact.
        def strip_nulls(obj):
            if isinstance(obj, dict):
                return {k: strip_nulls(v) for k, v in obj.items() if v is not None}
            if isinstance(obj, list):
                return [strip_nulls(x) for x in obj]
            return obj

        assert json.loads(transition.to_json()) == strip_nulls(json.loads(TRANSITION))
        assert Transition.from_json(transition.to_json()).bytes() == transition.bytes()

    def test_tpk_getter(self, transition):
        assert (
            str(transition.tpk)
            == "8426225807947287980879824833030089440060785195861154519084544916641544071836group"
        )

    def test_tcm_getter(self, transition):
        assert (
            str(transition.tcm)
            == "3226339871444496417979841037237975848011574524309845233165930705339306709897field"
        )

    def test_scm_getter(self, transition):
        assert (
            str(transition.scm)
            == "6845182532650964173356391969005331370591444046632036068754797772530920467754field"
        )

    def test_tvk(self, transition):
        pk = PrivateKey.from_string(TEST_PRIVATE_KEY)
        assert str(transition.tvk(pk.view_key)) == EXPECTED_TVK

    def test_records(self, transition):
        records = transition.records()
        assert len(records) == 1
        commitment, record_ciphertext = records[0]
        assert str(commitment) == OUTPUT_RECORD_COMMITMENT
        assert str(record_ciphertext) == OUTPUT_RECORD

    def test_find_record(self, transition):
        commitment = Field.from_string(OUTPUT_RECORD_COMMITMENT)
        rc = transition.find_record(commitment)
        assert rc is not None
        assert str(rc) == OUTPUT_RECORD
        assert transition.find_record(Field.random()) is None

    def test_contains_commitment(self, transition):
        assert transition.contains_commitment(Field.from_string(OUTPUT_RECORD_COMMITMENT))
        assert not transition.contains_commitment(Field.random())

    def test_contains_serial_number(self, transition):
        assert transition.contains_serial_number(Field.from_string(INPUT_RECORD_SERIAL_NUMBER))
        assert not transition.contains_serial_number(Field.random())

    def test_owned_records_random_vk_finds_none(self, transition):
        vk = PrivateKey.random().view_key
        assert len(transition.owned_records(vk)) == 0

    def test_inputs_structure(self, transition):
        inputs = transition.inputs()
        assert len(inputs) == 2
        assert inputs[0]["type"] == "record"
        assert inputs[0]["id"] == INPUT_RECORD_SERIAL_NUMBER
        assert inputs[0]["tag"] == INPUT_RECORD_TAG
        assert inputs[1]["type"] == "public"
        assert (
            inputs[1]["id"]
            == "4155661860779318196369465902681808025430867777096367712868886959018716227815field"
        )
        assert inputs[1]["value"] == "2853086u128"

    def test_outputs_structure(self, transition):
        outputs = transition.outputs()
        assert len(outputs) == 2

        assert outputs[0]["type"] == "record"
        assert outputs[0]["id"] == OUTPUT_RECORD_COMMITMENT
        assert outputs[0]["checksum"] == OUTPUT_CHECKSUM
        assert outputs[0]["value"] == OUTPUT_RECORD
        assert outputs[0]["sender_ciphertext"] is None

        assert outputs[1]["type"] == "future"
        assert outputs[1]["program"] == "token_registry.aleo"
        assert outputs[1]["function"] == "burn_private"
        arguments = outputs[1]["arguments"]
        assert len(arguments) == 4
        assert arguments[0] == (
            "3443843282313283355522573239085696902919850365217539366784739393210722344986field"
        )
        assert arguments[1] == "2853086u128"
        assert arguments[2] == "aleo1tjkv7vquk6yldxz53ecwsy5csnun43rfaknpkjc97v5223dlnyxsglv7nm"
        assert arguments[3] == (
            "5783861720504029593520331872442756678068735468923730684279741068753131773333field"
        )

    def test_decrypt_transition(self):
        pk = PrivateKey.from_string(DECRYPTION_PRIVATE_KEY)
        t = Transition.from_json(TRANSITION_MAINNET)
        tvk = t.tvk(pk.view_key)
        decrypted = t.decrypt_transition(tvk)
        assert json.loads(decrypted.to_json()) == json.loads(TRANSITION_MAINNET_DECRYPTED)

    def test_decrypt_transition_invalid_tvk(self):
        # A tvk derived from the wrong view key must fail decryption.
        wrong_pk = PrivateKey.from_string(TEST_PRIVATE_KEY)
        t = Transition.from_json(TRANSITION_MAINNET)
        invalid_tvk = t.tvk(wrong_pk.view_key)
        with pytest.raises(RuntimeError):
            t.decrypt_transition(invalid_tvk)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class TestExecution:
    def test_transitions(self, execution):
        transitions = execution.transitions()
        assert len(transitions) == 1
        assert str(transitions[0].program_id) == "credits.aleo"
        assert str(transitions[0].function_name) == "transfer_public"

    def test_global_state_root(self, execution):
        fixture = json.loads((VECTORS / "execution.json").read_text())
        assert execution.global_state_root == fixture["global_state_root"]
        assert execution.global_state_root.startswith("sr1")

    def test_proof(self, execution):
        fixture = json.loads((VECTORS / "execution.json").read_text())
        proof = execution.proof()
        assert proof is not None
        assert proof == fixture["proof"]
        assert proof.startswith("proof1")

    def test_execution_id(self, execution):
        assert str(execution.execution_id).endswith("field")


# ---------------------------------------------------------------------------
# Authorization
# ---------------------------------------------------------------------------


class TestAuthorization:
    def test_to_execution_id(self, authorization):
        eid = authorization.to_execution_id()
        assert isinstance(eid, Field)
        assert str(eid).endswith("field")
        # Deterministic for the same authorization.
        assert str(authorization.to_execution_id()) == str(eid)

    def test_transitions(self, authorization):
        transitions = authorization.transitions()
        assert len(transitions) == 1
        assert str(transitions[0].program_id) == "credits.aleo"
        assert str(transitions[0].function_name) == "transfer_public"

    def test_function_name(self, authorization):
        assert authorization.function_name() == "transfer_public"

    def test_is_fee_flags(self, authorization, process):
        assert authorization.is_fee_private() is False
        assert authorization.is_fee_public() is False
        assert authorization.is_split() is False

        # A fee_public authorization must report is_fee_public.
        fee_auth = process.authorize_fee_public(
            PrivateKey.random(), 1_000_000, 0, Field.random()
        )
        assert fee_auth.is_fee_public() is True
        assert fee_auth.is_fee_private() is False
        assert fee_auth.is_split() is False
        assert fee_auth.function_name() == "fee_public"

    def test_len(self, authorization):
        assert len(authorization) == 1

    def test_replicate(self, authorization):
        replica = authorization.replicate()
        assert replica.to_json() == authorization.to_json()
        assert str(replica.to_execution_id()) == str(authorization.to_execution_id())

    def test_insert_transition_duplicate_rejected(self, authorization):
        existing = authorization.transitions()[0]
        with pytest.raises(RuntimeError):
            authorization.insert_transition(existing)

    def test_insert_transition(self, process):
        auth = _authorize_transfer_public(process, PrivateKey.random())
        other = _authorize_transfer_public(process, PrivateKey.random())
        foreign_transition = other.transitions()[0]
        assert auth.insert_transition(foreign_transition) is None
        assert len(auth.transitions()) == 2

    def test_json_roundtrip(self, authorization):
        roundtripped = Authorization.from_json(authorization.to_json())
        assert roundtripped.to_json() == authorization.to_json()
        assert len(roundtripped) == len(authorization)


# ---------------------------------------------------------------------------
# Transaction (execute-type, built offline from the execution fixture)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def transaction(execution) -> Transaction:
    return Transaction.from_execution(execution, None)


class TestTransaction:
    def test_id_and_type(self, transaction):
        assert transaction.id.startswith("at1")
        assert transaction.transaction_type == "execute"
        assert transaction.is_execute() is True
        assert transaction.is_deploy() is False
        assert transaction.is_fee() is False

    def test_fee_amounts(self, transaction):
        # No fee was attached, so all fee amounts are zero.
        assert transaction.fee_amount == 0
        assert transaction.base_fee_amount == 0
        assert transaction.priority_fee_amount == 0

    def test_execution_and_transitions(self, transaction, execution):
        embedded = transaction.execution()
        assert embedded is not None
        assert str(embedded.execution_id) == str(execution.execution_id)
        transitions = transaction.transitions()
        assert len(transitions) == 1
        assert str(transitions[0].program_id) == "credits.aleo"

    def test_records_empty_for_public_transfer(self, transaction):
        assert transaction.records() == []
        assert len(transaction.owned_records(PrivateKey.random().view_key)) == 0

    def test_find_and_contains(self, transaction):
        random_field = Field.random()
        assert transaction.find_record(random_field) is None
        assert transaction.contains_commitment(random_field) is False
        assert transaction.contains_serial_number(random_field) is False

    def test_deploy_accessors_on_execute(self, transaction):
        assert transaction.deployed_program() is None
        assert transaction.verifying_keys() == []

    def test_summary(self, transaction):
        summary = transaction.summary()
        assert summary["id"] == transaction.id
        assert summary["type"] == "execute"
        assert summary["fee_amount"] == 0
        assert summary["base_fee"] == 0
        assert summary["priority_fee"] == 0

    def test_json_roundtrip(self, transaction):
        parsed = json.loads(transaction.to_json())
        assert parsed["type"] == "execute"
        roundtripped = Transaction.from_json(transaction.to_json())
        assert roundtripped == transaction
        assert roundtripped.id == transaction.id
