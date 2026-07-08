from aleo.mainnet import Program, Process, PrivateKey, Identifier, ProgramID, Value, Authorization


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
