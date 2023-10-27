# -*- coding: utf-8 -*-
import aleo

def test_sanity():
    c_private_key = "APrivateKey1zkp3dQx4WASWYQVWKkq14v3RoQDfY2kbLssUj7iifi1VUQ6"
    c_view_key = "AViewKey1cxguxtKkjYnT9XDza9yTvVMxt6Ckb1Pv4ck1hppMzmCB"
    c_address = "aleo184vuwr5u7u0ha5f5k44067dd2uaqewxx6pe5ltha5pv99wvhfqxqv339h4"

    private_key = aleo.PrivateKey.from_string(c_private_key)
    
    assert str(private_key) == c_private_key
    assert str(private_key.view_key()) == c_view_key
    assert str(private_key.address()) == c_address

    view_key = aleo.ViewKey.from_string(c_view_key)
    assert str(view_key) == c_view_key
    assert view_key == private_key.view_key()

    address = aleo.Address.from_string(c_address)
    assert str(address) == c_address
    assert address == private_key.address()

def test_decrypt_success():
    c_plaintext = """{
  owner: aleo1j7qxyunfldj2lp8hsvy7mw5k8zaqgjfyr72x2gh3x4ewgae8v5gscf5jh3.private,
  microcredits: 1500000000000000u64.private,
  _nonce: 3077450429259593211617823051143573281856129402760267155982965992208217472983group.public
}"""
    c_ciphertext = "record1qyqsqpe2szk2wwwq56akkwx586hkndl3r8vzdwve32lm7elvphh37rsyqyxx66trwfhkxun9v35hguerqqpqzqrtjzeu6vah9x2me2exkgege824sd8x2379scspmrmtvczs0d93qttl7y92ga0k0rsexu409hu3vlehe3yxjhmey3frh2z5pxm5cmxsv4un97q"
    c_viewkey = "AViewKey1ccEt8A2Ryva5rxnKcAbn7wgTaTsb79tzkKHFpeKsm9NX"

    view_key = aleo.ViewKey.from_string(c_viewkey)
    ciphertext = aleo.RecordCiphertext.from_string(c_ciphertext)
    plaintext = view_key.decrypt(ciphertext)
    assert str(plaintext) == c_plaintext

def test_signature_verify():
    address = aleo.Address.from_string("aleo16u4ecz4yqq0udtnmsy8qzvj8emnua24n27c264f2t3unekdlpy8sh4hat2")
    c_signature = "sign1q366eqppwqvmsq0epddmkpqr7ul5rkkltewatf4wdwd82l5yhypdwfnrng6tkj3ryx36wz2dptfq4aev8pwl85u9u6fk48mwmqe35q7h3ptmdtcfxxlcc6ardzayk5ykn2xzp5mhv3spwl3ajgc3y8mfqdmqs7fq3w4wc6j65e3z9ttthqwfy570yef6l9f8klnskzsu9adquzsjwhw"
    signature = aleo.Signature.from_string(c_signature)
    message = bytes("asd", "utf-8")
    bad_message = bytes("qwe", "utf-8")
    assert signature.verify(address, message)
    assert not signature.verify(address, bad_message)
    assert signature == aleo.Signature.from_string(c_signature)

def test_account_sanity():
    private_key = aleo.PrivateKey.from_string("APrivateKey1zkp3dQx4WASWYQVWKkq14v3RoQDfY2kbLssUj7iifi1VUQ6")
    account = aleo.Account.from_private_key(private_key)
    assert account.private_key() == private_key
    assert account == aleo.Account.from_private_key(private_key)
    message = bytes("asd", "utf-8")
    bad_message = bytes("qwe", "utf-8")
    signature = account.sign(message)
    assert account.verify(signature, message)
    assert not account.verify(signature, bad_message)
    assert signature.verify(account.address(), message)

def test_coinbase():
    address = aleo.Address.from_string("aleo16xwtrvntrfnan84sy3qg2gdkkp5u5p7sjc882lx8n06fjx2k0yqsklw8sv")
    solution_json = "{\"partial_solution\":{\"address\":\"aleo16xwtrvntrfnan84sy3qg2gdkkp5u5p7sjc882lx8n06fjx2k0yqsklw8sv\",\"nonce\":5751994693410499959,\"commitment\":\"puzzle163g3gms8kle6z7pfrnelsxmt5qk88sycdxjrfd2chfrmcaa58uv28u4amjhhzyc08wr6ur2hjsusqvgm7mp\"},\"proof.w\":{\"x\":\"46184004058746376929865476153864114989216680475842020861467330568081354981230088442717116178378251337401583339204\",\"y\":\"183283507821413711045927236980084997259573867323884239590264843665205515176450368153011402822680772267880564185790\",\"infinity\":false}}"
    challenge_json = "{\"epoch_number\":233,\"epoch_block_hash\":\"ab15lsq2zxsvr0am25afrvnczglagu7utpzuzn2sp94f3vyefm4558quexrn3\",\"degree\":8191}"
    challenge = aleo.EpochChallenge.from_json(challenge_json)
    solution = aleo.ProverSolution.from_json(solution_json)

    assert solution.address() == address
    assert str(challenge) == challenge_json
    assert str(solution) == solution_json

    # Skip it because it takes too much time to load the puzzle
    # puzzle = aleo.CoinbasePuzzle.load()
    # verifying_key = puzzle.verifying_key()
    # assert solution.verify(verifying_key, challenge, 100)

if __name__ == "__main__":
    test_sanity()
    test_decrypt_success()
    test_signature_verify()
    test_account_sanity()
    test_coinbase()
