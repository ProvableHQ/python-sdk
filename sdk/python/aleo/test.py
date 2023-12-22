# -*- coding: utf-8 -*-
import unittest
import aleo

class TestAleo(unittest.TestCase):

    def test_sanity(self):
        c_private_key = "APrivateKey1zkp3dQx4WASWYQVWKkq14v3RoQDfY2kbLssUj7iifi1VUQ6"
        c_view_key = "AViewKey1cxguxtKkjYnT9XDza9yTvVMxt6Ckb1Pv4ck1hppMzmCB"
        c_address = "aleo184vuwr5u7u0ha5f5k44067dd2uaqewxx6pe5ltha5pv99wvhfqxqv339h4"

        private_key = aleo.PrivateKey.from_string(c_private_key)

        self.assertEqual(str(private_key), c_private_key)
        self.assertEqual(str(private_key.view_key()), c_view_key)
        self.assertEqual(str(private_key.address()), c_address)

        view_key = aleo.ViewKey.from_string(c_view_key)
        self.assertEqual(str(view_key), c_view_key)
        self.assertEqual(view_key, private_key.view_key())

        address = aleo.Address.from_string(c_address)
        self.assertEqual(str(address), c_address)
        self.assertEqual(address, private_key.address())

    def test_decrypt_success(self):
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

        self.assertEqual(str(plaintext), c_plaintext)

    def test_signature_verify(self):
        address = aleo.Address.from_string(
            "aleo16u4ecz4yqq0udtnmsy8qzvj8emnua24n27c264f2t3unekdlpy8sh4hat2")
        c_signature = "sign1q366eqppwqvmsq0epddmkpqr7ul5rkkltewatf4wdwd82l5yhypdwfnrng6tkj3ryx36wz2dptfq4aev8pwl85u9u6fk48mwmqe35q7h3ptmdtcfxxlcc6ardzayk5ykn2xzp5mhv3spwl3ajgc3y8mfqdmqs7fq3w4wc6j65e3z9ttthqwfy570yef6l9f8klnskzsu9adquzsjwhw"
        signature = aleo.Signature.from_string(c_signature)
        message = bytes("asd", "utf-8")
        bad_message = bytes("qwe", "utf-8")
        self.assertTrue(signature.verify(address, message))
        self.assertFalse(signature.verify(address, bad_message))
        self.assertEqual(signature, aleo.Signature.from_string(c_signature))

    def test_account_sanity(self):
        private_key = aleo.PrivateKey.from_string(
            "APrivateKey1zkp3dQx4WASWYQVWKkq14v3RoQDfY2kbLssUj7iifi1VUQ6")
        account = aleo.Account.from_private_key(private_key)

        self.assertEqual(account.private_key(), private_key)
        self.assertEqual(account, aleo.Account.from_private_key(private_key))

        message = bytes("asd", "utf-8")
        bad_message = bytes("qwe", "utf-8")
        signature = account.sign(message)

        self.assertTrue(account.verify(signature, message))
        self.assertFalse(account.verify(signature, bad_message))
        self.assertTrue(signature.verify(account.address(), message))

    def test_encrypt_decrypt_sk(self):
        private_key = aleo.PrivateKey.from_string(
            "APrivateKey1zkpJYx2NZeJYB74JHpzvQGpKneTP75Dk8dao6paugZXtCz3")
        ciphertext = aleo.Ciphertext.from_string(
            "ciphertext1qvqt0sp0pp49gjeh50alfalt7ug3g8y7ha6cl3jkavcsnz8d0y9jwr27taxfrwd5kly8lah53qure3vxav6zxr7txattdvscv0kf3vcuqv9cmzj32znx4uwxdawcj3273zhgm8qwpxqczlctuvjvc596mgsqjxwz37f")
        recovered = Encryptor.decrypt_private_key_with_secret(ciphertext, "qwe123")

        self.assertEqual(private_key, recovered)

        encrypted = Encryptor.encrypt_private_key_with_secret(private_key, "asd123")
        other_recovered = Encryptor.decrypt_private_key_with_secret(encrypted, "asd123")

        self.assertEqual(private_key, other_recovered)

    def test_coinbase(self):
        address = aleo.Address.from_string(
            "aleo16xwtrvntrfnan84sy3qg2gdkkp5u5p7sjc882lx8n06fjx2k0yqsklw8sv")
        solution_json = "{\"partial_solution\":{\"address\":\"aleo16xwtrvntrfnan84sy3qg2gdkkp5u5p7sjc882lx8n06fjx2k0yqsklw8sv\",\"nonce\":5751994693410499959,\"commitment\":\"puzzle163g3gms8kle6z7pfrnelsxmt5qk88sycdxjrfd2chfrmcaa58uv28u4amjhhzyc08wr6ur2hjsusqvgm7mp\"},\"proof.w\":{\"x\":\"46184004058746376929865476153864114989216680475842020861467330568081354981230088442717116178378251337401583339204\",\"y\":\"183283507821413711045927236980084997259573867323884239590264843665205515176450368153011402822680772267880564185790\",\"infinity\":false}}"
        challenge_json = "{\"epoch_number\":233,\"epoch_block_hash\":\"ab15lsq2zxsvr0am25afrvnczglagu7utpzuzn2sp94f3vyefm4558quexrn3\",\"degree\":8191}"
        challenge = aleo.EpochChallenge.from_json(challenge_json)
        solution = aleo.ProverSolution.from_json(solution_json)

        self.assertEqual(solution.address(), address)
        self.assertEqual(str(challenge), challenge_json)
        self.assertEqual(str(solution), solution_json)

        puzzle = aleo.CoinbasePuzzle.load()
        verifying_key = puzzle.verifying_key()
        assert solution.verify(verifying_key, challenge, 100)

    def test_transfer(self):
        private_key = aleo.PrivateKey.from_string(
            "APrivateKey1zkp3dQx4WASWYQVWKkq14v3RoQDfY2kbLssUj7iifi1VUQ6")
        destination = aleo.Address.from_string(
            "aleo16u4ecz4yqq0udtnmsy8qzvj8emnua24n27c264f2t3unekdlpy8sh4hat2")
        amount = aleo.Credits(0.3)
        query = aleo.Query.rest("https://explorer.hamp.app")
        process = aleo.Process.load()
        credits = aleo.Program.credits()
        process.add_program(credits)
        transfer_name = aleo.Identifier.from_string("transfer_public")
        transfer_auth = process.authorize(private_key, credits.id(), transfer_name, [
            aleo.Value.from_literal(aleo.Literal.from_address(destination)),
            aleo.Value.from_literal(aleo.Literal.from_u64(
                aleo.U64(int(amount.micro())))),
        ])
        (_transfer_resp, transfer_trace) = process.execute(transfer_auth)
        transfer_trace.prepare(query)
        transfer_execution = transfer_trace.prove_execution(
            aleo.Locator(credits.id(), aleo.Identifier.from_string("transfer")))
        execution_id = transfer_execution.execution_id()
        process.verify_execution(transfer_execution)

        (fee_cost, _) = process.execution_cost(transfer_execution)
        fee_priority = None
        fee_auth = process.authorize_fee_public(
            private_key, fee_cost, execution_id, fee_priority)
        (_fee_resp, fee_trace) = process.execute(fee_auth)
        fee_trace.prepare(query)
        fee = fee_trace.prove_fee()
        process.verify_fee(fee, execution_id)

        transaction = aleo.Transaction.from_execution(transfer_execution, fee)
        print(transaction.to_json())


if __name__ == "__main__":
    unittest.main()
