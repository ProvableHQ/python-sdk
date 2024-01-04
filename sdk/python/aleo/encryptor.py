from __future__ import annotations

from . import PrivateKey, Ciphertext, Field, Network, Identifier, Plaintext, Literal


class Encryptor:
    """Class for encrypting and decrypting Aleo key material into ciphertext.
    """
    @staticmethod
    def encrypt_private_key_with_secret(private_key: PrivateKey, secret: str) -> Ciphertext:
        """Encrypts a private key into ciphertext using a secret.
        """
        seed = private_key.seed()
        return Encryptor.__encrypt_field(seed, secret, "private_key")

    @staticmethod
    def decrypt_private_key_with_secret(ciphertext: Ciphertext, secret: str) -> PrivateKey:
        """Decrypts a private key from ciphertext using a secret.
        """
        seed = Encryptor.__decrypt_field(ciphertext, secret, "private_key")
        return PrivateKey.from_seed(seed)

    @staticmethod
    # Encrypted a field element into a ciphertext representation
    def __encrypt_field(field: Field, secret: str, domain: str) -> Ciphertext:
        domain_f = Field.domain_separator(domain)
        secret_f = Field.domain_separator(secret)

        nonce = Field.random()
        blinding = Network.hash_psd2([domain_f, nonce, secret_f])
        key = blinding * field
        key_kv = (Identifier.from_string("key"),
                  Plaintext.new_literal(Literal.from_field(key)))
        nonce_kv = (Identifier.from_string("nonce"),
                  Plaintext.new_literal(Literal.from_field(nonce)))
        plaintext = Plaintext.new_struct([key_kv, nonce_kv])
        return plaintext.encrypt_symmetric(secret_f)

    @staticmethod
    def __extract_value(plaintext: Plaintext, identifier: str) -> Field:
        assert plaintext.is_struct()
        ident = Identifier.from_string(identifier)
        dec_map = plaintext.as_struct()
        val = dec_map[ident]
        assert val.is_literal()
        literal = val.as_literal()
        assert literal.type_name() == 'field'
        return Field.from_string(str(literal))

    @staticmethod
    # Recover a field element encrypted within ciphertext
    def __decrypt_field(ciphertext: Ciphertext, secret: str, domain: str) -> Field:
        domain_f = Field.domain_separator(domain)
        secret_f = Field.domain_separator(secret)
        decrypted = ciphertext.decrypt_symmetric(secret_f)
        assert decrypted.is_struct()
        recovered_key = Encryptor.__extract_value(decrypted, "key")
        recovered_nonce = Encryptor.__extract_value(decrypted, "nonce")
        recovered_blinding = Network.hash_psd2([domain_f, recovered_nonce, secret_f])
        return recovered_key / recovered_blinding

