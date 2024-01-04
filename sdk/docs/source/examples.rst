Examples
=================

.. testsetup:: *

   import aleo

Working with accounts
*********************

.. doctest::

    >>> account = aleo.Account()
    >>> private_key = account.private_key()
    >>> secret = str(private_key)
    >>> restored = aleo.PrivateKey.from_string(secret)
    >>> same_account = aleo.Account.from_private_key(private_key)
    >>> assert account == same_account

Encrypted key materials
************************************

.. doctest::

    >>> ciphertext = aleo.Ciphertext.from_string(
    ... "ciphertext1qvqt0sp0pp49gjeh50alfalt7ug3g8y7ha6\
    ... cl3jkavcsnz8d0y9jwr27taxfrwd5kly8lah53qure3vxav\
    ... 6zxr7txattdvscv0kf3vcuqv9cmzj32znx4uwxdawcj3273\
    ... zhgm8qwpxqczlctuvjvc596mgsqjxwz37f")
    >>> decrypted = aleo.Encryptor.decrypt_private_key_with_secret(ciphertext, "qwe123")
    >>> account = aleo.Account.from_private_key(decrypted)
    >>> str(account)
    'aleo1w58eg85ckl76c0pzzf4mdg2y39t9t7jfvp9u2fvnj2a2t8aquqpqrlycqt'

Working with signatures
***********************

.. doctest::

    >>> account = aleo.Account()
    >>> message = b'Hello world'
    >>> signature = account.sign(message)
    >>> serialized = str(signature)
    >>> restored = aleo.Signature.from_string(serialized)
    >>> assert account.verify(restored, message)


Calling a **transfer_public** function
**************************************

.. doctest::
    :options: +ELLIPSIS

    >>> private_key = aleo.PrivateKey()
    >>> destination = aleo.Account().address()
    >>> amount = aleo.Credits(0.3)
    >>> query = aleo.Query.rest("https://explorer.hamp.app")
    >>> process = aleo.Process.load()
    >>> credits = aleo.Program.credits()
    >>> process.add_program(credits)
    >>> transfer_name = aleo.Identifier.from_string("transfer_public")
    >>> transfer_auth = process.authorize(private_key, credits.id(), transfer_name, [
    ...     aleo.Value.from_literal(aleo.Literal.from_address(destination)),
    ...     aleo.Value.from_literal(aleo.Literal.from_u64(
    ...         aleo.U64(int(amount.micro())))),
    ... ])
    >>> (_transfer_resp, transfer_trace) = process.execute(transfer_auth)
    >>> transfer_trace.prepare(query)
    >>> transfer_execution = transfer_trace.prove_execution(
    ...     aleo.Locator(credits.id(), aleo.Identifier.from_string("transfer")))
    >>> execution_id = transfer_execution.execution_id()
    >>> process.verify_execution(transfer_execution)

    >>> (fee_cost, _) = process.execution_cost(transfer_execution)
    >>> fee_priority = None
    >>> fee_auth = process.authorize_fee_public(
    ...     private_key, fee_cost, execution_id, fee_priority)
    >>> (_fee_resp, fee_trace) = process.execute(fee_auth)
    >>> fee_trace.prepare(query)
    >>> fee = fee_trace.prove_fee()
    >>> process.verify_fee(fee, execution_id)

    >>> transaction = aleo.Transaction.from_execution(transfer_execution, fee)
    >>> transaction.to_json()
    '{"type":"execute","id":"at...
