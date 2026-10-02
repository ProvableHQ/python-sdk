"""Privy and Dynamic server-wallet signers, hermetically: fake provider clients sign with local keys
so every signature is real and every check the adapters make (sender recovery, message equality,
ed25519 verification, identity resolution) runs against genuine material.

The fakes speak the provider SDKs' Python surface (``client.wallets.ethereum.sign_transaction``,
``client.wallets.solana.sign_transaction``, async ``authenticate_api_token`` / ``load_wallet`` /
``sign_transaction``) and can misbehave on demand — a wrong key, a changed message, a zero
signature, an exception — so the tests prove what reaches the wire and what never does.
"""
from __future__ import annotations

import base64
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from eth_account import Account
from solders.hash import Hash
from solders.keypair import Keypair
from solders.message import MessageV0, to_bytes_versioned
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from aleo_bridge import _remote_signing as remote
from aleo_bridge.dynamic import DynamicEvmSigner, DynamicSolanaSigner
from aleo_bridge.errors import BridgeError, ConfigurationError
from aleo_bridge.eth import Ethereum
from aleo_bridge.privy import PrivyEvmSigner, PrivySolanaSigner
from aleo_bridge.sol import SolModule, Solana
from tests.fakes.fake_solana import FakeSolanaClient, stub_bridge
from tests.fakes.fake_web3 import fake_web3
from tests.fakes.sealevel_fixtures import TRANSFER

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

EVM_KEY = "0x" + "11" * 32
EVM_ACCOUNT = Account.from_key(EVM_KEY)
OTHER_ACCOUNT = Account.from_key("0x" + "22" * 32)
TO = "0x0000000000000000000000000000000000000002"
RECIPIENT = TRANSFER["recipientAleoAddress"]


# ── fakes ────────────────────────────────────────────────────────────────────

class FakePrivyClient:
    """``privy.PrivyClient`` as the signers see it: ``wallets.get`` plus the two signing services."""

    def __init__(self, *, evm_account: Any = EVM_ACCOUNT, sol_keypair: Keypair | None = None, mode: str = "ok") -> None:
        self.evm_account = evm_account
        self.sol_keypair = sol_keypair or Keypair()
        self.mode = mode
        self.calls: list[tuple[str, Any]] = []
        self.wallets = SimpleNamespace(get=self._get, ethereum=SimpleNamespace(sign_transaction=self._sign_evm),
                                       solana=SimpleNamespace(sign_transaction=self._sign_sol))

    def _get(self, *, wallet_id: str) -> Any:
        self.calls.append(("get", wallet_id))
        if wallet_id == "evm-wallet":
            return SimpleNamespace(id=wallet_id, chain_type="ethereum", address=self.evm_account.address.lower())
        if wallet_id == "sol-wallet":
            return SimpleNamespace(id=wallet_id, chain_type="solana", address=str(self.sol_keypair.pubkey()))
        return SimpleNamespace(id=wallet_id, chain_type="ethereum", address=OTHER_ACCOUNT.address)

    def _sign_evm(self, wallet_id: str, *, params: dict, address: str | None = None, request_options: Any = None) -> Any:
        self.calls.append(("eth_signTransaction", params, address, request_options))
        if self.mode == "error":
            raise RuntimeError("remote policy rejected signing")
        t = params["transaction"]
        tx: dict[str, Any] = {"chainId": t["chain_id"], "nonce": t["nonce"], "to": t["to"], "data": t["data"],
                              "value": t["value"], "gas": t["gas_limit"]}
        if t["type"] == 0:
            tx["gasPrice"] = t["gas_price"]
        else:
            tx.update({"maxFeePerGas": t["max_fee_per_gas"], "maxPriorityFeePerGas": t["max_priority_fee_per_gas"]})
        account = OTHER_ACCOUNT if self.mode == "wrong-key" else self.evm_account
        if self.mode == "tampered":                      # the right wallet signs, but not what was asked
            tx.update({"to": OTHER_ACCOUNT.address, "value": tx["value"] + 1})
        signed = account.sign_transaction(tx)
        return SimpleNamespace(signed_transaction="0x" + bytes(signed.raw_transaction).hex(), encoding="rlp")

    def _sign_sol(self, wallet_id: str, transaction: bytes, *, address: str | None = None, request_options: Any = None) -> Any:
        self.calls.append(("signTransaction", bytes(transaction), address, request_options))
        if self.mode == "error":
            raise RuntimeError("remote policy rejected signing")
        tx = VersionedTransaction.from_bytes(bytes(transaction))
        message = tx.message
        if self.mode == "changed-message":
            message = MessageV0(message.header, list(message.account_keys), Hash.new_unique(),
                                list(message.instructions), list(message.address_table_lookups))
        keys = list(message.account_keys)
        index = keys.index(self.sol_keypair.pubkey())
        # A provider response omits the signatures it did not make; the bridge must keep its own.
        signatures = [Signature.default()] * int(message.header.num_required_signatures)
        signatures[index] = (Signature.from_bytes(bytes(64)) if self.mode == "bad-signature"
                             else self.sol_keypair.sign_message(to_bytes_versioned(message)))
        wire = bytes(VersionedTransaction.populate(message, signatures))
        return SimpleNamespace(signed_transaction=base64.b64encode(wire).decode("ascii"), encoding="base64")


class FakeDynamicClient:
    """``dynamic_wallet_sdk`` wallet client as the signers see it (async, address-keyed)."""

    def __init__(self, *, chain_name: str, evm_account: Any = EVM_ACCOUNT, sol_keypair: Keypair | None = None,
                 wallet_id: str = "dyn-wallet", mode: str = "ok", v_style: str = "eip155", echo_address: str | None = None) -> None:
        self.chain_name = chain_name
        self.evm_account = evm_account
        self.sol_keypair = sol_keypair or Keypair()
        self.wallet_id = wallet_id
        self.mode = mode
        self.v_style = v_style
        self.echo_address = echo_address
        self.calls: list[tuple[str, Any]] = []
        self.closed = False

    @property
    def address(self) -> str:
        return self.evm_account.address if self.chain_name == "EVM" else str(self.sol_keypair.pubkey())

    async def authenticate_api_token(self, token: str) -> None:
        self.calls.append(("authenticate_api_token", token))

    async def load_wallet(self, address: str) -> Any:
        self.calls.append(("load_wallet", address))
        return SimpleNamespace(chain_name=self.chain_name, wallet_id=self.wallet_id,
                               account_address=self.echo_address or address)

    async def sign_transaction(self, address: str, tx: Any, password: str | None = None, key_shares: Any = None,
                               mfa_token: str | None = None) -> str:
        self.calls.append(("sign_transaction", address, tx, password, key_shares))
        if self.mode == "error":
            raise RuntimeError("remote policy rejected signing")
        if self.chain_name == "EVM":
            account = OTHER_ACCOUNT if self.mode == "wrong-key" else self.evm_account
            signed = account.sign_transaction(dict(tx))
            v = int(signed.v)
            recid = v - 35 - 2 * int(tx["chainId"])
            v = {"eip155": v, "recid": recid, "27": 27 + recid}[self.v_style]
            return "0x" + int(signed.r).to_bytes(32, "big").hex() + int(signed.s).to_bytes(32, "big").hex() + f"{v:02x}"
        if self.mode == "bad-signature":
            return "00" * 64
        return bytes(self.sol_keypair.sign_message(bytes(tx))).hex()

    async def close(self) -> None:
        self.closed = True


def privy_evm(mode: str = "ok") -> tuple[PrivyEvmSigner, FakePrivyClient]:
    client = FakePrivyClient(mode=mode)
    return PrivyEvmSigner(client, wallet_id="evm-wallet", address=EVM_ACCOUNT.address.lower()), client


def dynamic_evm(mode: str = "ok", **kwargs: Any) -> tuple[DynamicEvmSigner, FakeDynamicClient]:
    client = FakeDynamicClient(chain_name="EVM", mode=mode, **kwargs)
    signer = DynamicEvmSigner(client, address=EVM_ACCOUNT.address.lower(), api_token="dyn_token", password="pw")
    return signer, client


def privy_sol(mode: str = "ok") -> tuple[PrivySolanaSigner, FakePrivyClient]:
    client = FakePrivyClient(mode=mode)
    return PrivySolanaSigner(client, wallet_id="sol-wallet", address=str(client.sol_keypair.pubkey())), client


def dynamic_sol(mode: str = "ok", **kwargs: Any) -> tuple[DynamicSolanaSigner, FakeDynamicClient]:
    client = FakeDynamicClient(chain_name="SVM", mode=mode, **kwargs)
    signer = DynamicSolanaSigner(client, address=str(client.sol_keypair.pubkey()), api_token="dyn_token", password="pw")
    return signer, client


EVM_SIGNERS = {"privy": privy_evm, "dynamic": dynamic_evm}
SOL_SIGNERS = {"privy": privy_sol, "dynamic": dynamic_sol}


def sign_calls(client: Any) -> list[Any]:
    return [c for c in client.calls if c[0] in ("eth_signTransaction", "signTransaction", "sign_transaction")]


# ── EVM ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("provider", ["privy", "dynamic"])
def test_evm_signer_signs_remotely_and_broadcasts_through_the_connection(provider: str) -> None:
    signer, client = EVM_SIGNERS[provider]()
    w3 = fake_web3(chain_id=1)
    conn = Ethereum(w3=w3, signer=signer)
    assert conn.address == EVM_ACCOUNT.address and conn.can_sign
    assert sign_calls(client) == [] and w3.provider.sent == []              # construction contacts nothing
    produced: list[remote.RemoteSignedTransaction] = []
    original = signer.sign_transaction
    signer.sign_transaction = lambda tx: produced.append(original(tx)) or produced[-1]  # type: ignore[method-assign]

    tx_hash = conn.send_transaction({"to": TO, "data": "0x1234", "value": 123})

    assert len(sign_calls(client)) == 1 and len(produced) == 1
    raw = produced[0].raw_transaction
    assert Account.recover_transaction(raw) == EVM_ACCOUNT.address
    assert "0x" + produced[0].hash.hex() == tx_hash == w3.provider.sent[0]["hash"]
    sent = w3.provider.sent[0]
    assert (sent["from"], sent["to"], sent["value"], sent["data"]) == (EVM_ACCOUNT.address, TO, 123, "0x1234")
    # Dynamic can only sign legacy transactions; Privy gets the connection's usual EIP-1559 form.
    assert (raw[0] == 2) == (provider == "privy")


@pytest.mark.parametrize("provider", ["privy", "dynamic"])
def test_evm_signer_rejects_a_mismatched_sender_before_requesting_a_signature(provider: str) -> None:
    signer, client = EVM_SIGNERS[provider]()
    w3 = fake_web3(chain_id=1)
    with pytest.raises(ConfigurationError, match="does not match"):
        Ethereum(w3=w3, signer=signer).send_transaction({"from": TO, "to": TO, "data": "0x"})
    with pytest.raises(ConfigurationError, match="does not match"):
        signer.sign_transaction({"from": TO, "to": TO, "data": "0x", "chainId": 1, "nonce": 0, "gas": 21000, "gasPrice": 1})
    assert sign_calls(client) == [] and w3.provider.sent == []


@pytest.mark.parametrize("provider", ["privy", "dynamic"])
def test_evm_signer_refuses_a_signature_from_another_key_without_broadcasting(provider: str) -> None:
    signer, client = EVM_SIGNERS[provider]("wrong-key")
    w3 = fake_web3(chain_id=1)
    with pytest.raises(BridgeError, match="does not recover to the configured wallet"):
        Ethereum(w3=w3, signer=signer).send_transaction({"to": TO, "data": "0x", "value": 1})
    assert len(sign_calls(client)) == 1 and w3.provider.sent == []


def test_privy_evm_signer_refuses_a_correctly_signed_but_different_transaction() -> None:
    signer, client = privy_evm("tampered")
    w3 = fake_web3(chain_id=1)
    with pytest.raises(BridgeError, match=r"different transaction than requested \(to, value differ\)"):
        Ethereum(w3=w3, signer=signer).send_transaction({"to": TO, "data": "0x", "value": 1})
    assert len(sign_calls(client)) == 1 and w3.provider.sent == []


@pytest.mark.parametrize("envelope", ["legacy", "eip1559"])
@pytest.mark.parametrize("field,other", [("nonce", 9), ("chainId", 5), ("gas", 30000), ("data", "0xdead"),
                                         ("value", 7), ("fee", None)])
def test_verified_evm_transaction_compares_every_requested_field(envelope: str, field: str, other: Any) -> None:
    fee = {"gasPrice": 10**9} if envelope == "legacy" else {"maxFeePerGas": 2 * 10**9, "maxPriorityFeePerGas": 10**8}
    requested = {"to": TO, "value": 1, "data": "0x1234", "chainId": 1, "nonce": 3, "gas": 21000, **fee}
    signed_as = dict(requested)
    if field == "fee":
        signed_as.update({"gasPrice": 10**9 + 1} if envelope == "legacy" else {"maxFeePerGas": 3 * 10**9})
    else:
        signed_as[field] = other
    raw = bytes(EVM_ACCOUNT.sign_transaction(signed_as).raw_transaction)
    with pytest.raises(BridgeError, match="different transaction than requested"):
        remote.verified_evm_transaction(raw, sender=EVM_ACCOUNT.address, fields=requested, provider="Test")
    honest = bytes(EVM_ACCOUNT.sign_transaction(requested).raw_transaction)
    assert remote.verified_evm_transaction(honest, sender=EVM_ACCOUNT.address, fields=requested, provider="Test").raw_transaction == honest
    decoded = remote.decode_signed_evm_transaction(honest)
    assert decoded["to"] == TO and decoded["chainId"] == 1 and decoded["data"] == "0x1234"
    assert ("gasPrice" in decoded) == (envelope == "legacy")


@pytest.mark.parametrize("provider", ["privy", "dynamic"])
def test_evm_signing_failure_propagates_without_retry_or_broadcast(provider: str) -> None:
    signer, client = EVM_SIGNERS[provider]("error")
    w3 = fake_web3(chain_id=1)
    with pytest.raises(RuntimeError, match="remote policy"):
        Ethereum(w3=w3, signer=signer).send_transaction({"to": TO, "data": "0x", "value": 1})
    assert len(sign_calls(client)) == 1 and w3.provider.sent == []


def test_evm_signers_need_the_fields_the_connection_fills() -> None:
    signer, client = privy_evm()
    with pytest.raises(ConfigurationError, match="nonce, gas"):
        signer.sign_transaction({"to": TO, "chainId": 1, "gasPrice": 1})
    with pytest.raises(ConfigurationError, match="gasPrice or maxFeePerGas"):
        signer.sign_transaction({"to": TO, "chainId": 1, "nonce": 0, "gas": 21000})
    assert sign_calls(client) == []


@pytest.mark.parametrize("v_style", ["eip155", "recid", "27"])
def test_dynamic_evm_signer_folds_every_v_encoding_into_an_eip155_transaction(v_style: str) -> None:
    signer, client = dynamic_evm(v_style=v_style)
    signed = signer.sign_transaction({"from": EVM_ACCOUNT.address, "to": TO, "data": "0x", "value": 5, "chainId": 1,
                                      "nonce": 7, "gas": 21000, "gasPrice": 10**9})
    assert Account.recover_transaction(signed.raw_transaction) == EVM_ACCOUNT.address
    assert signed.hash == remote._eth_utils().keccak(signed.raw_transaction)
    unsigned = client.calls[-1][2]
    assert unsigned == {"to": TO, "value": 5, "nonce": 7, "gas": 21000, "gasPrice": 10**9, "chainId": 1, "data": "0x"}
    assert client.calls[-1][3] == "pw"                                     # the password reaches Dynamic, nothing else


def test_dynamic_evm_signer_refuses_eip1559_fields() -> None:
    signer, client = dynamic_evm()
    with pytest.raises(ConfigurationError, match="legacy"):
        signer.sign_transaction({"to": TO, "chainId": 1, "nonce": 0, "gas": 21000, "maxFeePerGas": 1, "maxPriorityFeePerGas": 1})
    assert sign_calls(client) == []


def test_dynamic_signer_authenticates_and_resolves_once_then_closes() -> None:
    signer, client = dynamic_evm()
    assert client.calls == []
    assert signer.resolve() == "dyn-wallet" and signer.wallet_id == "dyn-wallet"
    w3 = fake_web3(chain_id=1)
    conn = Ethereum(w3=w3, signer=signer)
    conn.send_transaction({"to": TO, "data": "0x", "value": 1})
    conn.send_transaction({"to": TO, "data": "0x", "value": 2})
    names = [c[0] for c in client.calls]
    assert names == ["authenticate_api_token", "load_wallet", "sign_transaction", "sign_transaction"]
    assert client.calls[0][1] == "dyn_token" and client.calls[1][1] == EVM_ACCOUNT.address
    signer.close()
    assert client.closed


def test_dynamic_signer_reauthenticates_once_its_session_is_old(monkeypatch: Any) -> None:
    from aleo_bridge import dynamic as dyn

    now = [1000.0]
    monkeypatch.setattr(dyn.time, "monotonic", lambda: now[0])
    client = FakeDynamicClient(chain_name="EVM")
    signer = DynamicEvmSigner(client, address=EVM_ACCOUNT.address, api_token="dyn_token", password="pw",
                              reauthenticate_seconds=100)
    tx = {"to": TO, "data": "0x", "value": 1, "chainId": 1, "nonce": 0, "gas": 21000, "gasPrice": 1}
    signer.sign_transaction(tx)
    now[0] += 99
    signer.sign_transaction(tx)
    assert [c[0] for c in client.calls].count("authenticate_api_token") == 1
    now[0] += 1
    signer.sign_transaction(tx)
    names = [c[0] for c in client.calls]
    assert names.count("authenticate_api_token") == 2 and names.count("load_wallet") == 1
    assert names[-2:] == ["authenticate_api_token", "sign_transaction"]       # the fresh session comes first


def test_dynamic_signer_reports_a_missing_wallet_id_as_none() -> None:
    client = FakeDynamicClient(chain_name="EVM", wallet_id="")
    signer = DynamicEvmSigner(client, address=EVM_ACCOUNT.address, api_token="t", password="pw")
    assert signer.resolve() is None and signer.wallet_id is None


def test_run_async_times_out_without_masking_a_providers_own_timeout() -> None:
    import asyncio

    async def slow() -> None:
        await asyncio.sleep(5)

    async def inner_timeout() -> None:
        raise TimeoutError("the MPC relay stalled")

    with pytest.raises(BridgeError, match="did not answer within 0.05 seconds"):
        remote.run_async(slow(), timeout=0.05)
    with pytest.raises(TimeoutError, match="MPC relay"):
        remote.run_async(inner_timeout(), timeout=5)


@pytest.mark.parametrize("problem", ["chain", "address", "wallet_id"])
def test_dynamic_signer_refuses_a_wallet_that_is_not_the_configured_one(problem: str) -> None:
    kwargs: dict[str, Any] = {}
    if problem == "chain":
        client = FakeDynamicClient(chain_name="SVM")
    elif problem == "address":
        client = FakeDynamicClient(chain_name="EVM", echo_address=OTHER_ACCOUNT.address)
    else:
        client = FakeDynamicClient(chain_name="EVM", wallet_id="another-wallet")
        kwargs["wallet_id"] = "dyn-wallet"
    signer = DynamicEvmSigner(client, address=EVM_ACCOUNT.address, api_token="t", password="pw", **kwargs)
    with pytest.raises(ConfigurationError, match="does not match|not the configured wallet_id"):
        signer.resolve()
    assert sign_calls(client) == []


def test_dynamic_signers_need_a_password_or_key_shares_and_a_real_client() -> None:
    with pytest.raises(ConfigurationError, match="password=.*key_shares="):
        DynamicEvmSigner(FakeDynamicClient(chain_name="EVM"), address=EVM_ACCOUNT.address)
    with pytest.raises(ConfigurationError, match="dynamic_wallet_sdk"):
        DynamicEvmSigner(object(), address=EVM_ACCOUNT.address, password="pw")
    with pytest.raises(ConfigurationError, match="must be a 0x-prefixed Ethereum address"):
        DynamicEvmSigner(FakeDynamicClient(chain_name="EVM"), address="nope", password="pw")


def test_privy_resolve_checks_identity_and_requires_a_privy_client() -> None:
    signer, client = privy_evm()
    assert signer.resolve() == EVM_ACCOUNT.address and client.calls == [("get", "evm-wallet")]
    other = PrivyEvmSigner(client, wallet_id="other-wallet", address=EVM_ACCOUNT.address)
    with pytest.raises(ConfigurationError, match="does not match the configured ethereum address"):
        other.resolve()
    sol, _ = privy_sol()
    assert sol.resolve() == sol.address
    with pytest.raises(ConfigurationError, match="privy.PrivyClient"):
        PrivyEvmSigner(object(), wallet_id="evm-wallet", address=EVM_ACCOUNT.address)
    with pytest.raises(ConfigurationError, match="wallet_id must not be empty"):
        PrivyEvmSigner(client, wallet_id=" ", address=EVM_ACCOUNT.address)


def test_privy_evm_signer_passes_authorization_keys_and_legacy_fields_through() -> None:
    client = FakePrivyClient()
    signer = PrivyEvmSigner(client, wallet_id="evm-wallet", address=EVM_ACCOUNT.address, authorization_private_keys=["wallet-auth:key"])
    signed = signer.sign_transaction({"to": TO, "data": b"\x12\x34", "value": 1, "chainId": 1, "nonce": 3, "gas": 21000, "gasPrice": 10**9})
    assert Account.recover_transaction(signed.raw_transaction) == EVM_ACCOUNT.address and signed.raw_transaction[0] != 2
    _, params, address, options = client.calls[-1]
    assert address == EVM_ACCOUNT.address
    assert params["transaction"] == {"from": EVM_ACCOUNT.address, "to": TO, "value": 1, "chain_id": 1, "nonce": 3,
                                     "gas_limit": 21000, "data": "0x1234", "type": 0, "gas_price": 10**9}
    assert tuple(options.authorization_context.authorization_private_keys) == ("wallet-auth:key",)


# ── Solana ───────────────────────────────────────────────────────────────────

def solana_module(signer: Any) -> tuple[SolModule, FakeSolanaClient]:
    fake = FakeSolanaClient()
    return SolModule(stub_bridge(), Solana(client=fake, signer=signer)), fake


@pytest.mark.parametrize("provider", ["privy", "dynamic"])
def test_solana_signer_adds_only_the_fee_payer_signature_and_keeps_the_ephemeral_one(provider: str) -> None:
    signer, client = SOL_SIGNERS[provider]()
    mod, fake = solana_module(signer)
    assert sign_calls(client) == []
    result = mod.transfer_remote(RECIPIENT, amount_atomic=1).send()
    assert len(sign_calls(client)) == 1 and len(fake.sent) == 1
    tx = VersionedTransaction.from_bytes(fake.sent[0])
    assert tx.message.account_keys[0] == signer.pubkey()
    assert all(signature != Signature.default() for signature in tx.signatures)
    assert tx.signatures[0].verify(signer.pubkey(), to_bytes_versioned(tx.message))
    tx.verify_and_hash_message()                                           # every signature, including the ephemeral one
    assert result.transaction_id == str(tx.signatures[0])
    if provider == "privy":
        wire = client.calls[-1][1]
        requested = VersionedTransaction.from_bytes(wire)
        assert to_bytes_versioned(requested.message) == to_bytes_versioned(tx.message)
        assert all(signature == Signature.default() for signature in requested.signatures)   # no key material travels
        assert client.calls[-1][2] == signer.address
    else:
        assert client.calls[-1][2] == to_bytes_versioned(tx.message) and client.calls[-1][3] == "pw"


@pytest.mark.parametrize("provider", ["privy", "dynamic"])
def test_solana_signer_refuses_a_bad_signature_without_broadcasting(provider: str) -> None:
    signer, client = SOL_SIGNERS[provider]("bad-signature")
    mod, fake = solana_module(signer)
    with pytest.raises(BridgeError, match="signature"):
        mod.transfer_remote(RECIPIENT, amount_atomic=1).send()
    assert len(sign_calls(client)) == 1 and fake.sent == []


@pytest.mark.parametrize("provider", ["privy", "dynamic"])
def test_solana_signing_failure_propagates_without_retry_or_broadcast(provider: str) -> None:
    signer, client = SOL_SIGNERS[provider]("error")
    mod, fake = solana_module(signer)
    with pytest.raises(RuntimeError, match="remote policy"):
        mod.transfer_remote(RECIPIENT, amount_atomic=1).send()
    assert len(sign_calls(client)) == 1 and fake.sent == []


def test_privy_solana_signer_rejects_a_changed_message_before_broadcasting() -> None:
    signer, client = privy_sol("changed-message")
    mod, fake = solana_module(signer)
    with pytest.raises(BridgeError, match="changed the Solana transaction message"):
        mod.transfer_remote(RECIPIENT, amount_atomic=1).send()
    assert len(sign_calls(client)) == 1 and fake.sent == []


@pytest.mark.parametrize("provider", ["privy", "dynamic"])
def test_solana_signer_refuses_a_message_it_is_not_a_required_signer_of(provider: str) -> None:
    signer, client = SOL_SIGNERS[provider]()
    payer = Keypair()
    message = MessageV0.try_compile(payer.pubkey(), [transfer(TransferParams(
        from_pubkey=payer.pubkey(), to_pubkey=signer.pubkey(), lamports=1))], [], Hash.default())
    with pytest.raises(BridgeError, match="not a required signer"):
        signer.sign_message(to_bytes_versioned(message))
    with pytest.raises(BridgeError, match="not a Solana message"):
        signer.sign_message(b"\x80garbage")
    assert sign_calls(client) == []


def test_dynamic_solana_signer_accepts_the_providers_case_folded_echo() -> None:
    client = FakeDynamicClient(chain_name="SOL")
    address = str(client.sol_keypair.pubkey())
    client.echo_address = address.lower()
    signer = DynamicSolanaSigner(client, address=address, api_token="t", password="pw")
    assert signer.resolve() == "dyn-wallet" and signer.address == address
    assert isinstance(signer.pubkey(), Pubkey)


# ── examples ─────────────────────────────────────────────────────────────────

def _signer_mock(address: str) -> Mock:
    signer = Mock()
    signer.address = address
    signer.wallet_id = "wallet-id"
    signer.resolve.return_value = address
    return signer


@pytest.mark.parametrize("module_name,env", [
    ("privy_wallets", {"PRIVY_APP_ID": "app", "PRIVY_APP_SECRET": "secret", "PRIVY_EVM_WALLET_ID": "w1",
                       "PRIVY_EVM_ADDRESS": EVM_ACCOUNT.address, "PRIVY_SOLANA_WALLET_ID": "w2",
                       "PRIVY_SOLANA_ADDRESS": "GBc9L6GVQ4UtXzpFAe1oBKSKgLtLvPJKfJEbz2T8CcJd"}),
    ("dynamic_wallets", {"DYNAMIC_ENVIRONMENT_ID": "env", "DYNAMIC_API_TOKEN": "dyn_token",
                         "DYNAMIC_EVM_ADDRESS": EVM_ACCOUNT.address, "DYNAMIC_EVM_WALLET_PASSWORD": "pw",
                         "DYNAMIC_SOLANA_ADDRESS": "GBc9L6GVQ4UtXzpFAe1oBKSKgLtLvPJKfJEbz2T8CcJd",
                         "DYNAMIC_SOLANA_WALLET_PASSWORD": "pw"}),
])
@pytest.mark.parametrize("chain", ["ethereum", "solana"])
@pytest.mark.parametrize("execute", [False, True])
def test_server_wallet_examples_quote_or_submit_once(module_name: str, env: dict[str, str], chain: str, execute: bool,
                                                      monkeypatch: Any, tmp_path: Any, capsys: Any) -> None:
    # The examples import the provider SDK at module level; dynamic-wallet-sdk needs Python 3.11+.
    pytest.importorskip("privy" if module_name == "privy_wallets" else "dynamic_wallet_sdk")
    module = importlib.import_module(module_name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    address = env[f"{module_name.split('_')[0].upper()}_{'EVM' if chain == 'ethereum' else 'SOLANA'}_ADDRESS"]
    evm_signer, sol_signer = _signer_mock(address), _signer_mock(address)
    for attr in ("Aleo", "Ethereum", "Solana", "PrivyClient", "DynamicEvmWalletClient", "DynamicSvmWalletClient"):
        if hasattr(module, attr):
            monkeypatch.setattr(module, attr, Mock())
    for attr in ("PrivyEvmSigner", "DynamicEvmSigner"):
        if hasattr(module, attr):
            monkeypatch.setattr(module, attr, Mock(return_value=evm_signer))
    for attr in ("PrivySolanaSigner", "DynamicSolanaSigner"):
        if hasattr(module, attr):
            monkeypatch.setattr(module, attr, Mock(return_value=sol_signer))
    bridge = Mock()
    bridge.quote.return_value = SimpleNamespace(amount_out="0.00001", fees=[], plan=object())
    bridge.execute.return_value = SimpleNamespace(next="done", error=None,
                                                  receipt=SimpleNamespace(id="receipt", source_tx_id="source"))
    monkeypatch.setattr(module, "Bridge", Mock(return_value=bridge))

    args = ["--chain", chain, "--recipient", RECIPIENT, "--journal", str(tmp_path)] + (["--execute"] if execute else [])
    assert module.main(args) == 0

    call = bridge.quote.call_args.kwargs
    assert call["source_chain"] == chain and call["destination_chain"] == "aleo" and call["sender"] == address
    assert call["amount"] == {"ethereum": "0.00001", "solana": "0.0001"}[chain]
    assert bridge.execute.call_count == (1 if execute else 0)
    assert bridge.resume.call_count == 0
    out = capsys.readouterr().out
    assert address in out and "secret" not in out and "dyn_token" not in out and "pw" not in out.split()


def test_server_wallet_examples_name_only_the_missing_variable(monkeypatch: Any, capsys: Any) -> None:
    pytest.importorskip("privy")
    module = importlib.import_module("privy_wallets")
    monkeypatch.delenv("PRIVY_APP_ID", raising=False)
    with pytest.raises(SystemExit) as exc:
        module.main(["--recipient", RECIPIENT])
    assert exc.value.code == 2 and "PRIVY_APP_ID" in capsys.readouterr().err
