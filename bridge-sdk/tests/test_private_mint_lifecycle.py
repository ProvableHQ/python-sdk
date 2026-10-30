"""The lifecycle verbs derive, keep and recover a private mint nonce for the bridge's own account.

``execute`` reserves one identity and deposits with its scalar; ``resume`` and ``complete`` find the
scalar again from the checkpointed hook / the attested payload, or re-derive it by counting up when
the store lost it. An explicit ``secret_nonce=`` bypasses the store; nothing derivable means a
refusal before any RPC — except for a quote, which prices the transfer regardless."""
import aleo
import pytest

from aleo_bridge import private_mint as pm
from aleo_bridge.encoding import xreserve_deposit_payload, xreserve_hook_data, xreserve_message_hash
from aleo_bridge.errors import ConfigurationError, NotResumableError
from aleo_bridge.lifecycle import complete, execute, prepare, quote, resume, to_progress
from aleo_bridge.types import Attestation, Receipt, Status
from tests.fakes.fake_bridge import EVM_ADDRESS, FakeBridge

ACCOUNT = aleo.mainnet.Account.from_private_key(aleo.mainnet.PrivateKey.random())
ME = str(ACCOUNT.address)
OTHER = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
APPROVAL = "0x" + "11" * 32
QUOTE = dict(source_chain="ethereum", source_asset="usdc", destination_chain="aleo", destination_asset="usdcx",
             amount="2", recipient=ME, sender=EVM_ADDRESS, mint_mode="private")


def _bridge():
    b = FakeBridge()
    b.aleo.default_account = ACCOUNT
    b.private_mint_identities = pm.MemoryPrivateMintIdentityStore()
    return b


def _plan(b, recipient=ME):
    return prepare(b.registry, **{**QUOTE, "recipient": recipient})


def _reserve(b, recipient=ME):
    return pm.reserve_identity(b.private_mint_identities, view_key=ACCOUNT.view_key, recipient=recipient,
                               network="mainnet")


# ── execute ──────────────────────────────────────────────────────────────────

def test_execute_reserves_one_identity_and_deposits_with_its_scalar():
    b = _bridge()
    execute(b, _plan(b))
    [identity] = b.private_mint_identities.load()
    assert b.calls[0] == ("eth.deposit_usdc", {"plan": _plan(b), "secret_nonce": identity.secret_nonce})
    assert identity.recipient == ME and identity.secret_nonce != "0scalar"


def test_execute_with_an_explicit_nonce_reserves_nothing():
    b = _bridge()
    execute(b, _plan(b), secret_nonce="7scalar")
    assert b.private_mint_identities.load() == [] and b.calls[0][1]["secret_nonce"] == "7scalar"


def test_execute_without_store_or_nonce_is_refused_before_any_rpc():
    b = _bridge()
    b.private_mint_identities = None
    with pytest.raises(ConfigurationError, match="secret_nonce"):
        execute(b, _plan(b))
    assert b.calls == []


# ── quote ────────────────────────────────────────────────────────────────────

def test_quote_prices_a_private_transfer_on_a_read_only_bridge():
    b = FakeBridge()                                   # no store, no view key
    q = quote(b, **QUOTE)
    assert q.kind == "evm-xreserve" and b.calls[0][1]["secret_nonce"] == "0scalar"


def test_quote_peeks_without_consuming_a_counter():
    b = _bridge()
    quote(b, **QUOTE)
    assert b.calls[0][1]["secret_nonce"] == pm.derive_secret_nonce(ACCOUNT.view_key, 0, "mainnet")
    assert b.private_mint_identities.load() == []


# ── resume ───────────────────────────────────────────────────────────────────

def _resumable(b, nonce):
    plan = _plan(b)
    hook = xreserve_hook_data("private", ME, "mainnet", nonce)
    b.eth.hook_data = hook
    receipt = Receipt(id=APPROVAL, protocol="xreserve", status=Status.SOURCE_SUBMISSION_PENDING,
                      protocol_state={"routeId": plan.route_id, "approvalTxIds": [APPROVAL],
                                      "sourceSender": EVM_ADDRESS, "hookData": "0x" + hook.hex()})
    b.eth.recover_result = receipt
    return plan, receipt


def test_resume_resolves_the_scalar_from_the_checkpointed_hook():
    b = _bridge()
    identity = _reserve(b)
    plan, receipt = _resumable(b, identity.secret_nonce)
    resume(b, to_progress(plan, receipt))
    assert b.calls[-1][1]["secret_nonce"] == identity.secret_nonce


def test_resume_recovers_by_scanning_when_the_store_was_lost():
    b = _bridge()
    nonce = pm.derive_secret_nonce(ACCOUNT.view_key, 3, "mainnet")
    plan, receipt = _resumable(b, nonce)
    resume(b, to_progress(plan, receipt))
    assert b.calls[-1][1]["secret_nonce"] == nonce and b.private_mint_identities.load()[0].counter == 3


def test_resume_refuses_a_malformed_private_hook_before_any_rpc():
    b = _bridge()
    plan, receipt = _resumable(b, "7scalar")
    bad = "0x02" + "11" * 32 + "01" + "00" * 31
    with pytest.raises((ConfigurationError, NotResumableError)):
        resume(b, to_progress(plan, receipt.replace(protocol_state={**receipt.protocol_state, "hookData": bad})))
    assert b.calls == []


# ── complete ─────────────────────────────────────────────────────────────────

def _ready(b, nonce, recipient=ME):
    plan = _plan(b, recipient)
    hook = xreserve_hook_data("private", recipient, "mainnet", nonce)
    payload = xreserve_deposit_payload(amount=2_000_000, remote_domain=10_002, remote_token=b"\x11" * 32,
                                       remote_recipient=b"\x22" * 32,
                                       local_token="0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238",
                                       depositor=EVM_ADDRESS, max_fee=100_000, nonce=b"\x00" * 32, hook_data=hook)
    message_hash = "0x" + xreserve_message_hash(payload).hex()
    receipt = Receipt(id=message_hash, protocol="xreserve", status=Status.DESTINATION_ACTION_REQUIRED,
                      source_tx_id="0x" + "22" * 32,
                      next_action={"kind": "xreserve-private-mint", "chainId": "aleo"},
                      protocol_state={"routeId": plan.route_id, "mintMode": "private", "intendedRecipient": recipient,
                                      "payload": "0x" + payload.hex(), "messageHash": message_hash,
                                      "attestation": "0x" + "33" * 65})
    return plan, receipt


def test_complete_resolves_the_scalar_from_the_attested_hook():
    b = _bridge()
    identity = _reserve(b)
    b.xreserve.expected_secret_nonce = identity.secret_nonce
    plan, receipt = _ready(b, identity.secret_nonce)
    out = complete(b, to_progress(plan, receipt))
    assert out.receipt.status is Status.DESTINATION_CONFIRMING
    assert b.calls[-1][1]["secret_nonce"] == identity.secret_nonce


def test_complete_refuses_when_a_stored_identity_names_another_recipient():
    """A stored record whose commitment matches the attested hook but whose recipient is not the
    plan's (tampering, or a hand-edited file) must stop before proving."""
    b = _bridge()
    nonce = pm.derive_secret_nonce(ACCOUNT.view_key, 0, "mainnet")
    plan, receipt = _ready(b, nonce, recipient=ME)                    # the deposit commits ME
    b.private_mint_identities.save([pm.PrivateMintIdentity(
        counter=0, recipient=OTHER, secret_nonce=nonce,
        address_commitment=pm.derive_address_commitment(ME, nonce, "mainnet"))])
    with pytest.raises(ConfigurationError, match="recipient"):
        complete(b, to_progress(plan, receipt))
    assert b.events == []


def test_tier2_private_mint_resolves_when_no_nonce_is_given():
    from aleo_bridge.xreserve import XReserveModule
    b = _bridge()
    identity = _reserve(b)
    plan, receipt = _ready(b, identity.secret_nonce)
    route = next(r for r in b.registry.routes(bridge_protocol="xreserve", environment="mainnet") if r.id == plan.route_id)
    att = Attestation(payload=bytes.fromhex(receipt.protocol_state["payload"][2:]),
                      message_hash=bytes.fromhex(receipt.id[2:]), attestation=b"\x33" * 65, status="complete")
    inputs = XReserveModule(b).build_private_mint_inputs(route, att, ME, None)
    assert inputs[3] == identity.secret_nonce and inputs[4] == ME
