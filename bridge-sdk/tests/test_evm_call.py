import pytest
from eth_account import Account
from web3 import Web3

from aleo_bridge._calls import EvmCall, EvmOutcome, EvmStep
from aleo_bridge._evm_abi import ERC20_ABI, WARP_ROUTE_ABI
from aleo_bridge.checkpoint import FileCheckpointStore
from aleo_bridge.errors import BridgeError, ConfigurationError
from aleo_bridge.eth import Ethereum, _plan_for
from aleo_bridge.registry import DEFAULT_REGISTRY
from aleo_bridge.types import DispatchReceipt, Receipt, Status
from tests.fakes.fake_web3 import fake_web3, tx_hash_for

KEY = "0x" + "11" * 32
ACCT = Account.from_key(KEY)
ALEO = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
WBTC = "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599"
ROUTER = "0x20CDC85778b732073F7EecEF3DF25c0d310f8772"
ROUTE = DEFAULT_REGISTRY.route("hyperlane:ethereum/wbtc->aleo/wbtc")
USDC_ROUTE = DEFAULT_REGISTRY.route("xreserve:ethereum/usdc->aleo/usdcx")


def make_call(w3, *, sender=None, store=None, approvals=1):
    conn = Ethereum(w3=w3, private_key=KEY)
    plan = _plan_for(DEFAULT_REGISTRY, ROUTE, amount_atomic=100_000, recipient=ALEO, sender=sender)
    token = w3.eth.contract(address=Web3.to_checksum_address(WBTC), abi=ERC20_ABI)
    warp = w3.eth.contract(address=Web3.to_checksum_address(ROUTER), abi=WARP_ROUTE_ABI)

    def steps(owner):
        out = [EvmStep("approve", token.address, token.encode_abi("approve", args=[warp.address, 100_000]), 0)
               for _ in range(approvals)]
        out.append(EvmStep("main", warp.address, warp.encode_abi("transferRemote", args=[1634493807, b"\x11" * 32, 100_000]), 50_000))
        return out

    def finish(outcome: EvmOutcome) -> DispatchReceipt:
        status = Status.DELIVERY_PENDING if outcome.status == "CONFIRMED" else Status(outcome.status)
        rid = outcome.source_tx_id or (outcome.approval_tx_ids[-1] if outcome.approval_tx_ids else "unsent")
        receipt = Receipt(id=rid, protocol="hyperlane", status=status, source_tx_id=outcome.source_tx_id,
                          protocol_state={"routeId": ROUTE.id, "approvalTxIds": list(outcome.approval_tx_ids),
                                          "sourceSender": outcome.sender})
        return DispatchReceipt(transaction_id=rid, route_id=ROUTE.id, message_id=None, amount_atomic=100_000, receipt=receipt)

    return EvmCall(conn, plan=plan, registry=DEFAULT_REGISTRY, steps=steps, finish=finish, store=store), plan


def test_plan_for_hyperlane_and_xreserve_shapes():
    plan = _plan_for(DEFAULT_REGISTRY, ROUTE, amount_atomic=100_000, recipient=ALEO, sender=ACCT.address)
    assert plan.route_id == ROUTE.id and plan.registry_version == DEFAULT_REGISTRY.version
    assert plan.protocol == "hyperlane" and plan.environment == "mainnet"
    assert plan.source_asset_id == "ethereum/wbtc" and plan.destination_asset_id == "aleo/wbtc"
    assert plan.amount == "0.001" and plan.amount_atomic == 100_000 and plan.recipient == ALEO
    assert plan.sender == ACCT.address and plan.mint_mode == "public"
    assert [(s.id, s.kind, s.executor, s.irreversible) for s in plan.steps] == [
        ("source-approval", "approve", "evm-wallet", False),
        ("source-dispatch", "dispatch", "evm-wallet", True),
        ("message-delivery", "wait-delivery", "protocol", False),
        ("destination-confirmation", "confirm-delivery", "protocol", False),
    ]
    eth_plan = _plan_for(DEFAULT_REGISTRY, DEFAULT_REGISTRY.route("hyperlane:ethereum/eth->aleo/eth"),
                         amount_atomic=100, recipient=ALEO, sender=None)
    assert [s.id for s in eth_plan.steps][0] == "source-dispatch"          # native: no approval step
    assert eth_plan.amount == "0.0000000000000001"
    private = _plan_for(DEFAULT_REGISTRY, USDC_ROUTE, amount_atomic=2_000_000, recipient=ALEO, sender=None, mint_mode="private")
    assert private.amount == "2" and private.mint_mode == "private"
    assert [(s.id, s.executor, s.irreversible) for s in private.steps] == [
        ("source-approval", "evm-wallet", False), ("source-deposit", "evm-wallet", True),
        ("deposit-attestation", "protocol", False), ("destination-mint", "aleo-wallet", False),
    ]
    public = _plan_for(DEFAULT_REGISTRY, USDC_ROUTE, amount_atomic=2_000_000, recipient=ALEO, sender=None)
    assert public.steps[-1].executor == "protocol"
    with pytest.raises(BridgeError, match="mint_mode"):
        _plan_for(DEFAULT_REGISTRY, ROUTE, amount_atomic=1, recipient=ALEO, sender=None, mint_mode="private")


def test_build_returns_unsigned_dicts_in_order_without_sending():
    w3 = fake_web3()
    call, _ = make_call(w3)
    txs = call.build()
    assert [t["to"] for t in txs] == [Web3.to_checksum_address(WBTC), Web3.to_checksum_address(ROUTER)]
    assert [t["value"] for t in txs] == [0, 50_000]
    assert [t["nonce"] for t in txs] == [0, 1]
    assert all(t["from"] == ACCT.address and t["chainId"] == 1 for t in txs)
    assert txs[1]["data"].startswith("0x") and w3.provider.sent == []


def test_send_runs_approve_then_main_and_checkpoints_each_hash_before_polling():
    w3 = fake_web3()
    # Make both receipts resolve only after a couple of pending polls, so the ordering
    # test actually exercises "checkpoint fires, THEN polling happens" rather than a
    # same-tick resolution that would pass even if the code checkpointed after polling.
    w3.provider.receipt_delay[tx_hash_for(1)] = 2
    w3.provider.receipt_delay[tx_hash_for(2)] = 2
    call, plan = make_call(w3)
    seen = []
    poll_counts_at_checkpoint = []

    def on_checkpoint(cp):
        seen.append(cp)
        poll_counts_at_checkpoint.append(dict(w3.provider.receipt_poll_counts))

    result = call.send(on_checkpoint=on_checkpoint, poll_seconds=0.001)
    assert isinstance(result, DispatchReceipt) and result.receipt.status == Status.DELIVERY_PENDING
    assert [t["to"] for t in w3.provider.sent] == [Web3.to_checksum_address(WBTC), Web3.to_checksum_address(ROUTER)]
    assert w3.provider.sent[1]["value"] == 50_000
    assert result.receipt.protocol_state["approvalTxIds"] == [tx_hash_for(1)] and result.receipt.source_tx_id == tx_hash_for(2)
    assert [cp.source for cp in seen] == [
        {"approvalTransactionIds": [tx_hash_for(1)]},
        {"approvalTransactionIds": [tx_hash_for(1)], "transactionId": tx_hash_for(2)},
        {"approvalTransactionIds": [tx_hash_for(1)], "transactionId": tx_hash_for(2)},
    ]
    assert all(cp.route == {"id": plan.route_id, "registryVersion": plan.registry_version} for cp in seen)
    assert seen[0].intent["sender"] == ACCT.address

    # Checkpoint-before-poll ordering, per hash:
    # cp0 (approval broadcast) fires before any eth_getTransactionReceipt for hash1.
    assert poll_counts_at_checkpoint[0].get(tx_hash_for(1), 0) == 0
    assert tx_hash_for(2) not in poll_counts_at_checkpoint[0]
    # cp1 (main broadcast) fires after hash1 was fully polled to confirmation, but
    # before any eth_getTransactionReceipt for hash2.
    assert poll_counts_at_checkpoint[1].get(tx_hash_for(1), 0) > 0
    assert poll_counts_at_checkpoint[1].get(tx_hash_for(2), 0) == 0
    # cp2 (confirmed) fires only after hash2 has itself been polled.
    assert poll_counts_at_checkpoint[2].get(tx_hash_for(2), 0) > 0
    # ... and each hash's poll count strictly increases after its own checkpoint fired
    # (the receipt_delay=2 knob forces at least one more poll beyond the checkpoint tick).
    assert w3.provider.receipt_poll_counts[tx_hash_for(1)] > poll_counts_at_checkpoint[0].get(tx_hash_for(1), 0)
    assert w3.provider.receipt_poll_counts[tx_hash_for(2)] > poll_counts_at_checkpoint[1].get(tx_hash_for(2), 0)

    # Full RPC sequence: approve is sent and fully confirmed (>=1 receipt poll) before
    # the main call is ever broadcast, and the main call is polled only afterwards.
    relevant = [m for m in w3.provider.methods if m in ("eth_sendRawTransaction", "eth_getTransactionReceipt")]
    first_send = relevant.index("eth_sendRawTransaction")
    second_send = relevant.index("eth_sendRawTransaction", first_send + 1)
    assert relevant[first_send] == "eth_sendRawTransaction"
    between = relevant[first_send + 1:second_send]
    assert between and all(m == "eth_getTransactionReceipt" for m in between)
    after = relevant[second_send + 1:]
    assert after and all(m == "eth_getTransactionReceipt" for m in after)


def test_approval_timeout_returns_pending_and_stops():
    w3 = fake_web3()
    w3.provider.pending.add(tx_hash_for(1))
    call, _ = make_call(w3)
    result = call.send(timeout_seconds=0.01, poll_seconds=0.001)
    assert result.receipt.status == Status.SOURCE_APPROVAL_PENDING and result.receipt.source_tx_id is None
    assert result.receipt.protocol_state["approvalTxIds"] == [tx_hash_for(1)] and len(w3.provider.sent) == 1


def test_main_timeout_returns_source_confirming():
    w3 = fake_web3()
    w3.provider.pending.add(tx_hash_for(1))
    call, _ = make_call(w3, approvals=0)
    result = call.send(timeout_seconds=0.01, poll_seconds=0.001)
    assert result.receipt.status == Status.SOURCE_CONFIRMING and result.receipt.source_tx_id == tx_hash_for(1)


def test_wait_false_returns_after_first_broadcast():
    w3 = fake_web3()
    call, _ = make_call(w3)
    result = call.send(wait=False)
    assert result.receipt.status == Status.SOURCE_APPROVAL_PENDING and len(w3.provider.sent) == 1


def test_reverted_transaction_raises():
    w3 = fake_web3()
    w3.provider.reverted.add(tx_hash_for(1))
    call, _ = make_call(w3)
    with pytest.raises(BridgeError, match=f"EVM transaction reverted: {tx_hash_for(1)}"):
        call.send(poll_seconds=0.001)


def test_plan_sender_must_match_connected_account():
    w3 = fake_web3()
    call, _ = make_call(w3, sender="0x0000000000000000000000000000000000000001")
    with pytest.raises(ConfigurationError, match="does not match connected account"):
        call.send()
    assert w3.provider.sent == []


def test_build_rejects_a_plan_sender_that_is_not_the_connected_account():
    w3 = fake_web3()
    call, _ = make_call(w3, sender="0x0000000000000000000000000000000000000001")
    with pytest.raises(ConfigurationError, match="does not match connected account"):
        call.build()
    assert w3.provider.sent == []


class ExplodingStore:
    """A checkpoint store whose disk is full / read-only."""

    def __init__(self):
        self.attempts = []

    def save(self, checkpoint):
        self.attempts.append(checkpoint)
        raise OSError("read-only file system")

    def load(self, checkpoint_id):          # pragma: no cover - never reached
        return None

    def list(self):                          # pragma: no cover - never reached
        return []

    def delete(self, checkpoint_id):         # pragma: no cover - never reached
        return None


def test_store_failure_after_broadcast_reports_the_tx_hash_and_never_hides_it():
    """The transaction is already on the wire: the caller's callback must have run first, the
    error must name the hash and the checkpoint, and no receipt poll may follow the failure."""
    w3 = fake_web3()
    store = ExplodingStore()
    call, _ = make_call(w3, store=store, approvals=0)
    seen = []
    with pytest.raises(BridgeError) as exc:
        call.send(on_checkpoint=seen.append, poll_seconds=0.001)
    message = str(exc.value)
    assert tx_hash_for(1) in message and "broadcast" in message and "checkpoint" in message.lower()
    assert [cp.id for cp in seen] == [tx_hash_for(1)]                 # callback ran before the store
    assert [cp.id for cp in store.attempts] == [tx_hash_for(1)]
    assert len(w3.provider.sent) == 1                                # broadcast happened exactly once
    assert "eth_getTransactionReceipt" not in w3.provider.methods    # nothing polled after the failure


def test_bound_store_saves_every_checkpoint(tmp_path):
    w3 = fake_web3()
    store = FileCheckpointStore(tmp_path)
    call, _ = make_call(w3, store=store)
    result = call.send(poll_seconds=0.001)
    ids = {cp.id for cp in store.list()}
    assert tx_hash_for(1) in ids and result.receipt.id in ids
