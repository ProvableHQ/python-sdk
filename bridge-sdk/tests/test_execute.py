"""``lifecycle.execute`` — the verb that commits funds on the source chain.

The invariants under test: every source leg is dispatched through the module's own ``plan=``
surface (never a re-derived asset/recipient/amount), an Aleo leg proves → checkpoints the exact
transaction → broadcasts → checkpoints the id (so a crash between proving and broadcast is
recoverable), and every boundary reaches the caller's callback once and the bound store once.
"""
import dataclasses
import json

import pytest

from aleo_bridge.checkpoint import FileCheckpointStore
from aleo_bridge.errors import ConfigurationError, RouteUnavailableError
from aleo_bridge.lifecycle import execute, prepare, quote
from aleo_bridge.types import Receipt, Status
from tests.fakes.fake_bridge import ALEO_RECIPIENT, EVM_ADDRESS, SOL_ADDRESS, FakeBridge

PREPARED_TX = {"preparedTransaction": {"transactionId": "at1fake1",
                                       "serializedTransaction": json.dumps(
                                           {"type": "execute", "id": "at1fake1", "fee": {}})}}


def _aleo_eth_plan(b, recipient=EVM_ADDRESS):
    return prepare(b.registry, source="aleo/eth", destination="ethereum/eth",
                   amount="0.000000000000000001", recipient=recipient)


def _usdc_plan(b, **kw):
    return prepare(b.registry, source="ethereum/usdc", destination="aleo/usdcx", amount="2",
                   recipient=ALEO_RECIPIENT, **kw)


def _wbtc_plan(b, **kw):
    return prepare(b.registry, source="ethereum/wbtc", destination="aleo/wbtc", amount="0.001",
                   recipient=ALEO_RECIPIENT, **kw)


def _sol_plan(b, **kw):
    return prepare(b.registry, source="solana/sol", destination="aleo/sol", amount="0.000000001",
                   recipient=ALEO_RECIPIENT, **kw)


# ── Aleo-origin legs ──────────────────────────────────────────────────────────

def test_aleo_hyperlane_checkpoints_prepared_tx_before_broadcast_and_requotes_igp():
    b = FakeBridge(ethereum=False)
    plan = _aleo_eth_plan(b)
    cps = []
    progress = execute(b, plan, on_checkpoint=cps.append)

    assert b.calls[0] == ("hyperlane.quote_gas_payment", "aleo/eth")            # re-quoted at the last moment
    assert b.calls[1][1]["gas_payment_microcredits"] == 8_174_147
    assert b.calls[1][1]["as_signer"] is False and b.calls[1][1]["amount_atomic"] == 1
    kinds = [e[0] for e in b.events]
    assert kinds == ["delegate_prepared", "checkpoint:SOURCE_SUBMISSION_PENDING", "submit",
                     "checkpoint:SOURCE_CONFIRMING"]
    assert b.events[2][2] is False                                             # submit_prepared(wait=False)
    assert cps[0].source == PREPARED_TX
    assert cps[1].source == {"transactionId": "at1fake1"}
    assert "deliveryVerification" not in cps[0].to_dict()                      # no ETH connection → no baseline
    assert progress.next == "wait" and progress.receipt.status is Status.SOURCE_CONFIRMING
    assert progress.receipt.source_tx_id == "at1fake1"
    assert progress.receipt.protocol_state["routeId"] == plan.route_id


def test_aleo_hyperlane_pinned_gas_and_signer_mode_and_local_proving():
    b = FakeBridge(ethereum=False)
    execute(b, _aleo_eth_plan(b), gas_payment_microcredits=1, mode="signer", proving="local")
    assert b.calls[0][0] == "hyperlane.transfer_remote"                          # no quote call
    assert b.calls[0][1]["gas_payment_microcredits"] == 1 and b.calls[0][1]["as_signer"] is True
    assert [e[0] for e in b.events][0] == "prove"
    with pytest.raises(ConfigurationError, match="proving"):
        execute(b, _aleo_eth_plan(b), gas_payment_microcredits=1, proving="wallet")
    with pytest.raises(ConfigurationError, match="mode"):
        execute(b, _aleo_eth_plan(b), gas_payment_microcredits=1, mode="private")


def test_aleo_hyperlane_captures_destination_balance_baseline_for_own_recipient():
    b = FakeBridge()                                    # ethereum configured, address == recipient
    b.eth.balances["ethereum/eth"] = 100
    plan = _aleo_eth_plan(b)
    cps = []
    progress = execute(b, plan, gas_payment_microcredits=1, on_checkpoint=cps.append)
    # read before the dispatch is even built, so the pre-broadcast checkpoint can carry it
    assert b.calls[0] == ("eth.balance", "ethereum/eth")
    assert b.calls[1][0] == "hyperlane.transfer_remote"
    assert cps[0].delivery_verification == {"balanceBeforeAtomic": "100", "expectedIncreaseAtomic": "1"}
    assert progress.receipt.protocol_state["destinationBalanceBeforeAtomic"] == "100"
    assert progress.receipt.protocol_state["expectedDestinationIncreaseAtomic"] == "1"
    # a recipient that is not our connection's address gets no baseline (we cannot read its balance)
    b2 = FakeBridge()
    cps2 = []
    execute(b2, _aleo_eth_plan(b2, recipient="0x0000000000000000000000000000000000000002"),
            gas_payment_microcredits=1, on_checkpoint=cps2.append)
    assert cps2[0].delivery_verification is None and ("eth.balance", "ethereum/eth") not in b2.calls


def test_aleo_xreserve_burn_modes_and_private_inputs():
    b = FakeBridge(ethereum=False)
    plan = prepare(b.registry, source="aleo/usdcx", destination="ethereum/usdc", amount="2.5",
                   recipient=EVM_ADDRESS)
    cps = []
    progress = execute(b, plan, record="{ owner: aleo1..., amount: 3000000u128.private }",
                       merkle_proof="[{ siblings: [...], leaf_index: 1u32 }, { ... }]", on_checkpoint=cps.append)
    assert b.calls[0][0] == "xreserve.burn"
    assert b.calls[0][1]["mode"] == "private" and b.calls[0][1]["amount_atomic"] == 2_500_000
    assert b.calls[0][1]["record"].startswith("{ owner") and b.calls[0][1]["merkle_proof"].startswith("[")
    assert [c.source for c in cps] == [PREPARED_TX, {"transactionId": "at1fake1"}]
    assert progress.receipt.protocol_state["burnMode"] == "private"
    b2 = FakeBridge(ethereum=False)
    execute(b2, plan, mode="public-as-signer")
    assert b2.calls[0][1]["mode"] == "public-as-signer"
    with pytest.raises(ConfigurationError, match="mode"):
        execute(b2, plan, mode="signer")


def test_aleo_xreserve_burn_captures_the_destination_balance_baseline_net_of_the_fee():
    """I2/R1: Circle has no delivery query for Aleo→EVM, so the baseline written here is the ONLY
    way ``get_status`` branch 8 can ever see the USDC land and let ``wait`` terminate. The expected
    increase is the quote's ``amount_out`` — the withdrawal fee is paid OUT OF the burned amount,
    so waiting for the full amount would wait for a delivery that can never arrive (and could be
    satisfied by unrelated inflow instead). Recorded exactly like the Hyperlane leg: only when our
    EVM connection is itself the recipient."""
    b = FakeBridge()                                    # ethereum configured, address == recipient
    b.eth.balances["ethereum/usdc"] = 100
    plan = prepare(b.registry, source="aleo/usdcx", destination="ethereum/usdc", amount="2.000001",
                   recipient=EVM_ADDRESS)
    quoted = quote(b, source="aleo/usdcx", destination="ethereum/usdc", amount="2.000001",
                   recipient=EVM_ADDRESS)
    cps = []
    progress = execute(b, plan, mode="public", on_checkpoint=cps.append)
    assert b.calls[0] == ("eth.balance", "ethereum/usdc")          # read before the burn is built
    assert b.calls[1][0] == "xreserve.burn"                        # the quote above is offline
    # 2.000001 USDCx burned − the registry's 2 USDCx withdrawal fee = 1 atomic unit expected out
    assert quoted.amount_out == "0.000001" and quoted.withdrawal_fee_atomic == 2_000_000
    assert cps[0].delivery_verification == {"balanceBeforeAtomic": "100", "expectedIncreaseAtomic": "1"}
    assert progress.receipt.protocol_state["destinationBalanceBeforeAtomic"] == "100"
    assert progress.receipt.protocol_state["expectedDestinationIncreaseAtomic"] == "1"
    # a recipient that is not our connection's address still gets no baseline, and no read at all
    b2 = FakeBridge()
    other = prepare(b2.registry, source="aleo/usdcx", destination="ethereum/usdc", amount="2.000001",
                    recipient="0x0000000000000000000000000000000000000002")
    cps2 = []
    execute(b2, other, mode="public", on_checkpoint=cps2.append)
    assert cps2[0].delivery_verification is None and ("eth.balance", "ethereum/usdc") not in b2.calls


# ── EVM- and Solana-origin legs ───────────────────────────────────────────────

def test_evm_hyperlane_forwards_intermediate_checkpoints_and_polling_controls():
    b = FakeBridge()
    plan = _wbtc_plan(b, sender=EVM_ADDRESS)
    approval = Receipt(id="0x" + "11" * 32, protocol="hyperlane", status=Status.SOURCE_APPROVAL_PENDING,
                       protocol_state={"routeId": plan.route_id, "approvalTxIds": ["0x" + "11" * 32],
                                       "sourceSender": EVM_ADDRESS})
    b.eth.intermediates = [approval]
    cps = []
    progress = execute(b, plan, on_checkpoint=cps.append, poll_seconds=2.0, timeout_seconds=300.0)
    assert b.calls == [("eth.transfer_remote", {"plan": plan})]         # the module re-derives nothing
    assert ("evm_send", 300.0, 2.0) in b.events
    assert [c.source for c in cps] == [{"approvalTransactionIds": ["0x" + "11" * 32]},
                                       {"transactionId": "0x" + "aa" * 32}]
    assert progress.next == "wait" and progress.receipt.status is Status.SOURCE_CONFIRMING


def test_evm_xreserve_passes_mint_mode_and_secret_nonce():
    b = FakeBridge()
    plan = _usdc_plan(b, mint_mode="private")
    progress = execute(b, plan, secret_nonce="7scalar")
    assert b.calls[0] == ("eth.deposit_usdc", {"plan": plan, "secret_nonce": "7scalar"})
    assert progress.receipt.status is Status.ATTESTATION_PENDING and progress.next == "wait"
    assert progress.receipt.protocol_state["mintMode"] == "private"
    assert "secretNonce" not in json.dumps(progress.receipt.protocol_state)
    assert "7scalar" not in json.dumps(progress.receipt.protocol_state)


def test_a_private_mint_without_a_secret_nonce_is_refused_before_any_rpc():
    """``secret_nonce`` is not a plan field and the SDK never stores it: a private deposit that
    silently fell back to "0scalar" would mint to a commitment the caller cannot ever reproduce."""
    b = FakeBridge()
    with pytest.raises(ConfigurationError, match="secret_nonce"):
        execute(b, _usdc_plan(b, mint_mode="private"))
    assert b.calls == []
    execute(b, _usdc_plan(b))                                    # public mint: the default is fine
    assert b.calls[0][1]["secret_nonce"] == "0scalar"
    b2 = FakeBridge()
    execute(b2, _usdc_plan(b2, mint_mode="record"))
    assert b2.calls[0][1]["secret_nonce"] == "0scalar"


def test_solana_hyperlane():
    b = FakeBridge(solana=True)
    plan = _sol_plan(b, sender=SOL_ADDRESS)
    cps = []
    progress = execute(b, plan, on_checkpoint=cps.append)
    assert b.calls == [("sol.transfer_remote", {"plan": plan})]
    assert cps[-1].source["blockhash"] == "recent" and cps[-1].source["lastValidBlockHeight"] == "123456789"
    assert progress.receipt.status is Status.SOURCE_CONFIRMING


# ── refusals ──────────────────────────────────────────────────────────────────

def test_sender_mismatch_missing_connection_and_unavailable_route():
    b = FakeBridge()
    with pytest.raises(ConfigurationError, match="sender"):
        execute(b, _wbtc_plan(b, sender="0x0000000000000000000000000000000000000009"))
    with pytest.raises(ConfigurationError, match="Solana connection"):
        execute(b, _sol_plan(b))
    with pytest.raises(RouteUnavailableError):
        execute(b, prepare(b.registry, source="aleo/aleo", destination="ethereum/aleo", amount="1",
                           recipient=EVM_ADDRESS))
    no_eth = FakeBridge(ethereum=False)                # bridge.ethereum is None → bridge.eth must not be touched
    with pytest.raises(ConfigurationError, match="Ethereum connection"):
        execute(no_eth, _wbtc_plan(no_eth))
    assert b.calls == [] and no_eth.calls == []


def test_the_sender_check_is_case_insensitive_for_evm_and_exact_for_solana():
    """EVM addresses are hex (checksum casing is cosmetic); Solana addresses are base58, where a
    case change is a different account entirely."""
    b = FakeBridge()
    b.ethereum.address = b.eth.address = "0xAbC0000000000000000000000000000000000001"
    execute(b, _wbtc_plan(b, sender="0xabc0000000000000000000000000000000000001"))
    assert b.calls[0][0] == "eth.transfer_remote"

    mixed = "So11111111111111111111111111111111111111112"
    s = FakeBridge(solana=True)
    s.solana.address = s.sol.address = mixed
    with pytest.raises(ConfigurationError, match="sender"):
        execute(s, _sol_plan(s, sender=mixed.lower()))
    assert s.calls == []
    execute(s, _sol_plan(s, sender=mixed))
    assert s.calls[0][0] == "sol.transfer_remote"


# ── the bound checkpoint store ────────────────────────────────────────────────

def test_bound_store_saves_every_checkpoint_and_replaces_superseded_ids(tmp_path):
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(checkpoints=store)
    plan = _usdc_plan(b)
    approval = Receipt(id="0x" + "11" * 32, protocol="xreserve", status=Status.SOURCE_APPROVAL_PENDING,
                       protocol_state={"routeId": plan.route_id, "approvalTxIds": ["0x" + "11" * 32]})
    b.eth.intermediates = [approval]
    execute(b, plan)
    saved = store.list()
    assert [c.id for c in saved] == ["0x" + "cc" * 32]           # approval file replaced by the deposit's
    assert saved[0].source == {"transactionId": "0x" + "bb" * 32, "hookData": "0x" + "00" * 65}


def test_module_and_lifecycle_checkpoint_channels_do_not_double_write(tmp_path):
    """The module saves the checkpoints it emits and ``execute`` emits the final receipt again:
    the caller still sees each boundary once and the store ends with one record per transfer."""
    store = FileCheckpointStore(tmp_path)
    b = FakeBridge(checkpoints=store)
    plan = _wbtc_plan(b, sender=EVM_ADDRESS)
    b.eth.intermediates = [Receipt(id="0x" + "11" * 32, protocol="hyperlane",
                                   status=Status.SOURCE_APPROVAL_PENDING,
                                   protocol_state={"routeId": plan.route_id,
                                                   "approvalTxIds": ["0x" + "11" * 32]})]
    cps = []
    execute(b, plan, on_checkpoint=cps.append)
    assert [c.id for c in cps] == ["0x" + "11" * 32, "0x" + "aa" * 32]      # no repeated boundary
    saved = store.list()
    assert [c.id for c in saved] == ["0x" + "aa" * 32]
    assert store.load("0x" + "aa" * 32).source == {"transactionId": "0x" + "aa" * 32}


def test_a_stale_plan_is_refused_before_anything_is_sent():
    b = FakeBridge()
    plan = dataclasses.replace(_wbtc_plan(b, sender=EVM_ADDRESS), registry_version="0000-00-00.stale")
    with pytest.raises(Exception, match="registry"):
        execute(b, plan)
    assert b.calls == [] and b.events == []


# ── review carry-overs (items 6-8) ────────────────────────────────────────────

def test_persist_never_leaves_the_store_empty_between_checkpoints(tmp_path):
    """Item 6: ``_persist`` saves the new checkpoint before deleting the superseded id, and a
    module-emitted checkpoint's own supersede is deferred (parked on the ``_Emitter``) until the
    module has actually saved it — otherwise a crash between delete and save loses the record."""
    store = FileCheckpointStore(tmp_path)
    observed = []
    orig_save, orig_delete = store.save, store.delete

    def save(cp):
        orig_save(cp)
        observed.append(len(store.list()))

    def delete(cid):
        orig_delete(cid)
        observed.append(len(store.list()))

    store.save, store.delete = save, delete
    b = FakeBridge(checkpoints=store)
    plan = _wbtc_plan(b, sender=EVM_ADDRESS)
    b.eth.intermediates = [Receipt(id="0x" + "11" * 32, protocol="hyperlane",
                                   status=Status.SOURCE_APPROVAL_PENDING,
                                   protocol_state={"routeId": plan.route_id,
                                                   "approvalTxIds": ["0x" + "11" * 32]})]
    execute(b, plan)
    assert observed and all(n >= 1 for n in observed)                # never empty in between
    assert [c.id for c in store.list()] == ["0x" + "aa" * 32]         # exactly one record after execute


def test_aleo_hyperlane_leg_refuses_a_sender_mismatch_before_proving():
    """Item 7: an Aleo leg checks the plan's sender against ``bridge.aleo_address()`` before
    proving anything."""
    b = FakeBridge(ethereum=False)
    plan = prepare(b.registry, source="aleo/eth", destination="ethereum/eth",
                   amount="0.000000000000000001", recipient=EVM_ADDRESS,
                   sender="aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqvfnl2t")
    with pytest.raises(ConfigurationError, match="sender"):
        execute(b, plan, gas_payment_microcredits=1)
    assert b.calls == [] and "delegate_prepared" not in [e[0] for e in b.events]


def test_aleo_xreserve_leg_refuses_a_sender_mismatch_before_proving():
    b = FakeBridge(ethereum=False)
    plan = prepare(b.registry, source="aleo/usdcx", destination="ethereum/usdc", amount="2.5",
                   recipient=EVM_ADDRESS,
                   sender="aleo1qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqvfnl2t")
    with pytest.raises(ConfigurationError, match="sender"):
        execute(b, plan, mode="public")
    assert b.calls == [] and "delegate_prepared" not in [e[0] for e in b.events]


def test_aleo_leg_with_no_sender_pinned_and_no_configured_account_is_unaffected():
    """A plan without a sender, or a bridge whose ``aleo_address()`` cannot be read, never blocks
    the leg — the check is a no-op when there is nothing to compare."""
    b = FakeBridge(ethereum=False)

    def raises(*a, **kw):
        raise ConfigurationError("no aleo account configured")

    b.aleo_address = raises
    execute(b, _aleo_eth_plan(b))
    assert b.calls[0][0] == "hyperlane.quote_gas_payment"


def test_destination_balance_baseline_read_is_best_effort(monkeypatch):
    """Item 8: an advisory destination-balance read never blocks funds movement."""
    b = FakeBridge()
    b.eth.balances["ethereum/eth"] = 100

    def raise_balance(asset):
        raise RuntimeError("RPC is down")

    monkeypatch.setattr(b.eth, "balance", raise_balance)
    plan = _aleo_eth_plan(b)
    cps = []
    progress = execute(b, plan, gas_payment_microcredits=1, on_checkpoint=cps.append)
    assert cps[0].delivery_verification is None
    assert "destinationBalanceBeforeAtomic" not in progress.receipt.protocol_state
