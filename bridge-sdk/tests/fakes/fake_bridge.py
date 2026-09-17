"""A duck-typed Bridge whose protocol modules record calls and return scripted results.

Shape fidelity matters: results carry the real ``types`` dataclasses, Aleo calls
expose ``delegate_prepared / prove / submit_prepared`` exactly like ``AleoCall``,
EVM/Solana calls invoke ``on_checkpoint(receipt)`` for every intermediate
submission before returning, and ``eth``/``sol`` are properties that raise
``ConfigurationError`` when the matching ``ethereum``/``solana`` connection is
``None`` — the conventions plans 2/3 implement.

``FakeBridge.aleo`` is ``tests.conftest.FakeAleo`` (the facade-level fake already
used by plan 1-3 tests) rather than a second, ad hoc Aleo fake: there is exactly
one Aleo fake in this test suite.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from aleo import AleoNetworkError
from aleo.facade.errors import TransactionNotFound

from aleo_bridge.errors import AttestationError, ConfigurationError, RegistryVersionMismatchError
from aleo_bridge.registry import DEFAULT_REGISTRY
from aleo_bridge.types import (Attestation, BridgeStatus, BurnReceipt, ChainStatus, DepositReceipt,
                               DispatchReceipt, EvmHyperlaneQuote, EvmXReserveQuote, GasQuote,
                               MintReceipt, PreparedTx, PrivacyReceipt, Receipt,
                               SolanaHyperlaneQuote, Status)

from tests.conftest import FakeAleo as _ConftestFakeAleo
from tests.conftest import default_mappings

ALEO_RECIPIENT = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
EVM_ADDRESS = "0x0000000000000000000000000000000000000001"
SOL_ADDRESS = "11111111111111111111111111111111"
OUTBOUND = {"aleo/eth": "ethereum/eth", "aleo/wbtc": "ethereum/wbtc",
            "aleo/usdt": "ethereum/usdt", "aleo/sol": "solana/sol"}


def serialized_tx(tx_id: str) -> str:
    return json.dumps({"type": "execute", "id": tx_id, "fee": {}})


class FakeAleoCall:
    def __init__(self, fake: "FakeBridge", program_id: str, function_name: str, inputs: list[str],
                 tx_id: str, make_result: Callable[[str], Any]) -> None:
        self.fake, self.program_id, self.function_name, self.inputs = fake, program_id, function_name, inputs
        self.tx_id, self._make = tx_id, make_result

    def simulate(self, account=None):
        self.fake.events.append(("simulate", self.function_name))
        return "authorized"

    def prove(self, account=None, **fee) -> PreparedTx:
        self.fake.events.append(("prove", self.tx_id))
        return PreparedTx(self.tx_id, serialized_tx(self.tx_id))

    def delegate_prepared(self, account=None, **fee) -> PreparedTx:
        self.fake.events.append(("delegate_prepared", self.tx_id))
        return PreparedTx(self.tx_id, serialized_tx(self.tx_id))

    def submit_prepared(self, prepared: PreparedTx, *, wait=True, wait_timeout=180.0):
        self.fake.events.append(("submit", prepared.transaction_id, wait))
        self.fake.submitted.append(prepared.serialized)
        return self._make(prepared.transaction_id)

    def transact(self, account=None, **fee):
        self.fake.events.append(("transact", self.tx_id))
        return self._make(self.tx_id)

    def delegate(self, account=None, **kw):
        self.fake.events.append(("delegate", self.tx_id))
        return self._make(self.tx_id)


class FakeEvmCall:
    def __init__(self, fake: "FakeBridge", intermediates: list[Receipt], final: Any) -> None:
        self.fake, self.intermediates, self.final = fake, intermediates, final

    def build(self) -> list[dict]:
        return [{"to": "0xrouter", "data": "0x", "value": 0}]

    def send(self, *, wait=True, timeout_seconds=120.0, poll_seconds=1.0, on_checkpoint=None):
        self.fake.events.append(("evm_send", timeout_seconds, poll_seconds))
        for receipt in self.intermediates:
            if on_checkpoint is not None:
                on_checkpoint(receipt)
        return self.final


FakeSolCall = FakeEvmCall


@dataclass
class FakeConnection:
    address: str | None
    can_sign: bool = True


class FakeHyperlane:
    def __init__(self, fake: "FakeBridge") -> None:
        self.fake = fake
        self.gas = GasQuote(route_id="hyperlane:aleo/eth->ethereum/eth", gas_limit=44_000,
                            gas_overhead=159_337, gas_price=1_000_000_000, exchange_rate=402,
                            payment_microcredits=8_174_147)
        self.delivered: dict[str, bool] = {}

    def quote_gas_payment(self, asset) -> GasQuote:
        self.fake.calls.append(("hyperlane.quote_gas_payment", asset))
        return replace(self.gas, route_id=f"hyperlane:{asset}->{OUTBOUND[asset]}")

    def transfer_remote(self, asset, recipient, *, amount=None, amount_atomic=None,
                        as_signer=False, gas_payment_microcredits=None) -> FakeAleoCall:
        kw = dict(asset=asset, recipient=recipient, amount=amount, amount_atomic=amount_atomic,
                  as_signer=as_signer, gas_payment_microcredits=gas_payment_microcredits)
        self.fake.calls.append(("hyperlane.transfer_remote", kw))
        route_id = f"hyperlane:{asset}->{OUTBOUND[asset]}"
        program = f"hyp_warp_token_{asset.split('/')[1]}_v2.aleo"
        fn = "transfer_remote_as_signer" if as_signer else "transfer_remote"

        def make(tx_id: str) -> DispatchReceipt:
            receipt = Receipt(id=tx_id, protocol="hyperlane", status=Status.SOURCE_CONFIRMING,
                              source_tx_id=tx_id,
                              protocol_state={"routeId": route_id, "sourceProgram": program,
                                              "sourceFunction": fn})
            return DispatchReceipt(tx_id, route_id, None, amount_atomic or 0, receipt)

        return FakeAleoCall(self.fake, program, fn, ["<7 literals>"] * 7, self.fake.next_tx_id(), make)

    def is_delivered(self, message_id) -> bool:
        key = message_id if isinstance(message_id, str) else "0x" + bytes(message_id).hex()
        self.fake.calls.append(("hyperlane.is_delivered", key))
        return self.delivered.get(key, False)


class FakeXReserve:
    def __init__(self, fake: "FakeBridge") -> None:
        self.fake = fake
        self.attestations: dict[str, Attestation] = {}   # messageHash hex -> Attestation
        self.delivered_nonces: set[str] = set()
        self.expected_secret_nonce: str | None = None    # private_mint raises when it differs

    def _route(self):
        env = self.fake.environment
        return ("xreserve:aleo/usdcx->ethereum/usdc" if env == "mainnet"
                else "xreserve:aleo-testnet/usdcx->sepolia/usdc")

    def burn(self, recipient, *, amount=None, amount_atomic=None, mode="private",
             record=None, merkle_proof=None) -> FakeAleoCall:
        kw = dict(recipient=recipient, amount=amount, amount_atomic=amount_atomic, mode=mode,
                  record=record, merkle_proof=merkle_proof)
        self.fake.calls.append(("xreserve.burn", kw))
        route_id = self._route()
        program = "shielded_usdcx_wrapper.aleo" if mode == "private" else "usdcx_bridge_v2.aleo"
        fn = {"private": "private_burn", "public": "burn_public",
              "public-as-signer": "burn_public_as_signer"}[mode]

        def make(tx_id: str) -> BurnReceipt:
            receipt = Receipt(id=tx_id, protocol="xreserve", status=Status.SOURCE_CONFIRMING,
                              source_tx_id=tx_id,
                              protocol_state={"routeId": route_id, "burnMode": mode,
                                              "sourceProgram": program, "sourceFunction": fn})
            return BurnReceipt(tx_id, route_id, mode, amount_atomic or 0, receipt)

        return FakeAleoCall(self.fake, program, fn, ["<burn inputs>"], self.fake.next_tx_id(), make)

    def private_mint(self, attestation: Attestation, recipient, *, secret_nonce="0scalar",
                     route=None) -> FakeAleoCall:
        self.fake.calls.append(("xreserve.private_mint", {"recipient": recipient,
                                                          "secret_nonce": secret_nonce,
                                                          "message_hash": "0x" + attestation.message_hash.hex()}))
        if self.expected_secret_nonce is not None and secret_nonce != self.expected_secret_nonce:
            raise AttestationError("Private mint secret nonce and recipient do not match the attested hook data")
        route_id = route.id if route is not None else "xreserve:ethereum/usdc->aleo/usdcx"

        def make(tx_id: str) -> MintReceipt:
            receipt = Receipt(id=tx_id, protocol="xreserve", status=Status.DESTINATION_CONFIRMING,
                              destination_tx_id=tx_id, protocol_state={"routeId": route_id})
            return MintReceipt(tx_id, route_id, receipt)

        return FakeAleoCall(self.fake, "shielded_usdcx_wrapper.aleo", "private_mint",
                            ["<5 inputs>"], self.fake.next_tx_id(), make)

    def get_attestation(self, message_hash, *, route=None) -> Attestation | None:
        key = message_hash if isinstance(message_hash, str) else "0x" + bytes(message_hash).hex()
        self.fake.calls.append(("xreserve.get_attestation", key.lower()))
        return self.attestations.get(key.lower())

    def is_delivered(self, nonce, *, route=None) -> bool:
        key = nonce if isinstance(nonce, str) else "0x" + bytes(nonce).hex()
        self.fake.calls.append(("xreserve.is_delivered", key.lower()))
        return key.lower() in self.delivered_nonces


class FakeEth:
    """Mirrors ``aleo_bridge.eth.EthModule``'s public surface for lifecycle tests.

    ``quote_transfer_remote``/``quote_deposit_usdc`` accept ``plan=`` exactly like the real
    module: mutually exclusive with ``asset=``/``route=``/``sender=`` (a ``ValueError`` otherwise),
    checked against ``DEFAULT_REGISTRY.version``, and the returned quote carries that same ``plan``
    object (``.plan is plan``) — ``lifecycle.quote`` is what canonicalizes the plan on the result, so
    the fake does not need to rebuild one the way the real module does.
    """

    def __init__(self, fake: "FakeBridge", address: str) -> None:
        self.fake, self.address = fake, address
        self.approval_required = False
        self.balances: dict[str, int] = {}
        self.delivered: dict[str, bool] = {}
        self.hook_data = bytes([0]) + b"\x00" * 64
        self.source_status_result: Receipt | None = None
        self.recover_result: Receipt | None = None
        self.intermediates: list[Receipt] = []

    def quote_transfer_remote(self, asset=None, recipient=None, *, amount=None, amount_atomic=None,
                              route=None, sender=None, plan=None):
        if plan is not None:
            if asset is not None or route is not None or sender is not None:
                raise ValueError("Pass plan= or asset=/route=/sender=, not both")
            if plan.registry_version != DEFAULT_REGISTRY.version:
                raise RegistryVersionMismatchError(
                    f"Plan uses registry {plan.registry_version}; this client has {DEFAULT_REGISTRY.version}")
            self.fake.calls.append(("eth.quote_transfer_remote", {"plan": plan}))
            return EvmHyperlaneQuote(kind="evm-hyperlane", plan=plan, fees=(), amount_out=None,
                                     recipient_bytes32=b"\x00" * 32,
                                     native_value_atomic=plan.amount_atomic + 1000,
                                     native_fee_atomic=1000, approval_required=self.approval_required)
        self.fake.calls.append(("eth.quote_transfer_remote", dict(asset=asset, recipient=recipient,
                                                                  amount_atomic=amount_atomic)))
        return EvmHyperlaneQuote(kind="evm-hyperlane", plan=None, fees=(), amount_out=None,
                                 recipient_bytes32=b"\x00" * 32, native_value_atomic=(amount_atomic or 0) + 1000,
                                 native_fee_atomic=1000, approval_required=self.approval_required)

    def transfer_remote(self, asset, recipient, *, amount=None, amount_atomic=None) -> FakeEvmCall:
        self.fake.calls.append(("eth.transfer_remote", dict(asset=asset, recipient=recipient,
                                                            amount_atomic=amount_atomic)))
        route_id = f"hyperlane:{asset}->aleo/{asset.split('/')[1]}"
        tx = "0x" + "aa" * 32
        receipt = Receipt(id=tx, protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id=tx,
                          protocol_state={"routeId": route_id, "approvalTxIds": [], "sourceSender": self.address,
                                          "amountAtomic": str(amount_atomic or 0)})
        return FakeEvmCall(self.fake, self.intermediates, DispatchReceipt(tx, route_id, None, amount_atomic or 0, receipt))

    def quote_deposit_usdc(self, recipient=None, *, amount=None, amount_atomic=None, mint_mode=None,
                           secret_nonce="0scalar", sender=None, route=None, plan=None):
        if plan is not None:
            if route is not None or sender is not None:
                raise ValueError("Pass plan= or route=/sender=, not both")
            if plan.registry_version != DEFAULT_REGISTRY.version:
                raise RegistryVersionMismatchError(
                    f"Plan uses registry {plan.registry_version}; this client has {DEFAULT_REGISTRY.version}")
            self.fake.calls.append(("eth.quote_deposit_usdc", {"plan": plan, "secret_nonce": secret_nonce}))
            return EvmXReserveQuote(kind="evm-xreserve", plan=plan, fees=(), amount_out=None,
                                    hook_data=self.hook_data, remote_recipient_bytes32=b"\x00" * 32,
                                    balance_atomic=10_000_000,
                                    allowance_atomic=0 if self.approval_required else 10_000_000,
                                    approval_required=self.approval_required, max_fee_atomic=100_000)
        mint_mode = "public" if mint_mode is None else mint_mode
        self.fake.calls.append(("eth.quote_deposit_usdc", dict(recipient=recipient, amount_atomic=amount_atomic,
                                                               mint_mode=mint_mode, secret_nonce=secret_nonce)))
        return EvmXReserveQuote(kind="evm-xreserve", plan=None, fees=(), amount_out=None, hook_data=self.hook_data,
                                remote_recipient_bytes32=b"\x00" * 32, balance_atomic=10_000_000,
                                allowance_atomic=0 if self.approval_required else 10_000_000,
                                approval_required=self.approval_required, max_fee_atomic=100_000)

    def deposit_usdc(self, recipient, *, amount=None, amount_atomic=None, mint_mode="public",
                     secret_nonce="0scalar") -> FakeEvmCall:
        self.fake.calls.append(("eth.deposit_usdc", dict(recipient=recipient, amount_atomic=amount_atomic,
                                                         mint_mode=mint_mode, secret_nonce=secret_nonce)))
        route_id = ("xreserve:ethereum/usdc->aleo/usdcx" if self.fake.environment == "mainnet"
                    else "xreserve:sepolia/usdc->aleo-testnet/usdcx")
        tx = "0x" + "bb" * 32
        message_hash = "0x" + "cc" * 32
        receipt = Receipt(id=message_hash, protocol="xreserve", status=Status.ATTESTATION_PENDING, source_tx_id=tx,
                          protocol_state={"routeId": route_id, "approvalTxIds": [], "sourceSender": self.address,
                                          "mintMode": mint_mode, "intendedRecipient": recipient,
                                          "hookData": "0x" + self.hook_data.hex(), "nonce": "0x" + "dd" * 32,
                                          "payload": "0x" + "ee" * 305, "messageHash": message_hash,
                                          "bridgeProgram": "usdcx_bridge_v2.aleo"})
        return FakeEvmCall(self.fake, self.intermediates,
                           DepositReceipt(tx, route_id, message_hash, "0x" + "dd" * 32, receipt))

    def balance(self, asset) -> int:
        self.fake.calls.append(("eth.balance", asset))
        return self.balances.get(asset, 0)

    def is_delivered(self, message_id) -> bool:
        key = message_id if isinstance(message_id, str) else "0x" + bytes(message_id).hex()
        self.fake.calls.append(("eth.is_delivered", key))
        return self.delivered.get(key, False)

    def source_status(self, plan, receipt) -> Receipt:
        self.fake.calls.append(("eth.source_status", receipt.status))
        return self.source_status_result or receipt

    def recover_source(self, plan, checkpoint, *, required=False) -> Receipt:
        self.fake.calls.append(("eth.recover_source", checkpoint.to_dict(), required))
        assert self.recover_result is not None, "script FakeEth.recover_result first"
        return self.recover_result


class FakeSol:
    """Mirrors ``aleo_bridge.sol.SolModule``'s public surface for lifecycle tests.

    ``quote_transfer_remote`` mirrors the REAL ``SolModule.quote_transfer_remote`` signature
    exactly: ``recipient`` is required positionally (no default, unlike ``EthModule``'s quote
    methods) and, when ``plan=`` is given, the real module silently overwrites
    recipient/amount/amount_atomic/sender from the plan rather than raising ``ValueError`` on a
    conflict — there is no ``asset=``/``route=`` kwarg to conflict with in the first place. This
    fake matches that real behavior rather than the more Eth-like ``ValueError`` ruling.
    """

    def __init__(self, fake: "FakeBridge", address: str) -> None:
        self.fake, self.address = fake, address
        self.balance_lamports = 0
        self.source_status_result: Receipt | None = None
        self.intermediates: list[Receipt] = []

    def quote_transfer_remote(self, recipient, *, amount=None, amount_atomic=None, sender=None, plan=None):
        if plan is not None:
            if plan.registry_version != DEFAULT_REGISTRY.version:
                raise RegistryVersionMismatchError(
                    f"Plan uses registry {plan.registry_version}; this client has {DEFAULT_REGISTRY.version}")
            self.fake.calls.append(("sol.quote_transfer_remote", {"plan": plan}))
            return SolanaHyperlaneQuote(kind="solana-hyperlane", plan=plan, fees=(), amount_out=None,
                                        igp_lamports=2_900_000, network_fee_lamports=10_000, rent_lamports=5_004_240,
                                        total_lamports=plan.amount_atomic + 7_914_240,
                                        unique_message_address="uniq1111111111111111111111111111111111111111")
        self.fake.calls.append(("sol.quote_transfer_remote", dict(recipient=recipient, amount_atomic=amount_atomic)))
        return SolanaHyperlaneQuote(kind="solana-hyperlane", plan=None, fees=(), amount_out=None,
                                    igp_lamports=2_900_000, network_fee_lamports=10_000, rent_lamports=5_004_240,
                                    total_lamports=(amount_atomic or 0) + 7_914_240,
                                    unique_message_address="uniq1111111111111111111111111111111111111111")

    def transfer_remote(self, recipient, *, amount=None, amount_atomic=None) -> FakeSolCall:
        self.fake.calls.append(("sol.transfer_remote", dict(recipient=recipient, amount_atomic=amount_atomic)))
        route_id = "hyperlane:solana/sol->aleo/sol"
        sig = "5igNature" * 8
        receipt = Receipt(id=sig, protocol="hyperlane", status=Status.SOURCE_CONFIRMING, source_tx_id=sig,
                          protocol_state={"routeId": route_id, "signature": sig, "blockhash": "recent",
                                          "lastValidBlockHeight": "123456789"})
        return FakeSolCall(self.fake, self.intermediates, DispatchReceipt(sig, route_id, None, amount_atomic or 0, receipt))

    def balance(self) -> int:
        self.fake.calls.append(("sol.balance",))
        return self.balance_lamports

    def source_status(self, plan, receipt) -> Receipt:
        self.fake.calls.append(("sol.source_status", receipt.status))
        return self.source_status_result or receipt


class FakeBridge:
    """Duck-typed stand-in for ``aleo_bridge.client.Bridge`` (no network, no extras)."""

    def __init__(self, *, environment="mainnet", ethereum=True, solana=False, checkpoints=None) -> None:
        self.registry = DEFAULT_REGISTRY
        self.environment = self.network = environment
        self.checkpoints = checkpoints
        self.events: list[tuple] = []        # ordered side effects (prove/submit/checkpoint...)
        self.calls: list[tuple] = []         # module method calls with kwargs
        self.submitted: list[str] = []
        self._tx = 0
        self.aleo = _ConftestFakeAleo(mappings=default_mappings(), network_name=environment)
        self.hyperlane = FakeHyperlane(self)
        self.xreserve = FakeXReserve(self)
        self.ethereum = FakeConnection(EVM_ADDRESS) if ethereum else None
        self._eth = FakeEth(self, EVM_ADDRESS) if ethereum else None
        self.solana = FakeConnection(SOL_ADDRESS) if solana else None
        self._sol = FakeSol(self, SOL_ADDRESS) if solana else None
        self.public_balances: dict[str, int] = {}

    @property
    def eth(self) -> FakeEth:
        """Mirrors the real ``Bridge.eth`` property: ``ConfigurationError`` when ``ethereum`` is None."""
        if self._eth is None:
            raise ConfigurationError("Pass ethereum=Ethereum(...) to Bridge(...) or set ETHEREUM_RPC_URL")
        return self._eth

    @property
    def sol(self) -> FakeSol:
        """Mirrors the real ``Bridge.sol`` property: ``ConfigurationError`` when ``solana`` is None."""
        if self._sol is None:
            raise ConfigurationError(
                "Solana is not configured: pass solana=Solana(rpc_url, private_key=...) or a solana-py Client to Bridge(), "
                "or set SOLANA_PRIVATE_KEY (and optionally SOLANA_RPC_URL) for Bridge.from_env()")
        return self._sol

    def next_tx_id(self) -> str:
        self._tx += 1
        return f"at1fake{self._tx}"

    def aleo_address(self) -> str:
        return ALEO_RECIPIENT

    # Plan-1 surface the agent tools touch
    def shield(self, asset, *, amount=None, amount_atomic=None, recipient=None) -> FakeAleoCall:
        self.calls.append(("shield", dict(asset=asset, amount=amount, amount_atomic=amount_atomic)))
        return FakeAleoCall(self, "arc20_eth.aleo", "shield", [f"{amount_atomic}u128"], self.next_tx_id(),
                            lambda tx: PrivacyReceipt(tx, asset, str(amount or amount_atomic), amount_atomic or 0, "shield"))

    def unshield(self, asset, *, amount=None, amount_atomic=None, record=None, merkle_proof=None, recipient=None):
        self.calls.append(("unshield", dict(asset=asset, amount=amount, amount_atomic=amount_atomic)))
        return FakeAleoCall(self, "arc20_eth.aleo", "unshield", ["<record>", f"{amount_atomic}u128"], self.next_tx_id(),
                            lambda tx: PrivacyReceipt(tx, asset, str(amount or amount_atomic), amount_atomic or 0, "unshield"))

    def status(self) -> BridgeStatus:
        chains = [ChainStatus("aleo" if self.environment == "mainnet" else "aleo-testnet", ALEO_RECIPIENT, True,
                              dict(self.public_balances))]
        if self.ethereum is not None:
            chains.append(ChainStatus("ethereum", self.ethereum.address, True, dict(self.eth.balances)))
        if self.solana is not None:
            chains.append(ChainStatus("solana", self.solana.address, True, {"solana/sol": self.sol.balance_lamports}))
        from aleo_bridge.lifecycle import recover
        pending = [recover(self, cp) for cp in self.checkpoints.list()] if self.checkpoints else []
        return BridgeStatus(self.environment, self.registry.version, chains, pending)
