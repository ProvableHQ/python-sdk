"""Live checks of the Privy and Dynamic server-wallet signers against the real providers and mainnet.

Read-only by default (``BRIDGE_LIVE_READS=1``): provider authentication, wallet resolution,
balances and Hyperlane quotes for the Ethereum and Solana wallets — the same validation the
veil PR ran. Nothing here submits a transaction.

``BRIDGE_LIVE_REMOTE_WALLET_SIGNING=1`` additionally requests one real signature per wallet over a
throw-away transaction that is never broadcast (a zero-value self-transfer with nonce 0 on
Ethereum, a one-lamport transfer over a zero blockhash on Solana) and verifies it locally — that
exercises every provider code path a transfer uses except the broadcast, and moves no funds.

Credentials come from the operator's shell under the names the veil examples use:
``PRIVY_APP_ID`` / ``PRIVY_APP_SECRET`` / ``PRIVY_EVM_WALLET_ID`` / ``PRIVY_EVM_ADDRESS`` /
``PRIVY_SOLANA_WALLET_ID`` / ``PRIVY_SOLANA_ADDRESS`` (optional ``PRIVY_AUTHORIZATION_PRIVATE_KEY``),
and ``DYNAMIC_ENVIRONMENT_ID`` / ``DYNAMIC_API_TOKEN`` / ``DYNAMIC_EVM_ADDRESS`` /
``DYNAMIC_SOLANA_ADDRESS`` / ``DYNAMIC_EVM_WALLET_PASSWORD`` / ``DYNAMIC_SOLANA_WALLET_PASSWORD``
(optional ``DYNAMIC_EVM_WALLET_ID`` / ``DYNAMIC_SOLANA_WALLET_ID``). ``ETHEREUM_RPC_URL`` and
``SOLANA_RPC_URL`` default to the public mainnet endpoints. Values are never printed.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import pytest
import requests

from aleo_bridge.eth import Ethereum
from aleo_bridge.sol import Solana

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("BRIDGE_LIVE_READS") != "1", reason="set BRIDGE_LIVE_READS=1"),
]

ALEO_RECIPIENT = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
ETHEREUM_RPC_URL = os.environ.get("ETHEREUM_RPC_URL", "").strip() or "https://ethereum-rpc.publicnode.com"
SOLANA_RPC_URL = os.environ.get("SOLANA_RPC_URL", "").strip() or "https://api.mainnet-beta.solana.com"
SIGNING = os.environ.get("BRIDGE_LIVE_REMOTE_WALLET_SIGNING") == "1"

PRIVY_VARS = ("PRIVY_APP_ID", "PRIVY_APP_SECRET", "PRIVY_EVM_WALLET_ID", "PRIVY_EVM_ADDRESS",
              "PRIVY_SOLANA_WALLET_ID", "PRIVY_SOLANA_ADDRESS")
DYNAMIC_VARS = ("DYNAMIC_ENVIRONMENT_ID", "DYNAMIC_API_TOKEN", "DYNAMIC_EVM_ADDRESS", "DYNAMIC_SOLANA_ADDRESS",
                "DYNAMIC_EVM_WALLET_PASSWORD", "DYNAMIC_SOLANA_WALLET_PASSWORD")


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _require(names: tuple[str, ...]) -> None:
    missing = [name for name in names if not _env(name)]
    if missing:
        pytest.skip(f"set {', '.join(missing)}")


def _run(fn: Callable[[], Any]) -> Any:
    """Run *fn*; a 429/5xx from a public RPC is an environment condition, not a test failure."""
    try:
        return fn()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 429 or (status is not None and status >= 500):
            pytest.skip(f"public RPC rate-limited (HTTP {status})")
        raise


def _bridge(**connections: Any) -> Any:
    from aleo import Aleo, HTTPProvider

    from aleo_bridge import Bridge

    aleo = Aleo(HTTPProvider(_env("ALEO_RPC_URL") or "https://edge.provable.com/api", network="mainnet"))
    return Bridge(aleo, **connections)


def _signers(provider: str) -> tuple[Any, Any]:
    """``(evm_signer, solana_signer)`` for *provider*, built from the shell only; skips when unset."""
    if provider == "privy":
        _require(PRIVY_VARS)
        pytest.importorskip("privy")
        from privy import PrivyClient

        from aleo_bridge.privy import PrivyEvmSigner, PrivySolanaSigner

        client = PrivyClient(app_id=_env("PRIVY_APP_ID"), app_secret=_env("PRIVY_APP_SECRET"))
        keys = [_env("PRIVY_AUTHORIZATION_PRIVATE_KEY")] if _env("PRIVY_AUTHORIZATION_PRIVATE_KEY") else None
        return (PrivyEvmSigner(client, wallet_id=_env("PRIVY_EVM_WALLET_ID"), address=_env("PRIVY_EVM_ADDRESS"),
                               authorization_private_keys=keys),
                PrivySolanaSigner(client, wallet_id=_env("PRIVY_SOLANA_WALLET_ID"), address=_env("PRIVY_SOLANA_ADDRESS"),
                                  authorization_private_keys=keys))
    _require(DYNAMIC_VARS)
    pytest.importorskip("dynamic_wallet_sdk")
    from dynamic_wallet_sdk import DynamicEvmWalletClient, DynamicSvmWalletClient

    from aleo_bridge.dynamic import DynamicEvmSigner, DynamicSolanaSigner

    environment, token = _env("DYNAMIC_ENVIRONMENT_ID"), _env("DYNAMIC_API_TOKEN")
    return (DynamicEvmSigner(DynamicEvmWalletClient(environment), address=_env("DYNAMIC_EVM_ADDRESS"), api_token=token,
                             password=_env("DYNAMIC_EVM_WALLET_PASSWORD"), wallet_id=_env("DYNAMIC_EVM_WALLET_ID") or None),
            DynamicSolanaSigner(DynamicSvmWalletClient(environment), address=_env("DYNAMIC_SOLANA_ADDRESS"),
                                api_token=token, password=_env("DYNAMIC_SOLANA_WALLET_PASSWORD"),
                                wallet_id=_env("DYNAMIC_SOLANA_WALLET_ID") or None))


@pytest.fixture(params=["privy", "dynamic"])
def provider(request: Any) -> str:
    return request.param


def test_wallets_resolve_read_balances_and_quote_hyperlane_transfers(provider: str) -> None:
    evm_signer, sol_signer = _signers(provider)
    assert evm_signer.resolve()                                      # authenticates + identity check
    assert sol_signer.resolve()
    ethereum = Ethereum(ETHEREUM_RPC_URL, signer=evm_signer)
    solana = Solana(SOLANA_RPC_URL, signer=sol_signer)
    assert ethereum.address == evm_signer.address and ethereum.can_sign
    assert solana.address == sol_signer.address and solana.can_sign
    bridge = _bridge(ethereum=ethereum, solana=solana)

    status = _run(bridge.status)
    by_chain = {chain.chain_id: chain for chain in status.chains}
    assert by_chain["ethereum"].address == evm_signer.address and by_chain["ethereum"].can_sign
    assert by_chain["solana"].address == sol_signer.address and by_chain["solana"].can_sign

    eth_quote = _run(lambda: bridge.quote(source_chain="ethereum", source_asset="eth", destination_chain="aleo",
                                          amount="0.00001", recipient=ALEO_RECIPIENT, sender=evm_signer.address))
    assert eth_quote.plan.sender == evm_signer.address and eth_quote.amount_out == "0.00001"
    assert any(fee.asset_id == "ethereum/eth" for fee in eth_quote.fees)
    sol_quote = _run(lambda: bridge.quote(source_chain="solana", source_asset="sol", destination_chain="aleo",
                                          amount="0.0001", recipient=ALEO_RECIPIENT, sender=sol_signer.address))
    assert sol_quote.plan.sender == sol_signer.address and sol_quote.total_lamports > 0


@pytest.mark.skipif(not SIGNING, reason="set BRIDGE_LIVE_REMOTE_WALLET_SIGNING=1 to request real (unbroadcast) signatures")
def test_wallets_sign_remotely_and_the_signatures_verify(provider: str) -> None:
    from eth_account import Account
    from solders.hash import Hash
    from solders.keypair import Keypair
    from solders.message import MessageV0, to_bytes_versioned
    from solders.system_program import TransferParams, transfer

    evm_signer, sol_signer = _signers(provider)
    tx = {"from": evm_signer.address, "to": evm_signer.address, "value": 0, "data": "0x", "chainId": 1,
          "nonce": 0, "gas": 21000}
    tx.update({"gasPrice": 10**9} if evm_signer.legacy_transactions_only
              else {"maxFeePerGas": 2 * 10**9, "maxPriorityFeePerGas": 10**8})
    signed = evm_signer.sign_transaction(tx)                         # never broadcast
    assert Account.recover_transaction(signed.raw_transaction) == evm_signer.address
    assert len(signed.hash) == 32

    message = MessageV0.try_compile(
        sol_signer.pubkey(),
        [transfer(TransferParams(from_pubkey=sol_signer.pubkey(), to_pubkey=Keypair().pubkey(), lamports=1))],
        [], Hash.default())                                           # a zero blockhash can never land
    signature = sol_signer.sign_message(to_bytes_versioned(message))
    assert signature.verify(sol_signer.pubkey(), to_bytes_versioned(message))
