"""Leg 1 of the acceptance matrix: 2 USDC Sepolia -> USDCx aleo-testnet, public mint.

Moves testnet funds. Gated by BRIDGE_LIVE_FUNDS=1, BRIDGE_LIVE_STATE_DIR, SEPOLIA_RPC_URL,
EVM_PRIVATE_KEY, ALEO_E2E_PRIVATE_KEY. Every checkpoint and the final receipt are written to
BRIDGE_LIVE_STATE_DIR (outside the repo) so plan 4's rehearsal runner can `recover`/`wait` and
run leg 2 from them. Keys are read from the environment and never written or printed.

Output layout matches spec §12 / plan 4's rehearsal convention:
  - <BRIDGE_LIVE_STATE_DIR>/checkpoints/<id>.json  — the FileCheckpointStore (one file per receipt id)
  - <BRIDGE_LIVE_STATE_DIR>/leg1.json               — latest checkpoint boundary, overwritten atomically
  - <BRIDGE_LIVE_STATE_DIR>/receipts.jsonl          — one appended line per leg run
"""
import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from aleo_bridge.checkpoint import FileCheckpointStore
from aleo_bridge.eth import Ethereum
from aleo_bridge.types import Status

REQUIRED = ("BRIDGE_LIVE_STATE_DIR", "SEPOLIA_RPC_URL", "EVM_PRIVATE_KEY", "ALEO_E2E_PRIVATE_KEY")
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("BRIDGE_LIVE_FUNDS") != "1" or any(not os.environ.get(v) for v in REQUIRED),
        reason="set BRIDGE_LIVE_FUNDS=1, BRIDGE_LIVE_STATE_DIR, SEPOLIA_RPC_URL, EVM_PRIVATE_KEY, ALEO_E2E_PRIVATE_KEY"),
]


def _atomic_write(path: Path, text: str) -> None:
    """Write *text* to *path* via a same-directory temp file + os.replace (no partial reads)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _bridge():
    from aleo import Aleo, HTTPProvider

    from aleo_bridge import Bridge

    aleo = Aleo(HTTPProvider(os.environ.get("ALEO_ENDPOINT", "https://edge.provable.com/api"), network="testnet"))
    aleo.default_account = aleo.account.from_private_key(os.environ["ALEO_E2E_PRIVATE_KEY"])
    state_dir = Path(os.environ["BRIDGE_LIVE_STATE_DIR"])
    state_dir.mkdir(parents=True, exist_ok=True)
    return Bridge(aleo, ethereum=Ethereum(os.environ["SEPOLIA_RPC_URL"], private_key=os.environ["EVM_PRIVATE_KEY"]),
                  checkpoints=FileCheckpointStore(state_dir / "checkpoints")), state_dir


def test_sepolia_usdc_deposit_public_mint():
    bridge, state_dir = _bridge()
    eth = bridge.eth
    assert eth.conn.chain_id == 11155111 and eth.chain.id == "sepolia"
    recipient = bridge.aleo_address()
    usdc_balance = eth.balance("usdc")
    if usdc_balance < 2_000_000:
        pytest.fail(f"{eth.conn.address} holds {usdc_balance} µUSDC on Sepolia; leg 1 needs 2 USDC plus ETH for gas")
    quote = eth.quote_deposit_usdc(recipient, amount="2", mint_mode="public")
    assert quote.plan.route_id == "xreserve:sepolia/usdc->aleo-testnet/usdcx" and quote.plan.recipient == recipient

    checkpoint_path = state_dir / "leg1.json"

    def save(checkpoint):
        _atomic_write(checkpoint_path, checkpoint.to_json() + "\n")   # latest boundary wins; the store keeps every id

    result = eth.deposit_usdc(recipient, amount="2", mint_mode="public").send(
        on_checkpoint=save, timeout_seconds=240.0, poll_seconds=3.0)
    receipt = result.receipt

    with (state_dir / "receipts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "leg": 1, "route_id": result.route_id, "source_tx": receipt.source_tx_id,
            "message_hash": result.message_hash, "status": receipt.status.value,
            "delivery_confirmed": False, "amount_atomic": quote.plan.amount_atomic,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }) + "\n")

    assert checkpoint_path.exists() and bridge.checkpoints.load(receipt.id) is not None
    assert receipt.status in (Status.ATTESTATION_PENDING, Status.SOURCE_CONFIRMING, Status.SOURCE_APPROVAL_PENDING)
    if receipt.status == Status.ATTESTATION_PENDING:
        assert len(result.message_hash) == 66 and len(result.nonce) == 66
        assert receipt.protocol_state["depositLogIndex"] >= 0 and receipt.protocol_state["mintMode"] == "public"
    else:
        # A timeout is not a failure: the receipt and checkpoint carry the hashes for recover()/source_status().
        refreshed = eth.source_status(quote.plan, receipt)
        assert refreshed.status != Status.FAILED
