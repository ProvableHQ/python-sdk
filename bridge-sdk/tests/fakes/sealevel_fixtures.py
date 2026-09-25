"""Golden fixtures from veil's test/fixtures (mainnet TransferRemote + inner IGP account).

Pure: no solders import, so the _sealevel layout tests run on an Aleo-only install.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TRANSFER: dict = json.loads((FIXTURES / "sealevel-transfer-remote.json").read_text())
IGP: dict = json.loads((FIXTURES / "sealevel-igp-account.json").read_text())

WARP_PROGRAM_ADDRESS = "8YGT2pZwyZe94qBpGzWfY2TMEVcwaQ1bXAE7YAgpUaM7"
ALEO_MAINNET_DOMAIN = 1634493807
DESTINATION_GAS_AMOUNT = 464_000
EXPECTED_IGP_PAYMENT_LAMPORTS = 2_900_000
NETWORK_FEE_LAMPORTS = 10_000
GAS_PAYMENT_RENT_LAMPORTS = 1_872_240
DISPATCHED_MESSAGE_RENT_LAMPORTS = 2_241_120
FEE_PAYER_RENT_LAMPORTS = 890_880
EXPECTED_MESSAGE_ID = "0xffe0409d00c184769b4dfa2a1eaac5a0a79bfe52458a38e1d9a71a9e5c677805"


def igp_account_data() -> bytes:
    return base64.b64decode(IGP["dataBase64"])


def metadata_from_fixture(*, overhead: bool = True) -> dict[str, str | int | bool]:
    """Route metadata (veil camelCase keys) read off the fixture's ordered account list (SEALEVEL_NOTES §2)."""
    accounts = TRANSFER["accounts"]
    metadata: dict[str, str | int | bool] = {
        "warpProgramAddress": WARP_PROGRAM_ADDRESS,
        "tokenPda": accounts[2]["address"],
        "nativeCollateralPda": accounts[15]["address"],
        "dispatchAuthorityPda": accounts[5]["address"],
        "mailboxProgramAddress": accounts[3]["address"],
        "mailboxOutboxPda": accounts[4]["address"],
        "igpProgramAddress": accounts[9]["address"],
        "igpProgramDataPda": accounts[10]["address"],
        "igpAccount": accounts[13]["address"],
        "splNoopProgramAddress": accounts[1]["address"],
        "destinationDomain": ALEO_MAINNET_DOMAIN,
        "destinationGasAmount": str(DESTINATION_GAS_AMOUNT),
        "registryCommit": "418056e21734d26a7d14692e0ec5e902cc9e86bf",
        "solanaReviewedAt": "2026-08-28T00:00:00Z",
        "solanaConfigSource": "hyperlane-registry@418056e2:deployments/warp_routes/SOL/aleo-config.yaml",
    }
    if overhead:
        metadata["igpOverheadAccount"] = accounts[12]["address"]
    return metadata
