"""Devnode AMM fixture: the full shield_swap stack on a local devnode.

Python port of the TS suite's ``devnodeAmm.ts``: boots ``aleo-devnode``,
advances past the last TEST consensus-version height (the devnode's snarkVM
uses the test schedule — V17 from height 20), deploys the vendored AMM +
multisig import and two locally-compiled ARC-20 test tokens, runs the admin
configuration, and funds a non-admin user.  Fully hermetic — no live network.

Two execution ladders (see the suite docstring):

* **proven** (default): every execution transaction is fully proven locally
  via the facade's ``transact`` — slow (SNARK params + key synthesis) but
  exercises exactly what a real network requires.
* **unproven** (``ALEO_DEVNODE_UNPROVEN=1``): executions are built without
  proofs (the devnode skips proof verification) via the SDK's
  ``Execution.from_authorization_unproven`` — fast, devnode-only.

Deployment transactions are always proofless (both ladders): dummy verifying
keys/certificates via ``Deployment.from_program_unproven`` plus an unproven
fee — mirroring the wasm SDK's ``buildDevnodeDeploymentTransaction``.  Real
key synthesis for the AMM takes ~14 minutes and buys nothing on a devnode,
which skips certificate verification anyway.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from aleo_shield_swap._core import ensure_programs

FIXTURES = Path(__file__).parents[1] / "fixtures" / "programs"

AMM_PROGRAM = "shield_swap_v3.aleo"
MULTISIG_PROGRAM = "test_shield_swap_multisig_core.aleo"
TOKEN_A = "test_token_a.aleo"
TOKEN_B = "test_token_b.aleo"

# One million tokens at 6 decimals — the per-user provisioning amount.
TOKEN_SUPPLY = 1_000_000_000_000

# The devnode's snarkVM uses TEST consensus heights: the last version
# activates at height 20.  Advance past it so records and costs are built
# and validated at the same (latest) version on both sides.
LAST_TEST_CONSENSUS_HEIGHT = 20

UNPROVEN = os.environ.get("ALEO_DEVNODE_UNPROVEN") == "1"


def identifier_to_field(identifier: str) -> str:
    """Encodes a program identifier (no ``.aleo``) the way the AMM keys
    tokens: little-endian bytes as a field literal."""
    value = 0
    for byte in reversed(identifier.encode()):
        value = (value << 8) | byte
    return f"{value}field"


def read_fixture(file_name: str) -> str:
    return (FIXTURES / file_name).read_text()


def patch_admin_address(source: str, admin_address: str) -> str:
    """Rewrites the baked deployer/admin address in the AMM source to
    *admin_address* — the constructor promotes that literal to the ``admin``
    mapping at edition 0."""
    constructor = source[source.index("constructor:"):]
    baked = re.search(r"aleo1[a-z0-9]{58}", constructor)
    if not baked:
        raise RuntimeError("No admin address literal found in the AMM constructor")
    return source.replace(baked.group(0), admin_address)


@dataclass
class AmmDevnode:
    """The running fixture: node, facade client, actors, and helpers."""

    devnode: Any
    aleo: Any
    admin: Any                       # facade Account (genesis 0 — token mint admin)
    user: Any                        # facade Account (non-admin, funded)
    imports: dict[str, str]          # program id -> source (dynamic dispatch)
    token0_field: str
    token1_field: str
    token0_program: str
    token1_program: str

    # ── Chain plumbing ───────────────────────────────────────────────────────

    def _net(self) -> Any:
        from aleo import testnet
        return testnet

    def state_root(self) -> str:
        return str(self.aleo.network.get_state_root())

    def submit_and_confirm(self, tx: Any, label: str) -> str:
        """Broadcast a built transaction, mine it, and require acceptance."""
        tx_id = self.aleo.network.submit_transaction(tx)
        self.devnode.advance(1)
        confirmed = self.aleo.network.get_confirmed_transaction(str(tx_id))
        status = str(confirmed)
        if '"accepted"' not in status and "'accepted'" not in status:
            raise RuntimeError(
                f"{label}: transaction {tx_id} was not accepted: {status[:400]}\n"
                f"devnode logs:\n" + "\n".join(self.devnode.logs()[-15:])
            )
        return str(tx_id)

    def wait_queryable(self, program_id: str) -> None:
        """Wait until the devnode serves a deployed program's source."""
        for _ in range(20):
            try:
                src = self.aleo.network.get_program(program_id)
                if f"program {program_id}" in str(src):
                    return
            except Exception:
                pass
            self.devnode.advance(1)
            time.sleep(0.3)
        raise RuntimeError(f"{program_id} never became queryable on the devnode")

    # ── Deployment (unproven fee — devnode only) ─────────────────────────────

    def deploy_program(self, source: str, label: str) -> str:
        """Deploy *source* proofless: dummy verifying keys (no synthesis) and
        an unproven public fee paid by the admin — devnode-only."""
        net = self._net()
        process = self.aleo.process
        program = net.Program.from_source(source)

        deployment = net.Deployment.from_program_unproven(program, self.admin.address)
        cost = process.deployment_cost(deployment)
        fee_auth = process.authorize_fee_public(
            self.admin.private_key, cost, 0, deployment.deployment_id())
        fee = net.Fee.from_authorization_unproven(fee_auth, self.state_root())
        tx = net.Transaction.from_deployment(self.admin.private_key, deployment, fee)
        tx_id = self.submit_and_confirm(tx, f"deploy {label}")
        self.wait_queryable(label)
        # Later deployments/executions resolve this program from the process.
        process.add_program(program)
        return tx_id

    # ── Execution (ladder: proven via facade, or unproven) ──────────────────

    def execute(self, account: Any, program_id: str, function: str,
                inputs: list[Any], label: str) -> str:
        """Run one program function as *account* and require acceptance.

        Proven ladder: the facade's ``build_transaction`` (full local proof).
        Unproven ladder: authorization + proofless execution/fee.
        """
        ensure_programs(self.aleo, [program_id], self.imports)
        bound = self.aleo.programs.get(program_id).functions[function](*inputs)

        if not UNPROVEN:
            result = bound.build_transaction(account)
            return self.submit_and_confirm(result.raw, label)

        net = self._net()
        process = self.aleo.process
        auth = bound.authorize(account).raw
        root = self.state_root()
        execution = net.Execution.from_authorization_unproven(auth, root)
        cost, _ = process.execution_cost(execution)
        fee_auth = process.authorize_fee_public(
            account.private_key, cost, 0, execution.execution_id)
        fee = net.Fee.from_authorization_unproven(fee_auth, root)
        tx = net.Transaction.from_execution(execution, fee)
        return self.submit_and_confirm(tx, label)

    # ── Records ──────────────────────────────────────────────────────────────

    def records_of(self, account: Any, tx_id: str) -> list[str]:
        """Record plaintexts of a confirmed transaction owned by *account*."""
        tx = self.aleo.network.get_transaction_object(tx_id)
        return [str(r) for r in tx.owned_records(account.view_key)]

    def privatize_token(self, account: Any, token_program: str, amount: int) -> str:
        """``transfer_public_to_private`` and return the new Token record."""
        tx_id = self.execute(
            account, token_program, "transfer_public_to_private",
            [str(account.address), f"{amount}u128"],
            f"privatize {amount} of {token_program}",
        )
        records = [r for r in self.records_of(account, tx_id) if "amount" in r]
        if not records:
            raise RuntimeError(
                f"No Token record in {token_program} privatize outputs (tx {tx_id})")
        return records[0]

    def read_mapping(self, mapping: str, key: str) -> Optional[str]:
        from aleo_shield_swap._core import normalize_mapping_value
        raw = self.aleo.programs.get(AMM_PROGRAM).mapping(mapping).get(key)
        return normalize_mapping_value(raw)

    def stop(self) -> None:
        self.devnode.stop()


def setup_amm_devnode() -> AmmDevnode:
    """Boot a devnode with the full AMM stack deployed, configured, funded."""
    from aleo.testing import Devnode

    devnode = Devnode().start()
    aleo = devnode.aleo
    admin = devnode.accounts[0]      # genesis 0 — also the tokens' baked admin
    aleo.default_account = admin
    aleo.record_provider = None      # no scanner on a devnode — explicit records

    # Advance past the last TEST consensus-version height so the node (and
    # everything built against it) runs at the latest consensus version.
    devnode.advance(LAST_TEST_CONSENSUS_HEIGHT + 2)

    multisig_source = read_fixture(MULTISIG_PROGRAM)
    amm_source = patch_admin_address(read_fixture(AMM_PROGRAM), str(admin.address))
    token_a_source = read_fixture(f"{TOKEN_A.removesuffix('.aleo')}.aleo")
    token_b_source = read_fixture(f"{TOKEN_B.removesuffix('.aleo')}.aleo")
    credits_source = str(aleo.network.get_program("credits.aleo"))

    imports = {
        MULTISIG_PROGRAM: multisig_source,
        AMM_PROGRAM: amm_source,
        TOKEN_A: token_a_source,
        TOKEN_B: token_b_source,
        "credits.aleo": credits_source,
    }

    a_field = identifier_to_field("test_token_a")
    b_field = identifier_to_field("test_token_b")
    a_first = int(a_field.removesuffix("field")) < int(b_field.removesuffix("field"))

    ctx = AmmDevnode(
        devnode=devnode, aleo=aleo, admin=admin, user=None, imports=imports,
        token0_field=a_field if a_first else b_field,
        token1_field=b_field if a_first else a_field,
        token0_program=TOKEN_A if a_first else TOKEN_B,
        token1_program=TOKEN_B if a_first else TOKEN_A,
    )

    # Deploy in dependency order; the AMM statically imports the multisig.
    ctx.deploy_program(multisig_source, MULTISIG_PROGRAM)
    ctx.deploy_program(amm_source, AMM_PROGRAM)
    ctx.deploy_program(token_a_source, TOKEN_A)
    ctx.deploy_program(token_b_source, TOKEN_B)

    # Admin configuration: fee tiers, spacings, bindings, token registration.
    for label, function, inputs in [
        ("add_fee_tier 3000", "add_fee_tier", ["3000u16"]),
        ("add_fee_tier 500", "add_fee_tier", ["500u16"]),
        ("add_tick_spacing 60", "add_tick_spacing", ["60u32"]),
        ("add_tick_spacing 10", "add_tick_spacing", ["10u32"]),
        ("bind 3000->60", "bind_fee_to_tick_spacing", ["3000u16", "60u32"]),
        ("bind 500->10", "bind_fee_to_tick_spacing", ["500u16", "10u32"]),
        ("decimals A", "set_token_decimals", [a_field, "6u8"]),
        ("decimals B", "set_token_decimals", [b_field, "6u8"]),
        ("allow A", "allow_token", [a_field]),
        ("allow B", "allow_token", [b_field]),
        ("open pool creation", "set_pool_creation_is_open", ["true"]),
    ]:
        ctx.execute(admin, AMM_PROGRAM, function, inputs, f"admin {label}")

    # A non-admin user proves the open-pool-creation gate: fund fees, mint
    # both tokens to it publicly.
    user = aleo.account.create()
    ctx.user = user
    ctx.execute(admin, "credits.aleo", "transfer_public",
                [str(user.address), "100000000u64"], "fund user")
    for token in (TOKEN_A, TOKEN_B):
        ctx.execute(admin, token, "mint_public",
                    [str(user.address), f"{TOKEN_SUPPLY}u128"], f"mint {token}")

    return ctx
