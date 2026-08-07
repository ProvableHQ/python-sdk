"""Devnode fixture for freeze-list Merkle exclusion proofs.

Boots ``aleo-devnode``, deploys the shield_swap stack **fetched live from the
network** rather than from repo fixtures — the deployed programs are the
authoritative statement of what the verifier accepts, and a vendored copy can
drift from them silently.  Then it populates the freeze list with real
addresses and leaves the stack ready for a transition that must prove
non-inclusion.

This is what makes the proofs meaningful: against an *empty* freeze list the
all-zero placeholder literal verifies, so only a populated list exercises
:class:`aleo.MerkleExclusionProof` for real.

Deployment is proofless (dummy verifying keys, unproven fee), as in
``devnode_amm``: synthesizing real keys for the AMM takes many minutes and buys
nothing on a node that skips certificate verification.

Execution ladders:

* **unproven** (default here): the authorization still *evaluates* the
  transition, so ``verify_merkle_non_inclusion`` and its asserts run in full —
  only the SNARK is skipped.  The finalize, including
  ``assert_valid_freeze_list_root``, runs on-chain either way.  Both halves of
  the proof are therefore checked.
* **proven** (``ALEO_DEVNODE_MERKLE_PROVEN=1``): full local proving, much
  slower.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from aleo import MerkleExclusionProof

#: The node API is the source of truth for deployed program source.
NETWORK_API = os.environ.get(
    "ALEO_DEVNODE_SOURCE_API", "https://api.provable.com/v2/testnet")

AMM_PROGRAM = "shield_swap.aleo"
FREEZELIST_PROGRAM = "shield_swap_freezelist.aleo"
AMM_MULTISIG = "shield_swap_multisig_core.aleo"
TOKEN_MULTISIG = "test_arc20_multisig_core.aleo"
TOKEN0_PROGRAM = "test_arc20_usdc.aleo"
TOKEN1_PROGRAM = "test_arc20_usdt.aleo"

#: Deployment order — imports before importers.
DEPLOY_ORDER = [
    AMM_MULTISIG,
    FREEZELIST_PROGRAM,
    AMM_PROGRAM,
    TOKEN_MULTISIG,
    TOKEN0_PROGRAM,
    TOKEN1_PROGRAM,
]

#: Administrator literals baked into the deployed programs.  On a devnode these
#: keys do not exist, so every occurrence is rewritten to a genesis account.
#: The AMM and its freeze list share one; the test ARC-20s share another.
BAKED_ADMINS = [
    "aleo1z3zwzgpgakk89xpknync5rtklkjkyv33g7cvaqe0gku64zs3lv9qyux0qc",
    "aleo1axurgcdhztu8m23ttzju38qzchtzs8kyk7nga9n58zyrmnxzmuqqf6wqdc",
]

#: Freeze-list role bits: 8 grants roles, 16 edits the list.
FREEZELIST_MANAGER_ROLE = 24
#: ARC-20 role bits: 8 grants roles, 1 mints.
TOKEN_MINTER_ROLE = 9

#: Blocks the previous freeze-list root stays valid after a rotation.
FREEZELIST_WINDOW = 100
#: The devnode's snarkVM uses TEST consensus heights; the last activates at 20.
LAST_TEST_CONSENSUS_HEIGHT = 20

TOKEN_SUPPLY = 1_000_000_000_000

PROVEN = os.environ.get("ALEO_DEVNODE_MERKLE_PROVEN") == "1"


def identifier_to_field(identifier: str) -> str:
    """A program identifier (no ``.aleo``) as the field the AMM keys tokens by.

    Little-endian bytes, verified against every entry of the live token
    registry.
    """
    value = 0
    for byte in reversed(identifier.encode()):
        value = (value << 8) | byte
    return f"{value}field"


def fetch_program(program_id: str) -> str:
    """Deployed source for *program_id*, straight from the node API."""
    response = requests.get(f"{NETWORK_API}/program/{program_id}", timeout=60)
    response.raise_for_status()
    source = response.json()
    if not isinstance(source, str) or f"program {program_id}" not in source:
        raise RuntimeError(f"{program_id} did not come back as program source")
    return source


def repoint_admins(source: str, admin_address: str) -> str:
    """Rewrite every baked administrator literal to *admin_address*.

    Both the constructor gate and the finalize gates (``initialize``,
    ``initialize_token``) compare against the same literal, so all occurrences
    have to move together or the program deploys but cannot be configured.
    """
    for baked in BAKED_ADMINS:
        source = source.replace(baked, admin_address)
    return source


@dataclass
class MerkleDevnode:
    """A devnode running the network's shield_swap stack with a live freeze list."""

    devnode: Any
    aleo: Any
    admin: Any
    user: Any
    sources: dict[str, str]
    token0_id: str
    token1_id: str
    token0_program: str
    token1_program: str
    #: Addresses currently frozen, in insertion order.
    frozen: list[str] = field(default_factory=list)
    merkle: MerkleExclusionProof = field(default_factory=MerkleExclusionProof)

    # ── Chain plumbing ──────────────────────────────────────────────────────

    def _net(self) -> Any:
        from aleo import testnet
        return testnet

    def state_root(self) -> str:
        return str(self.aleo.network.get_state_root())

    def submit_and_confirm(self, tx: Any, label: str) -> str:
        tx_id = self.aleo.network.submit_transaction(tx)
        self.devnode.advance(1)
        confirmed = str(self.aleo.network.get_confirmed_transaction(str(tx_id)))
        if '"accepted"' not in confirmed and "'accepted'" not in confirmed:
            raise RuntimeError(
                f"{label}: transaction {tx_id} was not accepted: "
                f"{confirmed[:400]}\ndevnode logs:\n"
                + "\n".join(self.devnode.logs()[-15:])
            )
        return str(tx_id)

    def wait_queryable(self, program_id: str) -> None:
        for _ in range(20):
            try:
                if f"program {program_id}" in str(
                        self.aleo.network.get_program(program_id)):
                    return
            except Exception:
                pass
            self.devnode.advance(1)
            time.sleep(0.3)
        raise RuntimeError(f"{program_id} never became queryable")

    def deploy_program(self, program_id: str) -> str:
        """Deploy proofless: dummy verifying keys and an unproven public fee.

        The devnode ships its own snarkVM, which need not price deployments
        identically to the bindings computing the fee here.  When the node
        rejects the base fee it names the amount it wants, so a short fee is
        retried at that figure rather than guessing a margin.
        """
        net = self._net()
        process = self.aleo.process
        program = net.Program.from_source(self.sources[program_id])
        deployment = net.Deployment.from_program_unproven(program, self.admin.address)

        cost = process.deployment_cost(deployment)
        for attempt in range(2):
            fee_auth = process.authorize_fee_public(
                self.admin.private_key, cost, 0, deployment.deployment_id())
            fee = net.Fee.from_authorization_unproven(fee_auth, self.state_root())
            tx = net.Transaction.from_deployment(
                self.admin.private_key, deployment, fee)
            try:
                tx_id = self.submit_and_confirm(tx, f"deploy {program_id}")
                break
            except Exception as exc:
                required = re.search(r"requires (\d+) microcredits", str(exc))
                if attempt or not required:
                    raise
                cost = int(required.group(1))

        self.wait_queryable(program_id)
        # Registers the program for later executions and dynamic dispatch.
        process.add_program(program)
        return tx_id

    def execute(self, account: Any, program_id: str, function: str,
                inputs: list[Any], label: str) -> str:
        """Run one transition as *account* and require on-chain acceptance."""
        bound = self.aleo.programs.get(program_id).functions[function](*inputs)

        if PROVEN:
            return self.submit_and_confirm(
                bound.build_transaction(account).raw, label)
        # authorize() evaluates the transition, so in-circuit asserts — the
        # Merkle verification included — run here even though no SNARK follows.
        return self.submit_and_confirm(self._unproven_tx(bound, account), label)

    def submit_call(self, call: Any, account: Any, label: str) -> str:
        """Drive a ``DexCall`` through the same ladder as :meth:`execute`.

        Used for the transitions that take no Merkle proof (pool creation), so
        those keep going through the shipped client rather than hand-built
        inputs.
        """
        if PROVEN:
            result = call.transact(account)
            self.devnode.advance(1)
            return str(result.transaction_id)
        return self.submit_and_confirm(
            self._unproven_tx(call._bound, account), label)

    def _unproven_tx(self, bound: Any, account: Any) -> Any:
        """A proofless execution transaction for an already-bound call."""
        net = self._net()
        process = self.aleo.process
        auth = bound.authorize(account).raw
        root = self.state_root()
        execution = net.Execution.from_authorization_unproven(auth, root)
        cost, _ = process.execution_cost(execution)
        fee_auth = process.authorize_fee_public(
            account.private_key, cost, 0, execution.execution_id)
        fee = net.Fee.from_authorization_unproven(fee_auth, root)
        return net.Transaction.from_execution(execution, fee)

    def records_of(self, account: Any, tx_id: str) -> list[str]:
        tx = self.aleo.network.get_transaction_object(tx_id)
        return [str(r) for r in tx.owned_records(account.view_key)]

    def privatize_token(self, account: Any, token_program: str,
                        amount: int) -> str:
        tx_id = self.execute(
            account, token_program, "transfer_public_to_private",
            [str(account.address), f"{amount}u128"],
            f"privatize {amount} {token_program}")
        records = [r for r in self.records_of(account, tx_id) if "amount" in r]
        if not records:
            raise RuntimeError(f"no Token record from {token_program} (tx {tx_id})")
        return records[0]

    def read_mapping(self, program_id: str, mapping: str, key: str) -> Optional[str]:
        from aleo_shield_swap._core import normalize_mapping_value
        return normalize_mapping_value(
            self.aleo.programs.get(program_id).mapping(mapping).get(key))

    # ── Freeze list ─────────────────────────────────────────────────────────

    def on_chain_root(self) -> str:
        """The freeze list's current root, as the AMM's finalize reads it."""
        root = self.read_mapping(FREEZELIST_PROGRAM, "freeze_list_root", "1u8")
        if root is None:
            raise RuntimeError("freeze list has no root — initialize first")
        return root

    def local_tree(self) -> list[int]:
        """The tree over the currently frozen addresses, built locally."""
        return self.merkle.build_tree(
            self.merkle.leaves_from_addresses(self.frozen))

    def freeze(self, address: str) -> str:
        """Add *address* to the freeze list, rotating the root to match.

        The contract stores whatever root the manager hands it, so this is
        where the locally computed root becomes the one every later proof must
        reproduce.  Entries occupy indices from 1 upward; index 0 is the zero
        address written at initialization.
        """
        previous_root = self.on_chain_root()
        self.frozen.append(address)
        new_root = self.merkle.root(self.local_tree())
        tx_id = self.execute(
            self.admin, FREEZELIST_PROGRAM, "update_freeze_list",
            [address, "true", f"{len(self.frozen)}u32", previous_root, new_root],
            f"freeze {address[:12]}…")
        return tx_id

    def exclusion_proof(self, address: str) -> str:
        """A ``[MerkleProof; 2]`` literal proving *address* is not frozen."""
        return self.merkle.exclusion_proof(self.local_tree(), address)

    def stop(self) -> None:
        self.devnode.stop()


def setup_merkle_devnode(freeze_count: int = 5) -> MerkleDevnode:
    """Boot a devnode with the network's stack deployed and a populated list.

    Args:
        freeze_count: How many generated addresses to put on the freeze list.
            Anything above zero makes the placeholder proof invalid, which is
            the point.
    """
    from aleo.testing import Devnode

    devnode = Devnode().start()
    aleo = devnode.aleo
    admin = devnode.accounts[0]
    aleo.default_account = admin
    aleo.record_provider = None          # no scanner on a devnode
    devnode.advance(LAST_TEST_CONSENSUS_HEIGHT + 2)

    sources = {
        pid: repoint_admins(fetch_program(pid), str(admin.address))
        for pid in DEPLOY_ORDER
    }

    usdc_id = identifier_to_field(TOKEN0_PROGRAM.removesuffix(".aleo"))
    usdt_id = identifier_to_field(TOKEN1_PROGRAM.removesuffix(".aleo"))
    # Pools order their sides by token id, not by name.
    usdc_first = int(usdc_id.removesuffix("field")) < int(usdt_id.removesuffix("field"))

    ctx = MerkleDevnode(
        devnode=devnode, aleo=aleo, admin=admin, user=None, sources=sources,
        token0_id=usdc_id if usdc_first else usdt_id,
        token1_id=usdt_id if usdc_first else usdc_id,
        token0_program=TOKEN0_PROGRAM if usdc_first else TOKEN1_PROGRAM,
        token1_program=TOKEN1_PROGRAM if usdc_first else TOKEN0_PROGRAM,
    )

    for program_id in DEPLOY_ORDER:
        ctx.deploy_program(program_id)

    # Freeze list: initialize grants the admin role 8 (grant roles only), so it
    # has to promote itself to 24 before it can edit the list.
    ctx.execute(admin, FREEZELIST_PROGRAM, "initialize",
                [str(admin.address), f"{FREEZELIST_WINDOW}u32"],
                "freezelist initialize")
    ctx.execute(admin, FREEZELIST_PROGRAM, "update_role",
                [str(admin.address), f"{FREEZELIST_MANAGER_ROLE}u16"],
                "freezelist grant manager role")

    # Tokens: same shape — initialize grants role 8, minting needs bit 1.
    for token_program in (TOKEN0_PROGRAM, TOKEN1_PROGRAM):
        ctx.execute(admin, token_program, "initialize_token",
                    [str(admin.address)], f"initialize {token_program}")
        ctx.execute(admin, token_program, "update_role",
                    [str(admin.address), f"{TOKEN_MINTER_ROLE}u16"],
                    f"grant minter on {token_program}")

    # AMM admin configuration, mirroring the deployed testnet parameters.
    for label, function, inputs in [
        ("fee tier 3000", "add_fee_tier", ["3000u16"]),
        ("tick spacing 60", "add_tick_spacing", ["60u32"]),
        ("bind 3000->60", "bind_fee_to_tick_spacing", ["3000u16", "60u32"]),
        ("allow token0", "allow_token", [ctx.token0_id, ctx.token0_id]),
        ("allow token1", "allow_token", [ctx.token1_id, ctx.token1_id]),
        ("open pool creation", "set_pool_creation_is_open", ["true"]),
    ]:
        ctx.execute(admin, AMM_PROGRAM, function, inputs, f"admin {label}")

    # A funded non-admin does the proving work, so the signer under test is not
    # the same account that administers the list.
    user = aleo.account.create()
    ctx.user = user
    ctx.execute(admin, "credits.aleo", "transfer_public",
                [str(user.address), "100000000u64"], "fund user")
    for token_program in (TOKEN0_PROGRAM, TOKEN1_PROGRAM):
        ctx.execute(admin, token_program, "mint_public",
                    [str(user.address), f"{TOKEN_SUPPLY}u128"],
                    f"mint {token_program} to user")

    # Populate the list with addresses that are NOT the user, so the user can
    # still prove non-inclusion against a genuinely non-empty tree.
    for _ in range(freeze_count):
        ctx.freeze(str(aleo.account.create().address))

    return ctx
