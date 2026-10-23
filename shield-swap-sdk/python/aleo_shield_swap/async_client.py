"""AsyncShieldSwap — the shield_swap client over an ``AsyncAleo`` facade.

Reads, balances, and the two-transaction private-swap flow of
:class:`~aleo_shield_swap.client.ShieldSwap`, with every I/O method
``async``.  Liquidity methods are sync-client-only for now.  Pure logic is
shared from ``_core``/``derivations`` — only the I/O differs.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import math
import time
import logging
from decimal import Decimal
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar

from aleo import AleoNetworkError, ProgramNotFound
from aleo.compliance import AsyncFreezeListProofs, ComplianceProofContext, async_freezelist_program
from aleo.inventory import (
    AsyncRecordInventory, InventoryPlan, InventoryResult, InventoryTarget, RecordAsset, RecordReservationConflictError,
    managed_inventory,
)

from . import _generated as g
from ._calls import RecordReselection, extract_tx_id, root_outputs
from ._core import (
    reserved_nonces,
    swap_context_guard,
    token_record_candidates,
    underlying_program_for,
    _amount_to_base_units,
    decode_position_record,
    default_merkle_proofs,
    find_position_plaintext,
    mapping_flag_set,
    normalize_mapping_value,
    generate_swap_nonce,
    parse_token_record_info,
    parse_unsigned_literal,
    pick_covering_record,
    program_imports,
    record_plaintext,
    register_program_sources,
    resolve_swap_params,
    unreadable_airdrop_transactions,
)
from ._routing import claim_route, swap_route
from .compliance import (
    SHIELD_SWAP_FREEZELIST, ClaimProofs, Slot, SwapProofs, check_proof_arguments, claim_slots,
    merge_legacy_wrapper_proofs, planned_proof_slots, supplied_literals, swap_slots,
)
from .rebalance import RebalancePlan, plan_rebalance as _plan_rebalance
from .position_math import (
    amounts_for_liquidity,
    fee_growth_inside,
    fee_owed,
    u256_of,
)
from .api import AsyncApiClient, ConfirmAirdropResult, api_url_for
from .tick_math import get_sqrt_price_at_tick_x128, int_to_u256_plaintext
from .derivations import (
    BlindedIdentity,
    BlindedIdentityCache,
    derive_pool_key as _derive_pool_key,
    derive_tick_key as _derive_tick_key,
    find_unused_counter_async,
)
from .errors import (
    InsufficientRecordsError,
    PoolNotFoundError,
    PoolNotInitializedError,
    ShieldSwapError,
    SwapOutputNotFinalizedError,
)
from ._quotes import quote_request, build_quote, quote_amounts, resolve_quote, multi_hop_inputs
from .types import (
    SwapQuote,
    ClaimResult,
    OwnedPosition,
    OwnedPositionState,
    SlotView,
    SwapExecution,
    SwapHandle,
)

_log = logging.getLogger(__name__)

R = TypeVar("R")


T = TypeVar("T")


class AsyncDexCall(Generic[R]):
    """Async mirror of :class:`~aleo_shield_swap._calls.DexCall`."""

    def __init__(self, aleo: Any, bound: Any,
                 build_result: Callable[[str, list[Any]], R], *,
                 reselect: Optional[RecordReselection] = None) -> None:
        self._aleo = aleo
        self._bound = bound
        self._build = build_result
        self._reselect = reselect

    def simulate(self, account: Any = None) -> Any:
        """Local authorization — no proof, no send; inspect before spending.

        Synchronous even on the async client: ``AsyncBoundCall.simulate`` only
        authorizes locally, so there is nothing to await.
        """
        return self._bound.simulate(account)

    async def _submit(self, submit: Callable[[Any], Awaitable[T]]) -> T:
        """Run *submit* on the bound call, moving to the next candidate record on a reservation conflict."""
        while True:
            try:
                return await submit(self._bound)
            except RecordReservationConflictError as exc:
                replacement = self._reselect.next_bound(exc) if self._reselect is not None else None
                if replacement is None:
                    raise
                self._bound = replacement

    async def transact(self, account: Any = None, *, wait: bool = False,
                       wait_timeout: float = 180.0, **fee_kwargs: Any) -> R:
        """Prove locally, broadcast, and build the typed result.

        Root-transition outputs are harvested from the built transaction
        before broadcast, so the result is complete without waiting for
        confirmation. Set ``wait=True`` to wait for acceptance on chain.
        With record management enabled on the facade, inputs are reserved
        before proving and an automatically selected record is replaced on
        a reservation conflict; see :meth:`~aleo_shield_swap._calls.DexCall.transact`.
        """
        async def submit(bound: Any) -> tuple[str, Any]:
            if managed_inventory(self._aleo) is not None:
                return await bound.transact_with_result(account, **fee_kwargs)
            tx = await bound.build_transaction(account, **fee_kwargs)
            await self._aleo.network.submit_transaction(tx.raw)
            return tx.id, tx

        tx_id, tx = await self._submit(submit)
        outputs = root_outputs(tx.decoded(), self._bound.program_id,
                               self._bound.function_name)
        result = self._build(tx_id, outputs)
        if wait:
            await self._aleo.network.wait_for_transaction(tx_id, timeout=wait_timeout)
        return result

    async def delegate(self, account: Any = None, **fee_kwargs: Any) -> R:
        """Delegate proving to the DPS (fee master pays by default) and build
        the typed result from the root transition.

        Outputs are harvested from the DPS payload itself when it carries
        the full transaction, so ``wait=False`` returns as soon as the
        broadcast is accepted — callers that read chain state later (e.g.
        ``collect_all``) don't need to block on confirmation here. With
        record management enabled, an automatically selected record is
        replaced on a reservation conflict before proving.
        """
        async def submit(bound: Any) -> Any:
            return await bound.delegate(account, **fee_kwargs)

        payload = await self._submit(submit)
        tx_id = extract_tx_id(payload)
        await self._aleo.network.wait_for_transaction(tx_id)
        tx = await self._aleo.network.get_transaction_object(tx_id)
        decoded = [
            {"program": str(t.program_id), "function": str(t.function_name),
             "outputs": list(t.outputs())}
            for t in tx.transitions()
        ]
        outputs = root_outputs(decoded, self._bound.program_id,
                               self._bound.function_name)
        return self._build(tx_id, outputs)


class AsyncShieldSwap:
    """Typed async client for the shield_swap AMM over ``AsyncAleo``."""

    def __init__(self, aleo: Any, *, program: str = g.PROGRAM_ID,
                 api_url: Optional[str] = None) -> None:
        self._aleo = aleo
        self.program = program
        # Resolve the API from the bound client's network (see api_url_for).
        self.api = AsyncApiClient(api_url or api_url_for(aleo.network_name))
        #: The freeze list Shield Swap itself checks; override for a non-standard deployment.
        self.amm_freezelist_program: str = SHIELD_SWAP_FREEZELIST
        #: Freeze-list exclusion proofs, cached per list until its root changes.
        self.freezelist = AsyncFreezeListProofs(aleo)
        #: Private record inventory of the bound client; see :attr:`ShieldSwap.inventory`.
        self.inventory: AsyncRecordInventory = (
            aleo.inventory if hasattr(aleo, "inventory") else AsyncRecordInventory(aleo)
        )
        # allow_token relationships are immutable — cache probes for the
        # client's lifetime.
        self._wrapped_cache: dict[str, bool] = {}
        self._program_handles: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"AsyncShieldSwap(program={self.program!r}, api={self.api.base_url!r})"

    # ── Mapping plumbing ─────────────────────────────────────────────────────

    async def _program(self, program_id: str) -> Any:
        """Bound program handle, fetched once — see ``ShieldSwap._program``."""
        handle = self._program_handles.get(program_id)
        if handle is None:
            handle = await self._aleo.programs.get(program_id)
            self._program_handles[program_id] = handle
        return handle

    async def _mapping_value(self, mapping: str, key: str) -> Optional[str]:
        prog = await self._program(self.program)
        raw = await prog.mapping(mapping).get(key)
        return normalize_mapping_value(raw)

    # ── Chain reads ──────────────────────────────────────────────────────────

    async def get_pool(self, pool_key: str) -> g.PoolState:
        """Static pool configuration (token pair, fee, decimal scales)."""
        raw = await self._mapping_value("pools", pool_key)
        if raw is None:
            raise PoolNotFoundError(pool_key)
        return g.PoolState.from_plaintext(raw)

    async def get_slot(self, pool_key: str) -> SlotView:
        """Live trading state (sqrt price, tick, in-range liquidity).

        Raises :class:`PoolNotFoundError` when the pool does not exist, or
        :class:`PoolNotInitializedError` when it exists but has no slot yet.
        """
        raw = await self._mapping_value("slots", pool_key)
        if raw is None:
            if await self._mapping_value("pools", pool_key) is not None:
                raise PoolNotInitializedError(pool_key)
            raise PoolNotFoundError(pool_key)
        return SlotView(g.Slot.from_plaintext(raw))

    async def get_swap_output(self, swap: "SwapHandle | str") -> g.SwapOutput:
        """Chain-computed output of a finalized swap request.

        Accepts the :class:`SwapHandle` from ``swap()`` or a bare swap id.
        Raises :class:`SwapOutputNotFinalizedError` when the entry is absent —
        not finalized yet (retry after a few blocks) or already claimed.
        """
        swap_id = swap.swap_id if isinstance(swap, SwapHandle) else swap
        if not swap_id:
            raise ValueError("SwapHandle has no swap_id yet — wait for the "
                             "request transaction and recover it first.")
        raw = await self._mapping_value("swap_outputs", swap_id)
        if raw is None:
            raise SwapOutputNotFinalizedError(swap_id)
        return g.SwapOutput.from_plaintext(raw)

    async def get_swap_execution(self, swap: "SwapHandle | str") -> Optional[SwapExecution]:
        """Per-hop fill receipt — see :meth:`ShieldSwap.get_swap_execution`.
        The hop reads run concurrently."""
        import asyncio
        swap_id = swap.swap_id if isinstance(swap, SwapHandle) else swap
        if not swap_id:
            raise ValueError("SwapHandle has no swap_id yet — wait for the "
                             "request transaction and recover it first.")
        header = await self._mapping_value("swap_execution_headers", swap_id)
        if header is None:
            return None
        count = g.SwapExecutionHeader.from_plaintext(header).hop_count
        hops = await asyncio.gather(*(
            self._mapping_value("swap_execution_hops", SwapExecution.hop_key(swap_id, i))
            for i in range(count)))
        return SwapExecution.from_plaintexts(swap_id, header, list(hops))

    async def get_pool_creator(self, pool_key: str) -> Optional[str]:
        """Creator address of *pool_key* — see :meth:`ShieldSwap.get_pool_creator`."""
        return await self._mapping_value("pool_creators", pool_key)

    async def is_pool_initialized(self, pool_key: str) -> bool:
        """True once *pool_key* has been initialized on chain. Reads a mapping.

        A pool must be initialized before it will accept swaps or liquidity, so
        check this before quoting against a key you derived rather than listed.
        An absent mapping entry reads as False, which is also what an unknown key
        gives you — this does not distinguish the two.
        """
        raw = await self._mapping_value("initialized_pools", pool_key)
        return raw is not None and "true" in raw

    # ── Pure derivations (sync — no I/O) ─────────────────────────────────────

    def derive_pool_key(self, token0: str, token1: str, fee: int) -> str:
        """Compute the pool key for a token pair and fee tier. Local — no network.

        Deriving a key does not mean the pool exists; pass the result to
        :meth:`is_pool_initialized` before trading against it. The token order
        matters, and the derivation is network-scoped, so a key computed for one
        network will not match the other.

        Args:
            token0: First token id of the pair.
            token1: Second token id of the pair.
            fee: Fee tier in the contract's ``u16`` units.

        Returns:
            The pool key as a ``field`` literal.
        """
        return _derive_pool_key(token0, token1, fee, network=self._aleo.network_name)

    def derive_tick_key(self, pool_key: str, tick: int) -> str:
        """Compute the key of one tick within a pool. Local — no network.

        Args:
            pool_key: Pool the tick belongs to.
            tick: Signed tick index.

        Returns:
            The tick key as a ``field`` literal, usable to read the tick's
            on-chain state.
        """
        return _derive_tick_key(pool_key, tick, network=self._aleo.network_name)

    # ── Async record/identity/imports helpers ───────────────────────────────

    async def _next_blinded_identity(self, account: Any, *,
                                     start_counter: int = 0) -> BlindedIdentity:
        """The sync client's probe, awaited: the same galloping search over
        ``used_blinded_addresses`` (an account with a long swap history is
        O(log n) probes, not a fixed 64-counter window that then gives up),
        straight to the node's mapping endpoint — no program download."""
        identities = BlindedIdentityCache(self._aleo.network_name, account, self.program)

        async def is_used(counter: int) -> bool:
            raw = await self._aleo.network.get_program_mapping_value(
                self.program, "used_blinded_addresses", identities.at(counter).blinded_address)
            return mapping_flag_set(raw)

        try:
            counter = await find_unused_counter_async(is_used, start_counter=start_counter)
        except LookupError as exc:
            raise ValueError(f"{exc} for {self.program} — wrong program or scan range?") from None
        return identities.at(counter)

    async def confirm_airdrop(self, *, timeout: float = 600.0, poll_interval: float = 5.0,
                              account: Any = None) -> ConfirmAirdropResult:
        """Fund the account and wait for its airdrop records to become spendable.

        Requires API authentication and scanner registration. ``timeout`` covers
        faucet settlement and scanning. Check ``success`` and ``error`` before
        swapping. A record timeout retains the transfer results; inspect those
        transactions before requesting another airdrop. Scanner errors propagate.
        """
        if any(not math.isfinite(v) or v < 0 for v in (timeout, poll_interval)):
            raise ValueError("timeout and poll_interval must be finite and nonnegative")
        acct = self._account(account)
        provider = self._aleo.record_provider
        if provider is None:
            raise InsufficientRecordsError("No record provider configured")
        deadline = time.monotonic() + timeout
        funding = await self.api.confirm_airdrop(str(acct.address), timeout=timeout, poll_interval=poll_interval)
        if not funding.success:
            return funding
        if any(not result.tx_id for result in funding.job.results):
            return replace(funding, status="records_pending", message="Airdrop returned no transaction ID for a transfer; cannot confirm its record")
        transaction_ids = {result.tx_id for result in funding.job.results}
        while True:
            records = await provider.find(acct, unspent=True)
            pending = unreadable_airdrop_transactions(transaction_ids, records)
            if not pending:
                return funding
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return replace(funding, status="records_pending", message=
                               f"Airdrop accepted but records are not available for: {', '.join(sorted(pending))}. Inspect these transfers before requesting another airdrop.")
            await asyncio.sleep(min(poll_interval, remaining))

    async def has_swap_balance(self, token_in_id: str, amount_in: int | str | Decimal,
                               account: Any = None) -> bool:
        """Check for one covering record; use token units for strings/Decimal.

        Does not reserve records. Scanner errors propagate.
        """
        acct = self._account(account)
        provider = self._aleo.record_provider
        if provider is None:
            raise InsufficientRecordsError("No record provider configured")
        tokens = await self.api.get_tokens() if isinstance(amount_in, (str, Decimal)) else []
        amount = _amount_to_base_units(amount_in, token_in_id, tokens)
        program = await self._token_program(token_in_id)
        records = await provider.find(acct, program=program, unspent=True)
        return pick_covering_record(records, min_amount=amount, token_id=token_in_id,
                                    exclude_nonces=reserved_nonces(self._aleo, acct)) is not None

    async def _select_token_records(self, *, program: str, min_amount: int,
                                    token_id: Optional[str], account: Any) -> list[str]:
        """Eligible, unreserved record plaintexts covering *min_amount*, smallest first, from one scan."""
        provider = self._aleo.record_provider
        if provider is None:
            raise InsufficientRecordsError(
                "No record provider configured — pass token_record= explicitly.")
        records = await provider.find(account, program=program, unspent=True)
        candidates = token_record_candidates(records, min_amount=min_amount, token_id=token_id,
                                             exclude_nonces=reserved_nonces(self._aleo, account))
        if not candidates:
            raise InsufficientRecordsError(
                f"No unspent {program} record covers {min_amount} "
                f"(token_id={token_id or 'any'}) — privatize funds or pass token_record=.")
        return candidates

    async def _select_position_record(self, pool_key: str, account: Any,
                                      position_token_id: Optional[str] = None) -> str:
        records = await self._aleo.record_provider.find(
            account, program=self.program, unspent=True)
        plaintext = find_position_plaintext(records, pool_key, position_token_id)
        if plaintext is None:
            raise InsufficientRecordsError(
                f"No unspent PositionNFT record for pool {pool_key}.")
        return plaintext

    async def _ensure(self, token_programs: list, imports=None) -> None:
        """Fetch the import closure (async) and register it with the process."""
        sources: dict = dict(imports or {})
        stack = [self.program, *token_programs, *sources]
        seen: set = set()
        while stack:
            pid = stack.pop()
            if pid in seen or pid == "credits.aleo":
                continue
            seen.add(pid)
            if pid not in sources:
                prog = await self._program(pid)
                sources[pid] = str(prog.source)
            stack.extend(program_imports(sources[pid]))
        register_program_sources(self._aleo, sources)

    async def _is_wrapped(self, token_id: str) -> bool:
        """True when the AMM registers *token_id* as a wrapper (chain probe
        of ``from_wrapper_token_id``, cached; transport errors propagate —
        see the sync client)."""
        tid = str(token_id)
        if tid not in self._wrapped_cache:
            prog = await self._program(self.program)
            raw = await prog.mapping("from_wrapper_token_id").get(tid)
            self._wrapped_cache[tid] = normalize_mapping_value(raw) is not None
        return self._wrapped_cache[tid]

    async def _registry_row(self, token_id: str) -> Any:
        for tok in await self.api.get_tokens():
            if tok.address == token_id:
                return tok
        return None

    async def _token_program(self, token_id: str) -> str:
        """Program holding the records that FUND a token — the underlying
        program for wrapped assets, the ARC-20 program itself for plain."""
        tok = await self._registry_row(token_id)
        prog = tok and (getattr(tok, "underlying_program", None)
                        or getattr(tok, "amm_token_program", None)
                        or getattr(tok, "wrapper_program", None))
        if prog:
            return prog
        raise ValueError(
            f"Cannot resolve the record program for {token_id} — pass "
            "token_in_program= (or token_record=) explicitly.")

    async def _amm_token_program(self, token_id: str) -> Optional[str]:
        """Wrapper program the AMM stores balances in (routed calls'
        dynamic-dispatch callee).  Best-effort — None when the registry
        has no row or is unreachable; routing never depends on this."""
        try:
            tok = await self._registry_row(token_id)
        except Exception:
            return None
        return tok and (getattr(tok, "amm_token_program", None)
                        or getattr(tok, "wrapper_program", None))

    def _account(self, account: Any = None) -> Any:
        acct = account if account is not None else self._aleo.default_account
        if acct is None:
            raise ValueError("No signer: pass account= or set aleo.default_account")
        return acct

    # ── Balances ─────────────────────────────────────────────────────────────

    # ── Owned positions ──────────────────────────────────────────────────────

    async def _tick_info(self, pool_key: str, tick: int) -> "Optional[g.Tick]":
        """The on-chain ``Tick`` entry for *tick*, or None if uninitialized."""
        raw = await self._mapping_value("ticks", self.derive_tick_key(pool_key, tick))
        return g.Tick.from_plaintext(raw) if raw is not None else None

    async def _owned_position_state(self, pool_key: str, position: Any,
                                    slot: Any) -> Optional[OwnedPositionState]:
        """Async mirror of ``ShieldSwap._owned_position_state``.

        *slot* is passed in so one slot read serves every position in a pool.
        """
        lower = await self._tick_info(pool_key, int(position.tick_lower))
        upper = await self._tick_info(pool_key, int(position.tick_upper))
        if lower is None or upper is None:
            return None

        liquidity = int(position.liquidity)
        amount0, amount1 = amounts_for_liquidity(
            u256_of(slot.sqrt_price),
            get_sqrt_price_at_tick_x128(int(position.tick_lower)),
            get_sqrt_price_at_tick_x128(int(position.tick_upper)),
            liquidity,
        )
        inside0, inside1 = fee_growth_inside(
            (u256_of(lower.fee_growth_outside0_x_128),
             u256_of(lower.fee_growth_outside1_x_128)), int(lower.tick),
            (u256_of(upper.fee_growth_outside0_x_128),
             u256_of(upper.fee_growth_outside1_x_128)), int(upper.tick),
            int(slot.tick),
            (u256_of(slot.fee_growth_global0_x_128),
             u256_of(slot.fee_growth_global1_x_128)),
        )
        owed0, owed1 = int(position.tokens_owed0), int(position.tokens_owed1)
        return OwnedPositionState(
            liquidity=liquidity,
            amount0=amount0, amount1=amount1,
            collectible0=owed0 + fee_owed(
                inside0, u256_of(position.fee_growth_inside0_last_x_128), liquidity),
            collectible1=owed1 + fee_owed(
                inside1, u256_of(position.fee_growth_inside1_last_x_128), liquidity),
            tokens_owed0=owed0, tokens_owed1=owed1,
        )

    async def get_owned_positions(self, *, pool_key: Optional[str] = None,
                                  account: Any = None) -> list[OwnedPosition]:
        """Every position this account holds, joined with live chain state.

        Async mirror of :meth:`ShieldSwap.get_owned_positions` — same reads, same
        ``state is None`` semantics while a mint finalizes.
        """
        acct = self._account(account)
        records = await self._aleo.record_provider.find(
            acct, program=self.program, unspent=True)
        out: list[OwnedPosition] = []
        slots: dict[str, Any] = {}      # one slot read per pool, not per position
        for rec in records:
            plaintext = record_plaintext(rec)
            if not plaintext:
                continue
            decoded = decode_position_record(plaintext)
            if decoded is None:
                continue
            pool = str(decoded["pool"])
            if pool_key is not None and pool != pool_key:
                continue
            token_id = str(decoded["token_id"])
            raw = await self._mapping_value("positions", token_id)
            state = None
            if raw is not None:
                if pool not in slots:
                    slots[pool] = (await self.get_slot(pool)).raw
                state = await self._owned_position_state(
                    pool, g.Position.from_plaintext(raw), slots[pool])
            out.append(OwnedPosition(
                position_token_id=token_id, pool_key=pool,
                tick_lower=int(decoded["tick_lower"]),
                tick_upper=int(decoded["tick_upper"]),
                token0_id=str(decoded["token0_id"]),
                token1_id=str(decoded["token1_id"]),
                withdrawal=str(decoded["withdrawal"]),
                record=plaintext, state=state))
        return out

    async def plan_rebalance(self, *, pool_key: str, position_token_id: str,
                             tick_lower: int, tick_upper: int,
                             liquidity_target: Optional[int] = None,
                             max_funding0: Optional[int] = None,
                             max_funding1: Optional[int] = None) -> RebalancePlan:
        """Quote a close-and-remint — see :meth:`ShieldSwap.plan_rebalance`.
        Reads only; the boundary-tick and wrapped-ness reads run concurrently."""
        import asyncio
        pool = await self.get_pool(pool_key)
        slot = (await self.get_slot(pool_key)).raw
        raw = await self._mapping_value("positions", position_token_id)
        if raw is None:
            raise ValueError(f"Position does not exist: {position_token_id}")
        position = g.Position.from_plaintext(raw)
        lower, upper, w0, w1 = await asyncio.gather(
            self._tick_info(pool_key, int(position.tick_lower)),
            self._tick_info(pool_key, int(position.tick_upper)),
            self._is_wrapped(str(pool.token0)),
            self._is_wrapped(str(pool.token1)))
        if lower is None or upper is None:
            raise ValueError("The position's boundary ticks are not initialized: "
                             f"{position_token_id}")
        return _plan_rebalance(
            pool_key=pool_key, position_token_id=position_token_id,
            tick_lower=tick_lower, tick_upper=tick_upper,
            slot=slot, position=position, lower_tick=lower, upper_tick=upper,
            wrapped0=w0, wrapped1=w1, liquidity_target=liquidity_target,
            max_funding0=max_funding0, max_funding1=max_funding1)

    async def get_owned_position(self, position_token_id: str, *,
                                 account: Any = None) -> Optional[OwnedPosition]:
        """One owned position by token id, or None — see
        :meth:`ShieldSwap.get_owned_position`."""
        for owned in await self.get_owned_positions(account=account):
            if owned.position_token_id == position_token_id:
                return owned
        return None

    async def get_public_balances(self, programs: list[str],
                                  address: Optional[str] = None) -> dict[str, int]:
        """Public balances per program from each program's on-chain
        ``balances`` mapping — see :meth:`ShieldSwap.get_public_balances`."""
        if address is None:
            acct = self._aleo.default_account
            if acct is None:
                raise ValueError("No address: pass address= or set aleo.default_account")
            address = str(acct.address)
        out: dict[str, int] = {}
        for program in dict.fromkeys(programs):
            try:
                raw = normalize_mapping_value(
                    await self._aleo.network.get_program_mapping_value(program, "balances", address))
                out[program] = parse_unsigned_literal(raw, program, address)
            except (ProgramNotFound, ValueError) as exc:
                _log.warning("get_public_balances: skipping %s — %s", program, exc)
            except AleoNetworkError as exc:
                if exc.status != 404:
                    raise
                _log.warning("get_public_balances: %s is not deployed here — skipped", program)
        return out

    async def get_private_balances(self, programs: list[str],
                                   account: Any = None) -> dict[str, int]:
        """Sum of unspent record amounts per wrapper program (spendable
        privately).  Requires a configured record provider.
        """
        provider = self._aleo.record_provider
        out: dict[str, int] = {}
        for program in programs:
            total = 0
            for rec in await provider.find(account, program=program, unspent=True):
                plaintext = (rec.get("record_plaintext") if isinstance(rec, dict)
                             else getattr(rec, "record_plaintext", None))
                info = parse_token_record_info(plaintext) if plaintext else None
                if info is not None:
                    total += info["amount"]
            out[program] = total
        return out

    async def get_balances(self, address: Optional[str] = None,
                           account: Any = None, *,
                           include_private: bool = True) -> dict[str, dict[str, Any]]:
        """Public + private + total per token id, joined via the API's
        token registry (``include_private=False`` skips the record scan —
        see :meth:`ShieldSwap.get_balances`).  Defaults to the bound
        account's address; returns
        only tokens actually held.

        Private balances can only be scanned for the bound account's view
        key — when *address* names someone else, ``private`` is 0 for every
        token (their records are not scannable) rather than silently mixing
        in the caller's own private holdings.
        """
        acct = account if account is not None else self._aleo.default_account
        addr = address or (str(acct.address) if acct is not None else None)
        if addr is None:
            raise ValueError("No address: pass address= or set aleo.default_account")
        tokens = await self.api.get_tokens()
        # Public balances live in the AMM token program the DEX dispatches
        # through; spendable private records in the record-funding program —
        # the UNDERLYING program for wrapped assets, the ARC-20 itself for plain.
        public_by_program: dict[str, Any] = {
            tok.amm_token_program: tok for tok in tokens if tok.amm_token_program}
        private_by_program: dict[str, Any] = {}
        for tok in tokens:
            prog = tok.underlying_program or tok.amm_token_program
            if prog:
                private_by_program[prog] = tok
        own_address = str(acct.address) if acct is not None else None
        # Record scan first — see ShieldSwap.get_balances.
        private = (await self.get_private_balances(list(private_by_program), account=acct)
                   if include_private and addr == own_address
                   else {p: 0 for p in private_by_program})
        public = await self.get_public_balances(list(public_by_program), address=addr)

        out: dict[str, dict[str, Any]] = {}

        def entry_for(tok: Any) -> dict[str, Any]:
            return out.setdefault(tok.address, {
                "symbol": tok.symbol, "decimals": tok.decimals,
                "public": 0, "private": 0})

        for program, amount in public.items():
            if amount:
                entry_for(public_by_program[program])["public"] += amount
        for program, amount in private.items():
            if amount:
                entry_for(private_by_program[program])["private"] += amount
        for entry in out.values():
            entry["total"] = entry["public"] + entry["private"]
        return out

    # ── Writes ───────────────────────────────────────────────────────────────

    # ── Compliance proofs ────────────────────────────────────────────────────

    async def _freezelist_for_token(self, token_id: str, program: Optional[str] = None) -> Optional[str]:
        """The freeze list a wrapped token's wrapper checks — its underlying
        stablecoin's — or None for a wrapper that ignores the proof (credits).
        *program* is the underlying program when the caller named it."""
        if program is None:
            try:
                program = await self._token_program(token_id)
            except ValueError as exc:
                raise ValueError(
                    f"Cannot resolve the program behind {token_id} to find its freeze "
                    "list — pass compliance_proofs= explicitly.") from exc
        return await async_freezelist_program(self._aleo, program)

    async def _compliance_proofs(self, action: str, slots: list[Slot], compliance_proofs: Any,
                                 wrapper_proofs: Optional[str],
                                 programs: Optional[dict[str, Optional[str]]] = None) -> list[str]:
        """The ``[MerkleProof; 2]`` literal for every planned slot, in order
        (see the sync client)."""
        check_proof_arguments(compliance_proofs, wrapper_proofs)
        if compliance_proofs is not None:
            return supplied_literals(action, compliance_proofs, slots)
        if wrapper_proofs is not None:
            amm_only = [ComplianceProofContext(self.amm_freezelist_program, slot.subject) if slot.kind == "amm" else None
                        for slot in slots]
            return merge_legacy_wrapper_proofs(slots, await self.freezelist.resolve(amm_only), wrapper_proofs)
        known = programs or {}
        lists: dict[str, Optional[str]] = {}
        for slot in slots:
            if slot.kind == "wrapper" and slot.token_id is not None and slot.token_id not in lists:
                lists[slot.token_id] = await self._freezelist_for_token(slot.token_id, known.get(slot.token_id))
        planned = planned_proof_slots(slots, self.amm_freezelist_program, lambda token_id: lists[token_id])
        return await self.freezelist.resolve(planned)

    async def quote(self, *, token_in: str, token_out: str, amount_in: str | Decimal,
                    slippage_bps: int = 50) -> SwapQuote:
        """Quote a direct or multi-hop swap from two symbols and a token amount.

        Returns the API's best route and final slippage floor; pass it to
        ``swap(quote)``. Requires API authentication. Quoting spends no funds
        and needs no record scanning or transaction signature.
        """
        tokens = await self.api.get_tokens()
        source, target, amount = quote_request(tokens, token_in, token_out, amount_in, slippage_bps)
        route = await self.api.get_route(token_in=source.address, token_out=target.address, amount_in=amount)
        return build_quote(source, target, amount, slippage_bps, route, self._aleo.network_name, self.program)

    async def swap(self, quote: Optional[SwapQuote] = None, *, pool_key: Optional[str] = None,
                   token_in_id: Optional[str] = None, amount_in: Optional[int | str | Decimal] = None,
                   slippage_bps: Optional[int] = None, expected_out: Optional[int | str | Decimal] = None,
                   sqrt_price_limit: Optional[int] = None,
                   deadline_offset_blocks: int = 10_000,
                   nonce: Optional[int] = None,
                   token_in_program: Optional[str] = None,
                   token_record: Optional[str] = None,
                   identity: Optional[BlindedIdentity] = None,
                   compliance_proofs: Optional[SwapProofs] = None,
                   wrapper_proofs: Optional[str] = None,
                   imports: Optional[dict[str, str]] = None,
                   account: Any = None) -> AsyncDexCall[SwapHandle]:
        """Request a private swap — phase one of the two-transaction flow.

        Wrapped inputs route via the swap router automatically; fund them
        with UNDERLYING records — the deposit happens in-transaction.

        Resolves the intent against live pool state, derives a single-use
        blinded identity from the signer's view key, selects an unspent token
        record (or takes *token_record* verbatim), and returns a prepared
        call.  The terminal method (``transact``/``delegate``) returns a
        :class:`~aleo_shield_swap.types.SwapHandle` — persist it if the
        process might die before the claim.

        Integer ``amount_in`` and ``expected_out`` values are base units.
        Strings and ``Decimal`` values are token units (``"1.5"`` means 1.5
        tokens), converted using registry decimals. Floats, excess precision,
        non-finite values, and amounts outside u128 are rejected before proving.
        Returned handle amounts remain in base units.
        ``token_record`` bypasses scanning. Provider errors propagate.

        Quote first (``dex.api.get_route``) and pass *expected_out*: without
        it a spot estimate is used, which ignores fees and price impact.
        **This client has no journal**, so it cannot reserve blinding counters:
        the identity comes from an on-chain probe, and the probe is not atomic —
        two concurrent async swaps derive the same counter and the second reverts
        at finalize.  Pass *identity* explicitly (from a sync client's
        ``journal.reserve_counters``) for anything concurrent.  The sync
        :meth:`ShieldSwap.swap` reserves and journals automatically.  The default
        *deadline_offset_blocks* (~8h at ~3s blocks) absorbs delegated-
        proving latency; a tight deadline aborts at finalize when proving
        outlives it.
        """
        if quote is not None:
            if any(v is not None for v in (pool_key, token_in_id, amount_in, expected_out, slippage_bps, sqrt_price_limit)):
                raise ValueError("A quote cannot be combined with trade parameter overrides")
            amount_in, expected_out = quote_amounts(quote, self._aleo.network_name, self.program)
            pool_key, token_in_id, slippage_bps = quote.hops[0].pool_key, quote.token_in_id, quote.slippage_bps
        elif pool_key is None or token_in_id is None or amount_in is None:
            raise ValueError("Pass a quote or pool_key, token_in_id, and amount_in")
        slippage_bps = 50 if slippage_bps is None else slippage_bps
        multi_hop = quote is not None and len(quote.hops) > 1
        acct = self._account(account)
        pool = await self.get_pool(pool_key)
        tokens = []
        if isinstance(amount_in, (str, Decimal)) or isinstance(expected_out, (str, Decimal)):
            tokens = await self.api.get_tokens()
        amount_in = _amount_to_base_units(amount_in, token_in_id, tokens)
        if expected_out is not None:
            token_out_id = str(pool.token1) if token_in_id == str(pool.token0) else str(pool.token0)
            expected_out = _amount_to_base_units(expected_out, token_out_id, tokens)
        slot = await self.get_slot(pool_key)
        resolved = resolve_swap_params(
            pool=pool, slot=slot, token_in_id=token_in_id, amount_in=amount_in,
            slippage_bps=slippage_bps, expected_out=expected_out,
            sqrt_price_limit=sqrt_price_limit)
        hop_literals = []
        if quote is not None:
            pools = [pool] + [await self.get_pool(h.pool_key) for h in quote.hops[1:]]
            slots = [slot] + [await self.get_slot(h.pool_key) for h in quote.hops[1:]]
            resolved, hop_literals = resolve_quote(quote, pools, slots, amount_in, expected_out)

        record = token_record
        program = token_in_program
        candidates: list[str] = []
        if record is None:
            program = program or await self._token_program(token_in_id)
            # Keep every eligible candidate from one scan so a reservation
            # conflict at submission can move to the next record.
            candidates = await self._select_token_records(
                program=program, min_amount=amount_in,
                token_id=token_in_id, account=acct)
            record = candidates[0]
            token_programs = [program]
        else:
            token_programs = [token_in_program] if token_in_program else []

        deadline = int(await self._aleo.network.get_latest_height()) + deadline_offset_blocks
        swap_nonce = nonce if nonce is not None else generate_swap_nonce()
        if identity is None:
            identity = await self._next_blinded_identity(acct)

        route = swap_route(await self._is_wrapped(token_in_id), multi_hop)
        if route.program != self.program:
            token_programs.append(route.program)
            wrapper = await self._amm_token_program(token_in_id)
            if wrapper:
                token_programs.append(wrapper)
        if multi_hop:
            for hop in quote.hops:
                token_programs.append(await self._amm_token_program(hop.token_in))
                token_programs.append(await self._amm_token_program(hop.token_out))
        await self._ensure([p for p in token_programs if p], imports)
        proofs = await self._compliance_proofs(
            "swap", swap_slots(route.program != self.program, str(acct.address), token_in_id),
            compliance_proofs, wrapper_proofs, {token_in_id: program})
        sender_proof = proofs[0] if proofs else default_merkle_proofs()

        if route.program != self.program and resolved.amount_out_min <= 0:
            raise ValueError(
                "routed swaps require amount_out_min > 0 — pass "
                "expected_out or a slippage below 100%")
        prog = await self._program(route.program)

        def assemble(spend: str) -> Any:
            """Bind the call for one input record; every other term is fixed above."""
            inputs = [
                spend, identity.blinding_factor, identity.blinded_address,
                pool_key, resolved.zero_for_one,
                f"{amount_in}u128", f"{resolved.amount_out_min}u128",
                int_to_u256_plaintext(resolved.sqrt_price_limit),
                f"{swap_nonce}u64", f"{deadline}u32",
                pool.token0, pool.token1,
            ]
            if route.program != self.program:
                inputs = [spend, sender_proof, *inputs[1:]]
            if multi_hop:
                inputs = multi_hop_inputs(quote, hop_literals, spend, identity, amount_in,
                                          resolved.amount_out_min, swap_nonce, deadline,
                                          sender_proof, route.program != self.program)
            return getattr(prog.functions, route.function)(*inputs)

        bound = assemble(record)
        reselect = None
        if token_record is None and self.inventory.enabled:
            reselect = RecordReselection(candidates, assemble, self._swap_context_guard(acct, account is None))

        def build_result(tx_id: str, outputs: list[Any]) -> SwapHandle:
            """Assemble the swap handle, carrying the blinding secret forward.

            The first public ``field`` output is the swap id; it stays ``None`` if
            the transition published none, which leaves the handle unclaimable
            until the id is recovered from the transaction.
            """
            swap_id = next((o for o in outputs
                            if isinstance(o, str) and o.endswith("field")), None)
            return SwapHandle(
                swap_id=swap_id,
                blinding_factor=identity.blinding_factor,
                blinded_address=identity.blinded_address,
                token_in_id=token_in_id, token_out_id=resolved.token_out_id,
                pool_key=pool_key, amount_in=amount_in,
                transaction_id=tx_id, program=self.program,
                pool_keys=tuple(h.pool_key for h in quote.hops) if multi_hop else ())

        return AsyncDexCall(self._aleo, bound, build_result, reselect=reselect)

    def _swap_context_guard(self, acct: Any, from_default: bool) -> Callable[[], None]:
        """A check that the signer, network, and journal are the ones a swap was prepared for."""
        return swap_context_guard(self._aleo, acct, (lambda: self._account(None)) if from_default else None)

    # ── Record inventory ─────────────────────────────────────────────────────

    async def _underlying_program(self, token: str) -> str:
        return underlying_program_for(await self.api.get_tokens(), token)

    async def plan_inventory(self, token: str, target: InventoryTarget, *,
                             max_transactions: Optional[int] = None, account: Any = None) -> InventoryPlan:
        """Plan joins and splits for *token*'s underlying records (awaited). See :meth:`ShieldSwap.plan_inventory`."""
        program = await self._underlying_program(token)
        return await self.inventory.plan_record_inventory(
            RecordAsset(program), target, max_transactions=max_transactions, account=account)

    async def rebalance_inventory(self, token: str, target: InventoryTarget, *,
                                  max_transactions: Optional[int] = None, timeout: float = 120.0,
                                  poll_interval: float = 1.0, max_fee_microcredits: Optional[int] = None,
                                  cancel: Any = None, account: Any = None) -> InventoryResult:
        """Reshape *token*'s underlying records to *target* (awaited). See :meth:`ShieldSwap.rebalance_inventory`."""
        plan = await self.plan_inventory(token, target, max_transactions=max_transactions, account=account)
        return await self.inventory.rebalance_record_inventory(
            plan, timeout=timeout, poll_interval=poll_interval,
            max_fee_microcredits=max_fee_microcredits, cancel=cancel, account=account)

    async def claim_swap_output(self, handle: SwapHandle, *,
                                timeout: float = 0.0,
                                compliance_proofs: Optional[ClaimProofs] = None,
                                wrapper_proofs: Optional[str] = None,
                                imports: Optional[dict[str, str]] = None,
                                account: Any = None) -> AsyncDexCall[ClaimResult]:
        """Claim a private swap's output — phase two of the lifecycle.

        Reads the chain-computed result from ``swap_outputs`` (never an
        off-chain service — these amounts gate money movement), proves
        ownership of the blinded identity, and claims.  A wrapped output or
        refund routes automatically through the router, which unwraps to
        the signer in the same transaction — even for swaps that started
        as direct core calls.  The output and any refund arrive as private
        records owned by the signer (output first, refund second); the
        mapping entry is consumed.

        After swap confirmation, ``timeout`` sets the maximum wait in seconds
        (default 0). Missing output is polled every 2s without submitting. Timeout raises
        ``SwapOutputNotFinalizedError``; other errors propagate.
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and nonnegative")
        acct = self._account(account)
        if not handle.swap_id:
            raise ValueError("handle.swap_id is not set — recover it from the "
                             "confirmed request transaction before claiming.")
        if not handle.blinding_factor or not handle.blinded_address:
            raise ValueError("Claims need handle.blinding_factor and "
                             "handle.blinded_address (set by swap()).")
        deadline = time.monotonic() + timeout
        while True:
            try:
                out = await self.get_swap_output(handle.swap_id)
                break
            except SwapOutputNotFinalizedError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                await asyncio.sleep(min(2.0, remaining))
        w_out = await self._is_wrapped(str(out.token_out))
        w_in = await self._is_wrapped(str(out.token_in))
        no_refund = int(out.amount_remaining) == 0
        route = claim_route(w_out, w_in, no_refund=no_refund)
        token_programs = [route.program] if route.program != self.program else []
        # The core calls each leg's token program dynamically
        # (`transfer_public_to_private` on payout/refund), so both must be in
        # the process before authorization — wrapper for wrapped legs, the
        # ARC-20 itself for plain ones — exactly as the sync client does.
        for token_id, wrapped in ((out.token_in, w_in), (out.token_out, w_out)):
            if wrapped:
                wrapper = await self._amm_token_program(str(token_id))
                if wrapper:
                    token_programs.append(wrapper)
            else:
                token_programs.append(await self._token_program(str(token_id)))
        await self._ensure(token_programs, imports)
        inputs = [handle.blinding_factor, handle.blinded_address, handle.swap_id,
                  out.token_in, out.token_out, f"{out.amount_out}u128"]
        if not no_refund:
            inputs.append(f"{out.amount_remaining}u128")
        # The AMM signer bundle, then one receiver bundle per wrapped leg being
        # paid — the output first, then the refund (none on a no-refund claim).
        inputs.extend(await self._compliance_proofs(
            "claim_swap_output",
            claim_slots(w_out, w_in, no_refund, str(acct.address), str(out.token_in), str(out.token_out)),
            compliance_proofs, wrapper_proofs))
        prog = await self._program(route.program)
        bound = getattr(prog.functions, route.function)(*inputs)
        return AsyncDexCall(self._aleo, bound,
                            lambda tx_id, _o: ClaimResult(tx_id, out.amount_out,
                                                          out.amount_remaining))
