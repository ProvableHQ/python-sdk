"""AsyncShieldSwap — the shield_swap client over an ``AsyncAleo`` facade.

Reads, balances, and the two-transaction private-swap flow of
:class:`~aleo_shield_swap.client.ShieldSwap`, with every I/O method
``async``.  Liquidity methods are sync-client-only for now.  Pure logic is
shared from ``_core``/``derivations`` — only the I/O differs.
"""
from __future__ import annotations

from typing import Any, Callable, Generic, Optional, TypeVar

from . import _generated as g
from ._calls import extract_tx_id, root_outputs
from ._core import (
    default_merkle_proofs,
    find_position_plaintext,
    normalize_mapping_value,
    generate_swap_nonce,
    parse_token_record_info,
    pick_covering_record,
    program_imports,
    register_program_sources,
    resolve_swap_params,
)
from ._routing import claim_route, swap_route
from .api import AsyncApiClient, DEFAULT_API_URL
from .tick_math import int_to_u256_plaintext
from .derivations import (
    BlindedIdentity,
    derive_blinded_address,
    derive_blinding_factor,
    derive_pool_key as _derive_pool_key,
    derive_tick_key as _derive_tick_key,
)
from .errors import (
    InsufficientRecordsError,
    PoolNotFoundError,
    PoolNotInitializedError,
    SwapOutputNotFinalizedError,
)
from .types import ClaimResult, SlotView, SwapHandle

R = TypeVar("R")


class AsyncDexCall(Generic[R]):
    """Async mirror of :class:`~aleo_shield_swap._calls.DexCall`."""

    def __init__(self, aleo: Any, bound: Any,
                 build_result: Callable[[str, list[Any]], R]) -> None:
        self._aleo = aleo
        self._bound = bound
        self._build = build_result

    def simulate(self, account: Any = None) -> Any:
        """Local authorization — no proof, no send; inspect before spending.

        Synchronous even on the async client: ``AsyncBoundCall.simulate`` only
        authorizes locally, so there is nothing to await.
        """
        return self._bound.simulate(account)

    async def transact(self, account: Any = None, **fee_kwargs: Any) -> R:
        """Prove locally, broadcast, and build the typed result.

        Root-transition outputs are harvested from the built transaction
        before broadcast, so the result is complete without waiting for
        confirmation.
        """
        tx = await self._bound.build_transaction(account, **fee_kwargs)
        outputs = root_outputs(tx.decoded(), self._bound.program_id,
                               self._bound.function_name)
        await self._aleo.network.submit_transaction(tx.raw)
        return self._build(tx.id, outputs)

    async def delegate(self, account: Any = None, **fee_kwargs: Any) -> R:
        """Delegate proving to the DPS (fee master pays by default) and build
        the typed result from the root transition.

        Outputs are harvested from the DPS payload itself when it carries
        the full transaction, so ``wait=False`` returns as soon as the
        broadcast is accepted — callers that read chain state later (e.g.
        ``collect_all``) don't need to block on confirmation here.
        """
        payload = await self._bound.delegate(account, **fee_kwargs)
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
                 api_url: str = DEFAULT_API_URL) -> None:
        self._aleo = aleo
        self.program = program
        self.api = AsyncApiClient(api_url)
        # allow_token relationships are immutable — cache probes for the
        # client's lifetime.
        self._wrapped_cache: dict[str, bool] = {}

    def __repr__(self) -> str:
        return f"AsyncShieldSwap(program={self.program!r}, api={self.api.base_url!r})"

    # ── Mapping plumbing ─────────────────────────────────────────────────────

    async def _mapping_value(self, mapping: str, key: str) -> Optional[str]:
        prog = await self._aleo.programs.get(self.program)
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

    async def _next_blinded_identity(self, account: Any) -> BlindedIdentity:
        network = self._aleo.network_name
        scalar = str(account.view_key.to_scalar())
        signer = str(account.address)
        prog = await self._aleo.programs.get(self.program)
        mapping = prog.mapping("used_blinded_addresses")
        for counter in range(64):
            bf = derive_blinding_factor(scalar, counter, self.program, network=network)
            ba = derive_blinded_address(bf, signer, self.program, network=network)
            if (await mapping.get(ba)) in (None, "", "null", "false"):
                return BlindedIdentity(counter, bf, ba)
        raise ValueError(f"No unused blinded address in 64 counters for {self.program}")

    async def _select_token_record(self, *, program: str, min_amount: int,
                                   token_id: Optional[str], account: Any) -> str:
        provider = self._aleo.record_provider
        if provider is None:
            raise InsufficientRecordsError(
                "No record provider configured — pass token_record= explicitly.")
        records = await provider.find(account, program=program, unspent=True)
        chosen = pick_covering_record(records, min_amount=min_amount, token_id=token_id)
        if chosen is None:
            raise InsufficientRecordsError(
                f"No unspent {program} record covers {min_amount} "
                f"(token_id={token_id or 'any'}) — privatize funds or pass token_record=.")
        return chosen

    async def _select_position_record(self, pool_key: str, account: Any) -> str:
        records = await self._aleo.record_provider.find(
            account, program=self.program, unspent=True)
        plaintext = find_position_plaintext(records, pool_key)
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
                prog = await self._aleo.programs.get(pid)
                sources[pid] = str(prog.source)
            stack.extend(program_imports(sources[pid]))
        register_program_sources(self._aleo, sources)

    async def _is_wrapped(self, token_id: str) -> bool:
        """True when the AMM registers *token_id* as a wrapper (chain probe
        of ``from_wrapper_token_id``, cached; transport errors propagate —
        see the sync client)."""
        tid = str(token_id)
        if tid not in self._wrapped_cache:
            prog = await self._aleo.programs.get(self.program)
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
                           account: Any = None) -> dict[str, dict[str, Any]]:
        """Public + private + total per token id, joined via the API's
        token registry.  Defaults to the bound account's address; returns
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
        # Spendable private records live in the record-funding program: the
        # UNDERLYING program for wrapped assets, the ARC-20 itself for plain.
        by_program = {t.underlying_program or t.amm_token_program: t
                      for t in tokens
                      if t.underlying_program or t.amm_token_program}
        private = await self.get_private_balances(list(by_program), account=acct)
        out: dict[str, dict[str, Any]] = {}
        for bal in await self.api.get_public_balances(addr):
            entry = out.setdefault(bal.token_id, {
                "symbol": bal.symbol, "decimals": bal.decimals,
                "public": 0, "private": 0})
            entry["public"] += int(bal.balance)
        for program, amount in private.items():
            tok = by_program[program]
            if amount == 0 and tok.address not in out:
                continue
            entry = out.setdefault(tok.address, {
                "symbol": tok.symbol, "decimals": tok.decimals,
                "public": 0, "private": 0})
            entry["private"] += amount
        for entry in out.values():
            entry["total"] = entry["public"] + entry["private"]
        return out

    # ── Writes ───────────────────────────────────────────────────────────────

    async def swap(self, *, pool_key: str, token_in_id: str, amount_in: int,
                   slippage_bps: int = 50, expected_out: Optional[int] = None,
                   sqrt_price_limit: Optional[int] = None,
                   deadline_offset_blocks: int = 100,
                   nonce: Optional[int] = None,
                   token_in_program: Optional[str] = None,
                   token_record: Optional[str] = None,
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

        Quote first (``dex.api.get_route``) and pass *expected_out*: without
        it a spot estimate is used, which ignores fees and price impact.
        Pass *identity* (from journal-reserved counters) to skip the
        on-chain probe — required for concurrent swaps.  The default
        *deadline_offset_blocks* (~8h at ~3s blocks) absorbs delegated-
        proving latency; a tight deadline aborts at finalize when proving
        outlives it.
        """
        acct = self._account(account)
        pool = await self.get_pool(pool_key)
        slot = await self.get_slot(pool_key)
        resolved = resolve_swap_params(
            pool=pool, slot=slot, token_in_id=token_in_id, amount_in=amount_in,
            slippage_bps=slippage_bps, expected_out=expected_out,
            sqrt_price_limit=sqrt_price_limit)
        deadline = int(await self._aleo.network.get_latest_height()) + deadline_offset_blocks
        swap_nonce = nonce if nonce is not None else generate_swap_nonce()
        identity = await self._next_blinded_identity(acct)

        record = token_record
        if record is None:
            program = token_in_program or await self._token_program(token_in_id)
            record = await self._select_token_record(
                program=program, min_amount=amount_in,
                token_id=token_in_id, account=acct)
            token_programs = [program]
        else:
            token_programs = [token_in_program] if token_in_program else []

        route = swap_route(await self._is_wrapped(token_in_id))
        if route.program != self.program:
            token_programs.append(route.program)
            wrapper = await self._amm_token_program(token_in_id)
            if wrapper:
                token_programs.append(wrapper)
        await self._ensure(token_programs, imports)

        inputs = [
            record, identity.blinding_factor, identity.blinded_address,
            pool_key, resolved.zero_for_one,
            f"{amount_in}u128", f"{resolved.amount_out_min}u128",
            int_to_u256_plaintext(resolved.sqrt_price_limit),
            f"{swap_nonce}u64", f"{deadline}u32",
            pool.token0, pool.token1,
        ]
        if route.program != self.program:
            if resolved.amount_out_min <= 0:
                raise ValueError(
                    "routed swaps require amount_out_min > 0 — pass "
                    "expected_out or a slippage below 100%")
            inputs = [record, wrapper_proofs or default_merkle_proofs(),
                      *inputs[1:]]
        prog = await self._aleo.programs.get(route.program)
        bound = getattr(prog.functions, route.function)(*inputs)

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
                transaction_id=tx_id, program=self.program)

        return AsyncDexCall(self._aleo, bound, build_result)

    async def claim_swap_output(self, handle: SwapHandle, *,
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

        Raises :class:`SwapOutputNotFinalizedError` **at prepare time** when
        the output is not readable yet (retry after a few blocks) or was
        already claimed.
        """
        self._account(account)
        if not handle.swap_id:
            raise ValueError("handle.swap_id is not set — recover it from the "
                             "confirmed request transaction before claiming.")
        if not handle.blinding_factor or not handle.blinded_address:
            raise ValueError("Claims need handle.blinding_factor and "
                             "handle.blinded_address (set by swap()).")
        out = await self.get_swap_output(handle.swap_id)
        w_out = await self._is_wrapped(str(out.token_out))
        w_in = await self._is_wrapped(str(out.token_in))
        route = claim_route(w_out, w_in)
        token_programs = [route.program] if route.program != self.program else []
        for token_id, wrapped in ((out.token_in, w_in), (out.token_out, w_out)):
            if wrapped:
                wrapper = await self._amm_token_program(str(token_id))
                if wrapper:
                    token_programs.append(wrapper)
        await self._ensure(token_programs, imports)
        inputs = [handle.blinding_factor, handle.blinded_address, handle.swap_id,
                  out.token_in, out.token_out,
                  f"{out.amount_out}u128", f"{out.amount_remaining}u128",
                  default_merkle_proofs()]
        wp = wrapper_proofs or default_merkle_proofs()
        if w_out and w_in:
            inputs += [wp, wp]            # wp for the output, then the refund
        elif w_out or w_in:
            inputs += [wp]
        prog = await self._aleo.programs.get(route.program)
        bound = getattr(prog.functions, route.function)(*inputs)
        return AsyncDexCall(self._aleo, bound,
                            lambda tx_id, _o: ClaimResult(tx_id, out.amount_out,
                                                          out.amount_remaining))
