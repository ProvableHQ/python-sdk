"""ShieldSwap — the synchronous shield_swap client, bound to an ``Aleo`` facade.

Construction wires nothing new: signer, record provider, proving config, and
network all come from the bound client.  On-chain reads and writes live
directly on this object; the off-chain DEX API is namespaced under ``.api``
so a call site always shows whether a value came from the chain or the
service.
"""
from __future__ import annotations

from typing import Any, Optional

from aleo.codegen.runtime import parse_plaintext

from . import _generated as g
from ._core import (
    ensure_programs,
    find_position_plaintext,
    record_plaintext,
    generate_field_nonce,
    generate_swap_nonce,
    get_deadline,
    normalize_mapping_value,
    parse_token_record_info,
    resolve_swap_params,
    select_token_record,
)
from .api import ApiClient, DEFAULT_API_URL
from .derivations import (
    BlindedIdentity,
    blinded_identity_at,
    derive_pool_key as _derive_pool_key,
    derive_tick_key as _derive_tick_key,
    next_blinded_identity,
)
from .errors import (
    InsufficientRecordsError,
    InvalidFeeTierError,
    PoolNotFoundError,
    PoolNotInitializedError,
    ShieldSwapError,
    SwapOutputNotFinalizedError,
)
from .journal import Journal
from .lifecycle import run_onboard
from .profile import Profile
from .types import (
    ClaimResult,
    CollectReport,
    MintResult,
    OnboardReport,
    PositionView,
    SessionStatus,
    SlotView,
    SwapBatchReport,
    SwapHandle,
    TxResult,
)
from ._calls import DexCall
from .tick_hints import pick_insert_hint
from .tick_math import (
    MAX_TICK,
    MIN_TICK,
    get_sqrt_price_at_tick,
    round_tick_to_spacing,
)


class ShieldSwap:
    """Typed client for the shield_swap AMM, over an existing facade client.

    ::

        aleo = Aleo(Aleo.HTTPProvider(ENDPOINT))
        aleo.default_account = account
        dex = ShieldSwap(aleo)
    """

    def __init__(self, aleo: Any, *, program: str = g.PROGRAM_ID,
                 api_url: str = DEFAULT_API_URL) -> None:
        self._aleo = aleo
        self.program = program
        self.api = ApiClient(api_url)
        self.profile: Any = None          # set by from_profile()
        self.journal: Any = None          # set by from_profile()

    def __repr__(self) -> str:
        return f"ShieldSwap(program={self.program!r}, api={self.api.base_url!r})"

    @classmethod
    def from_profile(cls, home: Any = None) -> "ShieldSwap":
        """The client for the local participant profile (created on first use).

        Wires endpoint, network, signer, and (when present) delegated-proving
        credentials from ``$SHIELD_SWAP_HOME``/``~/.shield-swap``.  Run
        ``onboard()`` next on a fresh profile.
        """
        from aleo import Aleo, HTTPProvider

        profile = Profile.load_or_create(home)
        creds = profile.credentials
        provider = HTTPProvider(profile.endpoint, network=profile.network,
                                api_key=creds.get("dps_api_key"),
                                consumer_id=creds.get("dps_consumer_id"))
        aleo = Aleo(provider)
        aleo.default_account = aleo.account.from_private_key(profile.private_key)
        try:
            # Hosted-scanner registration (keyless); scanning itself needs the
            # DPS api key, which the credentials stage provisions.
            aleo.records.register(aleo.default_account)
        except Exception:
            pass  # offline or scanner down — record verbs will surface it
        dex = cls(aleo)
        dex.profile = profile
        dex.journal = Journal(profile.journal_path)
        if creds.get("jwt"):
            dex.api.set_token(creds["jwt"])       # session tier (24h)
        elif creds.get("dex_api_token"):
            dex.api.set_token(creds["dex_api_token"])  # durable data tier
        return dex

    def _refresh_credentials(self) -> None:
        """Push freshly provisioned profile credentials onto the live facade.

        ``from_profile()`` builds the provider before the credentials stage
        has run, so a fresh ``onboard()`` re-applies them here instead of
        requiring a rebuilt client.
        """
        creds = self.profile.credentials if self.profile else {}
        key = creds.get("dps_api_key")
        cid = creds.get("dps_consumer_id")
        if not key:
            return
        provider = getattr(self._aleo, "provider", None)
        if provider is not None:          # future lazy builds (e.g. scanner)
            provider._api_key = key
            if cid:
                provider._consumer_id = cid
        nc = getattr(self._aleo, "network_client", None)
        if nc is not None:                # delegated proving
            nc.api_key = key
            if cid:
                nc.consumer_id = cid
        records = getattr(self._aleo, "records", None)
        if records is not None:           # already-built scanner
            try:
                records.scanner.set_api_key(key)
                if cid:
                    records.scanner.consumer_id = cid
            except Exception:
                pass                      # scanner unreachable — built lazily later
            # Re-register now that the key exists: a keyless registration at
            # from_profile() time does not stick for fresh accounts.
            account = getattr(self._aleo, "default_account", None)
            if account is not None:
                records.register(account)

    def onboard(self, invite_code: Optional[str] = None) -> OnboardReport:
        """Register this profile end to end — safe to re-run any time.

        Runs only the registration stages not already satisfied (see
        ``lifecycle.REGISTRATION_STAGES``); a registered, funded account is
        a no-op.  The one thing it may need from you: *invite_code*, on the
        first run.  Requires a profile-bound client (``from_profile()``).
        """
        if self.profile is None:
            raise ValueError("onboard() needs a profile-bound client — "
                             "construct with ShieldSwap.from_profile().")
        return run_onboard(self, self.profile, invite_code)

    # ── Mapping plumbing ─────────────────────────────────────────────────────

    def _mapping_value(self, mapping: str, key: str) -> Optional[str]:
        raw = self._aleo.programs.get(self.program).mapping(mapping).get(key)
        return normalize_mapping_value(raw)

    # ── Chain reads ──────────────────────────────────────────────────────────

    def get_pool(self, pool_key: str) -> g.PoolState:
        """Static pool configuration (token pair, fee, decimal scales)."""
        raw = self._mapping_value("pools", pool_key)
        if raw is None:
            raise PoolNotFoundError(pool_key)
        return g.PoolState.from_plaintext(raw)

    def get_slot(self, pool_key: str) -> SlotView:
        """Live trading state (sqrt price, tick, in-range liquidity).

        Raises :class:`PoolNotFoundError` when the pool does not exist, or
        :class:`PoolNotInitializedError` when it exists but has no slot yet.
        """
        raw = self._mapping_value("slots", pool_key)
        if raw is None:
            if self._mapping_value("pools", pool_key) is not None:
                raise PoolNotInitializedError(pool_key)
            raise PoolNotFoundError(pool_key)
        return SlotView(g.Slot.from_plaintext(raw))

    def get_swap_output(self, swap: "SwapHandle | str") -> g.SwapOutput:
        """Chain-computed output of a finalized swap request.

        Accepts the :class:`SwapHandle` from ``swap()`` or a bare swap id.
        Raises :class:`SwapOutputNotFinalizedError` when the entry is absent —
        not finalized yet (retry after a few blocks) or already claimed.
        """
        swap_id = swap.swap_id if isinstance(swap, SwapHandle) else swap
        if not swap_id:
            raise ValueError("SwapHandle has no swap_id yet — wait for the "
                             "request transaction and recover it first.")
        raw = self._mapping_value("swap_outputs", swap_id)
        if raw is None:
            raise SwapOutputNotFinalizedError(swap_id)
        return g.SwapOutput.from_plaintext(raw)

    def is_pool_initialized(self, pool_key: str) -> bool:
        raw = self._mapping_value("initialized_pools", pool_key)
        return raw is not None and "true" in raw

    # ── Discovery ────────────────────────────────────────────────────────────

    def get_positions(self, account: Any = None) -> list[PositionView]:
        """Every open position — journaled ones plus a record scan.

        The scan catches positions the journal never saw (account used from
        another machine, journal lost); it needs a registered record
        provider and is skipped silently without one.
        """
        views: dict[Optional[str], PositionView] = {}
        if self.journal is not None:
            for p in self.journal.open_positions():
                views[p["position_token_id"]] = PositionView(
                    p["position_token_id"], p["pool_key"], "journal")
        acct = self._account(account)
        provider = getattr(self._aleo, "record_provider", None)
        if provider is not None:
            try:
                records = list(provider.find(acct, program=self.program,
                                             unspent=True))
            except Exception:
                records = []          # scanner unavailable — journal only
            for rec in records:
                plaintext = record_plaintext(rec)
                if not plaintext:
                    continue
                try:
                    decoded = parse_plaintext(plaintext)
                except (ValueError, TypeError):
                    continue
                if not (isinstance(decoded, dict) and "pool" in decoded):
                    continue
                pid = decoded.get("token_id")   # PositionNFT's id field
                pid = str(pid) if pid is not None else None
                if pid not in views:
                    views[pid] = PositionView(pid, str(decoded["pool"]), "scanned")
        return list(views.values())

    def status(self) -> SessionStatus:
        """One re-orientation call: identity, access, holdings, pending work.

        Run this first in any session — it answers "is this account already
        registered, what do I hold, what is in flight" from the profile,
        journal, chain, and API without changing anything.
        """
        authenticated = getattr(self.api, "_token", None) is not None
        has_access: Optional[bool] = None
        if authenticated:
            try:
                has_access = bool(self.api.access_status().has_access)
            except ShieldSwapError:
                has_access = None
        address = (self.profile.address if self.profile
                   else str(self._account().address))
        try:
            balances = self.get_balances()
        except Exception:
            # Private scan unavailable (e.g. credentials stage not run yet) —
            # degrade to public-only rather than reporting nothing.
            try:
                balances = {b.token_id: {"symbol": b.symbol,
                                         "decimals": b.decimals,
                                         "public": int(b.balance), "private": 0,
                                         "total": int(b.balance)}
                            for b in self.api.get_public_balances(address)}
            except Exception:
                balances = {}
        pending = self.journal.pending_claims() if self.journal else []
        return SessionStatus(
            address=address,
            network=getattr(self._aleo, "network_name", "testnet"),
            authenticated=authenticated,
            has_access=has_access,
            balances=balances,
            pending_claim_ids=[h.swap_id for h in pending if h.swap_id],
            open_positions=self.get_positions(),
            counter_cursor=(self.journal.counter_cursor() if self.journal else 0),
        )

    # ── Pure derivations (no network) ────────────────────────────────────────

    def derive_pool_key(self, token0: str, token1: str, fee: int) -> str:
        return _derive_pool_key(token0, token1, fee,
                                network=self._aleo.network_name)

    def derive_tick_key(self, pool_key: str, tick: int) -> str:
        return _derive_tick_key(pool_key, tick, network=self._aleo.network_name)

    # ── Balances ─────────────────────────────────────────────────────────────

    def get_private_balances(self, programs: list[str],
                             account: Any = None) -> dict[str, int]:
        """Sum of unspent record amounts per wrapper program (spendable
        privately).  Requires a configured record provider."""
        provider = self._aleo.record_provider
        out: dict[str, int] = {}
        for program in programs:
            total = 0
            for rec in provider.find(account, program=program, unspent=True):
                plaintext = record_plaintext(rec)
                info = parse_token_record_info(plaintext) if plaintext else None
                if info is not None:
                    total += info["amount"]
            out[program] = total
        return out

    def get_balances(self, address: Optional[str] = None,
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

        tokens = self.api.get_tokens()
        by_program = {t.wrapper_program: t for t in tokens if t.wrapper_program}
        own_address = str(acct.address) if acct is not None else None
        private = (self.get_private_balances(list(by_program), account=acct)
                   if addr == own_address else {p: 0 for p in by_program})

        out: dict[str, dict[str, Any]] = {}
        for bal in self.api.get_public_balances(addr):
            entry = out.setdefault(bal.token_id, {
                "symbol": bal.symbol, "decimals": bal.decimals,
                "public": 0, "private": 0,
            })
            entry["public"] += int(bal.balance)
        for program, amount in private.items():
            tok = by_program[program]
            if amount == 0 and tok.address not in out:
                continue
            entry = out.setdefault(tok.address, {
                "symbol": tok.symbol, "decimals": tok.decimals,
                "public": 0, "private": 0,
            })
            entry["private"] += amount
        for entry in out.values():
            entry["total"] = entry["public"] + entry["private"]
        return out

    # ── Writes ───────────────────────────────────────────────────────────────

    def _account(self, account: Any = None) -> Any:
        acct = account if account is not None else self._aleo.default_account
        if acct is None:
            raise ValueError("No signer: pass account= or set aleo.default_account")
        return acct

    def _ensure(self, token_programs: list[str],
                imports: Optional[dict[str, str]]) -> None:
        """Register the DEX program, its static imports, the involved token
        programs, and any *imports* overrides with the snarkVM process."""
        pids = [self.program, *token_programs, *(imports or {})]
        ensure_programs(self._aleo, pids, imports)

    def swap(
        self,
        *,
        pool_key: str,
        token_in_id: str,
        amount_in: int,
        slippage_bps: int = 50,
        expected_out: Optional[int] = None,
        sqrt_price_limit: Optional[int] = None,
        deadline_offset_blocks: int = 10_000,
        nonce: Optional[int] = None,
        identity: Optional[BlindedIdentity] = None,
        token_in_program: Optional[str] = None,
        token_record: Optional[str] = None,
        imports: Optional[dict[str, str]] = None,
        account: Any = None,
    ) -> DexCall[SwapHandle]:
        """Request a private swap — phase one of the two-transaction flow.

        Resolves the intent against live pool state, derives a single-use
        blinded identity from the signer's view key, selects an unspent token
        record (or takes *token_record* verbatim), and returns a prepared
        call.  The terminal verb (``transact``/``delegate``) returns a
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
        pool = self.get_pool(pool_key)
        slot = self.get_slot(pool_key)
        resolved = resolve_swap_params(
            pool=pool, slot=slot, token_in_id=token_in_id, amount_in=amount_in,
            slippage_bps=slippage_bps, expected_out=expected_out,
            sqrt_price_limit=sqrt_price_limit,
        )
        deadline = get_deadline(self._aleo, deadline_offset_blocks)
        swap_nonce = nonce if nonce is not None else generate_swap_nonce()
        identity = identity or next_blinded_identity(self._aleo, acct, self.program)

        record = token_record
        if record is None:
            program = token_in_program or self._token_program(token_in_id)
            record = select_token_record(
                self._aleo, program=program, min_amount=amount_in,
                token_id=token_in_id, account=acct,
            )
            token_programs = [program]
        else:
            token_programs = [token_in_program] if token_in_program else []
        # Dynamic dispatch: the prover cannot discover token callees
        # statically — register the DEX program and the involved token
        # programs with the process before authorization.
        self._ensure(token_programs, imports)

        # Input order per the contract's swap entrypoint (mirrors the TS SDK):
        # record, blinding slots, then pool/direction/amounts/bounds/timing/tokens.
        inputs = [
            record,
            identity.blinding_factor,
            identity.blinded_address,
            pool_key,
            resolved.zero_for_one,
            f"{amount_in}u128",
            f"{resolved.amount_out_min}u128",
            f"{resolved.sqrt_price_limit}u128",
            f"{swap_nonce}u64",
            f"{deadline}u32",
            pool.token0,
            pool.token1,
        ]
        bound = self._aleo.programs.get(self.program).functions.swap(*inputs)

        def build_result(tx_id: str, outputs: list[Any]) -> SwapHandle:
            swap_id = next(
                (o for o in outputs if isinstance(o, str) and o.endswith("field")),
                None,
            )
            return SwapHandle(
                swap_id=swap_id,
                blinding_factor=identity.blinding_factor,
                blinded_address=identity.blinded_address,
                token_in_id=token_in_id,
                token_out_id=resolved.token_out_id,
                pool_key=pool_key,
                amount_in=amount_in,
                transaction_id=tx_id,
                program=self.program,
            )

        return DexCall(self._aleo, bound, build_result)

    def claim_swap_output(
        self,
        handle: SwapHandle,
        *,
        imports: Optional[dict[str, str]] = None,
        account: Any = None,
    ) -> DexCall[ClaimResult]:
        """Claim a private swap's output — phase two of the lifecycle.

        Reads the chain-computed result from ``swap_outputs`` (never an
        off-chain service — these amounts gate money movement), proves
        ownership of the blinded identity, and prepares ``claim_swap_output``.
        The output and any refund arrive as private records owned by the
        signer; the mapping entry is consumed.

        Raises :class:`SwapOutputNotFinalizedError` **at prepare time** when
        the output is not readable yet (retry after a few blocks) or was
        already claimed.
        """
        self._account(account)
        if not handle.swap_id:
            raise ValueError(
                "handle.swap_id is not set — recover it from the confirmed "
                "request transaction (first public output of the swap "
                "transition) before claiming."
            )
        if not handle.blinding_factor or not handle.blinded_address:
            raise ValueError(
                "Claims need handle.blinding_factor and handle.blinded_address "
                "(set by swap())."
            )
        # Trust-critical read: the amounts the claim moves come from the chain.
        out = self.get_swap_output(handle.swap_id)
        self._ensure([], imports)

        inputs = [
            handle.blinding_factor,
            handle.blinded_address,
            handle.swap_id,
            out.token_in,
            out.token_out,
            f"{out.amount_out}u128",
            f"{out.amount_remaining}u128",
        ]
        bound = self._aleo.programs.get(self.program).functions.claim_swap_output(*inputs)

        def build_result(tx_id: str, outputs: list[Any]) -> ClaimResult:
            return ClaimResult(tx_id, out.amount_out, out.amount_remaining)

        return DexCall(self._aleo, bound, build_result)

    def _token_program(self, token_id: str) -> str:
        """Wrapper program holding a token's records, from the API registry."""
        for tok in self.api.get_tokens():
            if tok.address == token_id and tok.wrapper_program:
                return tok.wrapper_program
        raise ValueError(
            f"Cannot resolve the wrapper program for {token_id} — pass "
            "token_in_program= (or token_record=) explicitly."
        )

    # ── Liquidity ────────────────────────────────────────────────────────────

    def _token_decimals(self, token_id: str) -> Optional[int]:
        for t in self.api.get_tokens():
            if t.address == token_id:
                return int(t.decimals)
        return None

    def _quote_expected_out(self, *, token_in_id: str, token_out_id: str,
                            amount_in: int) -> Optional[int]:
        """Base-unit expected output for a trade, via the route quote.

        The route endpoint speaks canonical decimal amounts, the contract
        speaks base units — this converts in both directions using the
        token registry's decimals.  None (spot fallback) when either token
        is unknown or no route is quotable.
        """
        from decimal import Decimal
        dec_in = self._token_decimals(token_in_id)
        dec_out = self._token_decimals(token_out_id)
        if dec_in is None or dec_out is None:
            return None
        try:
            route = self.api.get_route(
                token_in=token_in_id, token_out=token_out_id,
                amount_in=Decimal(amount_in) / (10 ** dec_in))
        except ShieldSwapError:
            return None                   # no quotable route
        if not route.estimated_amount_out:
            return None
        return int(Decimal(route.estimated_amount_out) * (10 ** dec_out))

    def swap_many(
        self,
        *,
        pool_key: str,
        token_in_id: str,
        amount_in: int,
        count: int,
        slippage_bps: int = 50,
        account: Any = None,
    ) -> SwapBatchReport:
        """*count* private swaps of *amount_in* each, with reserved counters.

        Counters come from the journal (no probe races); every handle is
        journaled as soon as its broadcast is accepted (no confirmation
        wait), so a crash mid-batch loses nothing — ``collect_all()`` later
        claims whatever finalized.  A swap the network rejects simply never
        becomes claimable (it stays in ``still_pending``).  A failed
        broadcast burns its counter and the batch continues; failures are
        reported, not raised.  Requires ``from_profile()``.
        """
        if self.journal is None:
            raise ValueError("swap_many() needs a journal — construct with "
                             "ShieldSwap.from_profile().")
        acct = self._account(account)
        # Quote once for the batch: a spot estimate ignores the pool fee, so
        # min-out would exceed the real output and finalize would reject.
        pool = self.get_pool(pool_key)
        token_out_id = pool.token1 if token_in_id == pool.token0 else pool.token0
        expected_out = self._quote_expected_out(
            token_in_id=token_in_id, token_out_id=token_out_id,
            amount_in=amount_in)
        counters = self.journal.reserve_counters(count)
        handles: list[SwapHandle] = []
        failures: list[dict] = []
        for counter in counters:
            ident = blinded_identity_at(self._aleo, acct, self.program, counter)
            try:
                handle = self.swap(pool_key=pool_key, token_in_id=token_in_id,
                                   amount_in=amount_in, slippage_bps=slippage_bps,
                                   expected_out=expected_out,
                                   identity=ident, account=acct
                                   ).delegate(acct, wait=False)
                self.journal.record_swap(handle, counter)
                handles.append(handle)
            except Exception as exc:                  # journal + continue
                self.journal.record_swap_failed(counter, str(exc))
                failures.append({"counter": counter, "error": str(exc)})
        return SwapBatchReport(handles=handles, failures=failures)

    def _position_state(self, position_token_id: str) -> Optional[g.Position]:
        """The on-chain position entry (owed amounts), or None if absent."""
        raw = self._mapping_value("positions", position_token_id)
        return g.Position.from_plaintext(raw) if raw is not None else None

    def collect_all(self, account: Any = None) -> CollectReport:
        """Claim every finalized swap and collect owed fees on open positions.

        Safe to run any time, from any session: works off the journal, skips
        swaps whose finalize hasn't landed (they stay pending for next time),
        never double-claims, and requests exactly the owed amounts the chain
        reports.  Requires ``from_profile()``.
        """
        if self.journal is None:
            raise ValueError("collect_all() needs a journal — construct with "
                             "ShieldSwap.from_profile().")
        acct = self._account(account)
        claimed: list[dict] = []
        still_pending: list[str] = []
        for handle in self.journal.pending_claims():
            try:
                res = self.claim_swap_output(handle, account=acct).delegate(acct)
            except SwapOutputNotFinalizedError:
                still_pending.append(handle.swap_id or "")
                continue
            self.journal.record_claim(handle.swap_id or "", res.transaction_id,
                                      res.amount_out)
            claimed.append({"swap_id": handle.swap_id,
                            "transaction_id": res.transaction_id,
                            "amount_out": res.amount_out})
        fees: list[dict] = []
        for view in self.get_positions(account=acct):
            if view.source != "journal" or not view.position_token_id:
                continue          # scanned-only positions lack journal context
            pos = self._position_state(view.position_token_id)
            if pos is None or (pos.tokens_owed0 == 0 and pos.tokens_owed1 == 0):
                continue
            pool = self.get_pool(view.pool_key)
            # The contract asserts requested <= owed and requested % scale == 0;
            # owed is stored in scaled units, so request exactly owed * scale.
            res = self.collect(
                pool_key=view.pool_key,
                amount0_requested=pos.tokens_owed0 * pool.scale0,
                amount1_requested=pos.tokens_owed1 * pool.scale1,
                account=acct,
            ).delegate(acct)
            fees.append({"position_token_id": view.position_token_id,
                         "pool_key": view.pool_key,
                         "transaction_id": res.transaction_id})
        return CollectReport(claimed, still_pending, fees)

    def create_pool(
        self,
        *,
        token0_id: str,
        token1_id: str,
        fee: int,
        initial_tick: int,
        tick_spacing: Optional[int] = None,
        initial_sqrt_price: Optional[int] = None,
        imports: Optional[dict[str, str]] = None,
        account: Any = None,
    ) -> DexCall[TxResult]:
        """Create a pool — a single public transaction, no records involved.

        The fee tier must be registered with the program (validated before
        submission); tick spacing defaults to the tier's on-chain binding and
        the opening price to the tick's sqrt price.
        """

        self._account(account)
        if not MIN_TICK <= initial_tick < MAX_TICK:
            raise ValueError(f"initial_tick {initial_tick} outside [{MIN_TICK}, {MAX_TICK})")
        # The tier must be registered AND enabled (value exactly true) —
        # a disabled tier is present in the mapping with value false.
        if self._mapping_value("fee_tiers", f"{fee}u16") != "true":
            raise InvalidFeeTierError(f"Fee tier {fee} is not registered with {self.program}")
        spacing = tick_spacing
        if spacing is None:
            raw = self._mapping_value("fee_to_tick_spacing", f"{fee}u16")
            if raw is None:
                raise InvalidFeeTierError(
                    f"Fee {fee} has no tick spacing bound on chain — pass tick_spacing="
                )
            spacing = int(raw.removesuffix("u32"))
        sqrt_price = (initial_sqrt_price if initial_sqrt_price is not None
                      else get_sqrt_price_at_tick(initial_tick))
        self._ensure([], imports)

        inputs = [
            token0_id,
            token1_id,
            f"{fee}u16",
            f"{sqrt_price}u128",
            f"{spacing}u32",
            f"{initial_tick}i32",
        ]
        bound = self._aleo.programs.get(self.program).functions.create_pool(*inputs)

        def build_result(tx_id: str, outputs: list[Any]) -> TxResult:
            key = next((o for o in outputs if isinstance(o, str) and o.endswith("field")), None)
            return TxResult(position_token_id=key, transaction_id=tx_id)

        return DexCall(self._aleo, bound, build_result)

    def _select_position_record(self, pool_key: str, account: Any) -> str:
        """Unspent PositionNFT plaintext for *pool_key* from the shield_swap
        program's own records."""
        records = self._aleo.record_provider.find(
            account, program=self.program, unspent=True)
        plaintext = find_position_plaintext(records, pool_key)
        if plaintext is None:
            raise InsufficientRecordsError(
                f"No unspent PositionNFT record for pool {pool_key} — mint "
                "first or pass position_record=."
            )
        return plaintext

    def mint(
        self,
        *,
        pool_key: str,
        tick_lower: int,
        tick_upper: int,
        amount0_desired: int,
        amount1_desired: int,
        amount0_min: int = 0,
        amount1_min: int = 0,
        token0_program: Optional[str] = None,
        token1_program: Optional[str] = None,
        token0_record: Optional[str] = None,
        token1_record: Optional[str] = None,
        tick_lower_hint: Optional[int] = None,
        tick_upper_hint: Optional[int] = None,
        recipient: Optional[str] = None,
        nonce: Optional[str] = None,
        imports: Optional[dict[str, str]] = None,
        account: Any = None,
    ) -> DexCall[MintResult]:
        """Mint a concentrated-liquidity position as a private PositionNFT.

        Tick bounds are rounded to the pool's spacing; insert hints derive
        from the slot's neighbors unless given explicitly.
        """

        acct = self._account(account)
        pool = self.get_pool(pool_key)
        slot = self.get_slot(pool_key)

        lo = round_tick_to_spacing(tick_lower, slot.tick_spacing)
        hi = round_tick_to_spacing(tick_upper, slot.tick_spacing)
        if lo >= hi:
            raise ValueError(f"Empty tick range after spacing alignment: [{lo}, {hi})")

        lo_hint = tick_lower_hint if tick_lower_hint is not None else pick_insert_hint(slot, lo)
        upper_pred = tick_upper_hint if tick_upper_hint is not None else pick_insert_hint(slot, hi)
        # The finalize inserts tick_lower before validating the upper hint, so
        # when nothing initialized sits between the bounds, the upper tick's
        # predecessor is the just-inserted lower tick.
        hi_hint = lo if (tick_upper_hint is None and lo > upper_pred) else upper_pred

        request = g.MintPositionRequest(
            pool=pool_key, tick_lower=lo, tick_upper=hi,
            amount0_desired=amount0_desired, amount1_desired=amount1_desired,
            amount0_min=amount0_min, amount1_min=amount1_min,
            tick_lower_hint=lo_hint, tick_upper_hint=hi_hint,
        ).to_plaintext()

        record0 = token0_record or select_token_record(
            self._aleo, program=token0_program or self._token_program(pool.token0),
            min_amount=amount0_desired, token_id=pool.token0, account=acct)
        record1 = token1_record or select_token_record(
            self._aleo, program=token1_program or self._token_program(pool.token1),
            min_amount=amount1_desired, token_id=pool.token1, account=acct)
        self._ensure([p for p in (token0_program, token1_program) if p], imports)

        field_nonce = nonce if nonce is not None else generate_field_nonce()
        to = recipient or str(acct.address)

        inputs = [field_nonce, record0, record1, to, request, pool.token0, pool.token1]
        bound = self._aleo.programs.get(self.program).functions.mint(*inputs)
        return DexCall(self._aleo, bound, self._position_result(MintResult))

    def increase_liquidity(
        self,
        *,
        pool_key: str,
        amount0_desired: int,
        amount1_desired: int,
        amount0_min: int = 0,
        amount1_min: int = 0,
        token0_program: Optional[str] = None,
        token1_program: Optional[str] = None,
        token0_record: Optional[str] = None,
        token1_record: Optional[str] = None,
        position_record: Optional[str] = None,
        tick_lower_hint: Optional[int] = None,
        tick_upper_hint: Optional[int] = None,
        imports: Optional[dict[str, str]] = None,
        account: Any = None,
    ) -> DexCall[TxResult]:
        """Add funds to an existing position (range fixed at mint)."""

        acct = self._account(account)
        pool = self.get_pool(pool_key)
        slot = self.get_slot(pool_key)
        position = position_record or self._select_position_record(pool_key, acct)

        decoded = parse_plaintext(position)
        lo_hint = (tick_lower_hint if tick_lower_hint is not None
                   else pick_insert_hint(slot, int(decoded["tick_lower"])))
        hi_hint = (tick_upper_hint if tick_upper_hint is not None
                   else pick_insert_hint(slot, int(decoded["tick_upper"])))

        record0 = token0_record or select_token_record(
            self._aleo, program=token0_program or self._token_program(pool.token0),
            min_amount=amount0_desired, token_id=pool.token0, account=acct)
        record1 = token1_record or select_token_record(
            self._aleo, program=token1_program or self._token_program(pool.token1),
            min_amount=amount1_desired, token_id=pool.token1, account=acct)
        self._ensure([p for p in (token0_program, token1_program) if p], imports)

        inputs = [
            position, record0, record1,
            f"{amount0_desired}u128", f"{amount1_desired}u128",
            f"{amount0_min}u128", f"{amount1_min}u128",
            pool.token0, pool.token1,
            f"{lo_hint}i32", f"{hi_hint}i32",
        ]
        bound = self._aleo.programs.get(self.program).functions.increase_liquidity(*inputs)
        return DexCall(self._aleo, bound, self._position_result(TxResult))

    def decrease_liquidity(
        self,
        *,
        pool_key: str,
        liquidity_to_remove: int,
        amount0_min: int = 0,
        amount1_min: int = 0,
        position_record: Optional[str] = None,
        imports: Optional[dict[str, str]] = None,
        account: Any = None,
    ) -> DexCall[TxResult]:
        """Remove liquidity from a position; owed amounts become collectable."""
        acct = self._account(account)
        position = position_record or self._select_position_record(pool_key, acct)
        self._ensure([], imports)
        inputs = [position, f"{liquidity_to_remove}u128",
                  f"{amount0_min}u128", f"{amount1_min}u128"]
        bound = self._aleo.programs.get(self.program).functions.decrease_liquidity(*inputs)
        return DexCall(self._aleo, bound, self._position_result(TxResult))

    def collect(
        self,
        *,
        pool_key: str,
        amount0_requested: int,
        amount1_requested: int,
        recipient: Optional[str] = None,
        position_record: Optional[str] = None,
        imports: Optional[dict[str, str]] = None,
        account: Any = None,
    ) -> DexCall[TxResult]:
        """Collect owed token amounts from a position."""
        acct = self._account(account)
        pool = self.get_pool(pool_key)
        position = position_record or self._select_position_record(pool_key, acct)
        self._ensure([], imports)
        to = recipient or str(acct.address)
        inputs = [position, f"{amount0_requested}u128", f"{amount1_requested}u128",
                  pool.token0, pool.token1, to]
        bound = self._aleo.programs.get(self.program).functions.collect(*inputs)
        # collect's first output is the re-issued PositionNFT record, not a
        # public field — there is no positional id to read back.
        return DexCall(self._aleo, bound,
                       lambda tx_id, outputs: TxResult(None, tx_id))

    def burn(
        self,
        *,
        pool_key: str,
        position_record: Optional[str] = None,
        account: Any = None,
    ) -> DexCall[TxResult]:
        """Burn an empty position NFT."""
        acct = self._account(account)
        position = position_record or self._select_position_record(pool_key, acct)
        bound = self._aleo.programs.get(self.program).functions.burn(position)
        return DexCall(self._aleo, bound, self._position_result(TxResult))

    @staticmethod
    def _position_result(result_cls: Any) -> Any:
        """Result builder: first public ``field`` output is the position id."""
        def build(tx_id: str, outputs: list[Any]) -> Any:
            pid = next((o for o in outputs if isinstance(o, str) and o.endswith("field")), None)
            return result_cls(pid, tx_id)
        return build
