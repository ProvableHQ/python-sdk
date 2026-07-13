"""ShieldSwap — the synchronous shield_swap client, bound to an ``Aleo`` facade.

Construction wires nothing new: signer, record provider, proving config, and
network all come from the bound client.  On-chain reads and writes live
directly on this object; the off-chain DEX API is namespaced under ``.api``
so a call site always shows whether a value came from the chain or the
service.
"""
from __future__ import annotations

from typing import Any, Optional

from . import _generated as g
from ._core import (
    get_deadline,
    generate_swap_nonce,
    resolve_imports,
    resolve_swap_params,
    select_token_record,
)
from .api import ApiClient, DEFAULT_API_URL
from .derivations import (
    derive_pool_key as _derive_pool_key,
    derive_tick_key as _derive_tick_key,
    next_blinded_identity,
)
from .errors import (
    PoolNotFoundError,
    SwapOutputNotFinalizedError,
)
from .types import ClaimResult, SlotView, SwapHandle
from ._calls import DexCall

_ABSENT = (None, "", "null")


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

    def __repr__(self) -> str:
        return f"ShieldSwap(program={self.program!r}, api={self.api.base_url!r})"

    # ── Mapping plumbing ─────────────────────────────────────────────────────

    def _mapping_value(self, mapping: str, key: str) -> Optional[str]:
        raw = self._aleo.programs.get(self.program).mapping(mapping).get(key)
        if raw in _ABSENT:
            return None
        text = str(raw).strip()
        # Node deployments may return the value JSON-quoted.
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return None if text in ("", "null") else text

    # ── Chain reads ──────────────────────────────────────────────────────────

    def get_pool(self, pool_key: str) -> g.PoolState:
        """Static pool configuration (token pair, fee, decimal scales)."""
        raw = self._mapping_value("pools", pool_key)
        if raw is None:
            raise PoolNotFoundError(pool_key)
        return g.PoolState.from_plaintext(raw)

    def get_slot(self, pool_key: str) -> SlotView:
        """Live trading state (sqrt price, tick, in-range liquidity)."""
        raw = self._mapping_value("slots", pool_key)
        if raw is None:
            raise PoolNotFoundError(pool_key)
        return SlotView(g.Slot.from_plaintext(raw))

    def get_swap_output(self, swap_id: str) -> g.SwapOutput:
        """Chain-computed output of a finalized swap request.

        Raises :class:`SwapOutputNotFinalizedError` when the entry is absent —
        not finalized yet (retry after a few blocks) or already claimed.
        """
        raw = self._mapping_value("swap_outputs", swap_id)
        if raw is None:
            raise SwapOutputNotFinalizedError(swap_id)
        return g.SwapOutput.from_plaintext(raw)

    def is_pool_initialized(self, pool_key: str) -> bool:
        raw = self._mapping_value("initialized_pools", pool_key)
        return raw is not None and "true" in raw

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
        from ._core import parse_token_record_info

        provider = self._aleo.record_provider
        out: dict[str, int] = {}
        for program in programs:
            total = 0
            for rec in provider.find(account, program=program, unspent=True):
                plaintext = (rec.get("record_plaintext") if isinstance(rec, dict)
                             else getattr(rec, "record_plaintext", None))
                info = parse_token_record_info(plaintext) if plaintext else None
                if info is not None:
                    total += info["amount"]
            out[program] = total
        return out

    def get_balances(self, address: Optional[str] = None,
                     account: Any = None) -> dict[str, dict[str, Any]]:
        """Public + private + total per token id, joined via the API's
        token registry.  Defaults to the bound account's address; returns
        only tokens actually held."""
        acct = account if account is not None else self._aleo.default_account
        addr = address or (str(acct.address) if acct is not None else None)
        if addr is None:
            raise ValueError("No address: pass address= or set aleo.default_account")

        tokens = self.api.get_tokens()
        by_program = {t.wrapper_program: t for t in tokens if t.wrapper_program}
        private = self.get_private_balances(list(by_program), account=acct)

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

    def swap(
        self,
        *,
        pool_key: str,
        token_in_id: str,
        amount_in: int,
        slippage_bps: int = 50,
        expected_out: Optional[int] = None,
        sqrt_price_limit: Optional[int] = None,
        deadline_offset_blocks: int = 100,
        nonce: Optional[int] = None,
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
        identity = next_blinded_identity(self._aleo, acct, self.program)

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
        # statically; resolve their sources (cached) unless overridden.
        resolve_imports(self._aleo, token_programs, imports)

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
        resolve_imports(self._aleo, [], imports)

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
