"""AsyncShieldSwap — the shield_swap client over an ``AsyncAleo`` facade.

Same surface as :class:`~aleo_shield_swap.client.ShieldSwap`, with every
I/O method ``async``.  Pure logic (param resolution, derivations, tick math,
types) is shared from ``_core``/``derivations`` — only the I/O differs.
"""
from __future__ import annotations

from typing import Any, Callable, Generic, Optional, TypeVar

from aleo.codegen.runtime import parse_plaintext

from . import _generated as g
from ._core import (
    generate_field_nonce,
    generate_swap_nonce,
    parse_token_record_info,
    resolve_swap_params,
)
from .api import AsyncApiClient, DEFAULT_API_URL
from .derivations import (
    BlindedIdentity,
    derive_blinded_address,
    derive_blinding_factor,
    derive_pool_key as _derive_pool_key,
    derive_tick_key as _derive_tick_key,
)
from .errors import (
    InsufficientRecordsError,
    InvalidFeeTierError,
    PoolNotFoundError,
    SwapOutputNotFinalizedError,
)
from .tick_hints import pick_insert_hint
from .tick_math import MAX_TICK, MIN_TICK, get_sqrt_price_at_tick, round_tick_to_spacing
from .types import ClaimResult, MintResult, SlotView, SwapHandle, TxResult

_ABSENT = (None, "", "null")

R = TypeVar("R")


class AsyncDexCall(Generic[R]):
    """Async mirror of :class:`~aleo_shield_swap._calls.DexCall`."""

    def __init__(self, aleo: Any, bound: Any,
                 build_result: Callable[[str, list[Any]], R]) -> None:
        self._aleo = aleo
        self._bound = bound
        self._build = build_result

    def simulate(self, account: Any = None) -> Any:
        # AsyncBoundCall.simulate is synchronous (local authorization).
        return self._bound.simulate(account)

    async def transact(self, account: Any = None, **fee_kwargs: Any) -> R:
        tx = await self._bound.build_transaction(account, **fee_kwargs)
        outputs = [o.get("value") for group in tx.outputs() for o in group]
        await self._aleo.network.submit_transaction(tx.raw)
        return self._build(tx.id, outputs)

    async def delegate(self, account: Any = None, **fee_kwargs: Any) -> R:
        payload = await self._bound.delegate(account, **fee_kwargs)
        from ._calls import DexCall
        tx_id = DexCall._extract_tx_id(payload)
        await self._aleo.network.wait_for_transaction(tx_id)
        decoded = await self._aleo.decode_transition(tx_id)
        outputs = [o.get("value") if isinstance(o, dict) else o
                   for o in decoded.get("outputs", [])]
        return self._build(tx_id, outputs)


class AsyncShieldSwap:
    """Typed async client for the shield_swap AMM over ``AsyncAleo``."""

    def __init__(self, aleo: Any, *, program: str = g.PROGRAM_ID,
                 api_url: str = DEFAULT_API_URL) -> None:
        self._aleo = aleo
        self.program = program
        self.api = AsyncApiClient(api_url)

    def __repr__(self) -> str:
        return f"AsyncShieldSwap(program={self.program!r}, api={self.api.base_url!r})"

    # ── Mapping plumbing ─────────────────────────────────────────────────────

    async def _mapping_value(self, mapping: str, key: str) -> Optional[str]:
        prog = await self._aleo.programs.get(self.program)
        raw = await prog.mapping(mapping).get(key)
        if raw in _ABSENT:
            return None
        text = str(raw).strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return None if text in ("", "null") else text

    # ── Chain reads ──────────────────────────────────────────────────────────

    async def get_pool(self, pool_key: str) -> g.PoolState:
        raw = await self._mapping_value("pools", pool_key)
        if raw is None:
            raise PoolNotFoundError(pool_key)
        return g.PoolState.from_plaintext(raw)

    async def get_slot(self, pool_key: str) -> SlotView:
        raw = await self._mapping_value("slots", pool_key)
        if raw is None:
            raise PoolNotFoundError(pool_key)
        return SlotView(g.Slot.from_plaintext(raw))

    async def get_swap_output(self, swap_id: str) -> g.SwapOutput:
        raw = await self._mapping_value("swap_outputs", swap_id)
        if raw is None:
            raise SwapOutputNotFinalizedError(swap_id)
        return g.SwapOutput.from_plaintext(raw)

    async def is_pool_initialized(self, pool_key: str) -> bool:
        raw = await self._mapping_value("initialized_pools", pool_key)
        return raw is not None and "true" in raw

    # ── Pure derivations (sync — no I/O) ─────────────────────────────────────

    def derive_pool_key(self, token0: str, token1: str, fee: int) -> str:
        return _derive_pool_key(token0, token1, fee, network=self._aleo.network_name)

    def derive_tick_key(self, pool_key: str, tick: int) -> str:
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
        candidates: list[tuple[int, str]] = []
        for rec in records:
            plaintext = (rec.get("record_plaintext") if isinstance(rec, dict)
                         else getattr(rec, "record_plaintext", None))
            if not plaintext:
                continue
            info = parse_token_record_info(plaintext)
            if info is None or info["amount"] < min_amount:
                continue
            if token_id is not None and "token_id" in info and info["token_id"] != token_id:
                continue
            candidates.append((info["amount"], plaintext))
        if not candidates:
            raise InsufficientRecordsError(
                f"No unspent {program} record covers {min_amount}.")
        return min(candidates)[1]

    async def _select_position_record(self, pool_key: str, account: Any) -> str:
        provider = self._aleo.record_provider
        for rec in await provider.find(account, program=self.program, unspent=True):
            plaintext = (rec.get("record_plaintext") if isinstance(rec, dict)
                         else getattr(rec, "record_plaintext", None))
            if not plaintext:
                continue
            try:
                decoded = parse_plaintext(plaintext)
            except (ValueError, TypeError):
                continue
            if isinstance(decoded, dict) and decoded.get("pool") == pool_key:
                return plaintext
        raise InsufficientRecordsError(
            f"No unspent PositionNFT record for pool {pool_key}.")

    async def _token_program(self, token_id: str) -> str:
        for tok in await self.api.get_tokens():
            if tok.address == token_id and tok.wrapper_program:
                return tok.wrapper_program
        raise ValueError(
            f"Cannot resolve the wrapper program for {token_id} — pass "
            "token_in_program= (or token_record=) explicitly.")

    def _account(self, account: Any = None) -> Any:
        acct = account if account is not None else self._aleo.default_account
        if acct is None:
            raise ValueError("No signer: pass account= or set aleo.default_account")
        return acct

    # ── Balances ─────────────────────────────────────────────────────────────

    async def get_private_balances(self, programs: list[str],
                                   account: Any = None) -> dict[str, int]:
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
        acct = account if account is not None else self._aleo.default_account
        addr = address or (str(acct.address) if acct is not None else None)
        if addr is None:
            raise ValueError("No address: pass address= or set aleo.default_account")
        tokens = await self.api.get_tokens()
        by_program = {t.wrapper_program: t for t in tokens if t.wrapper_program}
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
                   account: Any = None) -> AsyncDexCall[SwapHandle]:
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

        inputs = [
            record, identity.blinding_factor, identity.blinded_address,
            pool_key, resolved.zero_for_one,
            f"{amount_in}u128", f"{resolved.amount_out_min}u128",
            f"{resolved.sqrt_price_limit}u128",
            f"{swap_nonce}u64", f"{deadline}u32",
            pool.token0, pool.token1,
        ]
        prog = await self._aleo.programs.get(self.program)
        bound = prog.functions.swap(*inputs)

        def build_result(tx_id: str, outputs: list[Any]) -> SwapHandle:
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
                                account: Any = None) -> AsyncDexCall[ClaimResult]:
        self._account(account)
        if not handle.swap_id:
            raise ValueError("handle.swap_id is not set — recover it from the "
                             "confirmed request transaction before claiming.")
        if not handle.blinding_factor or not handle.blinded_address:
            raise ValueError("Claims need handle.blinding_factor and "
                             "handle.blinded_address (set by swap()).")
        out = await self.get_swap_output(handle.swap_id)
        inputs = [handle.blinding_factor, handle.blinded_address, handle.swap_id,
                  out.token_in, out.token_out,
                  f"{out.amount_out}u128", f"{out.amount_remaining}u128"]
        prog = await self._aleo.programs.get(self.program)
        bound = prog.functions.claim_swap_output(*inputs)
        return AsyncDexCall(self._aleo, bound,
                            lambda tx_id, _o: ClaimResult(tx_id, out.amount_out,
                                                          out.amount_remaining))
