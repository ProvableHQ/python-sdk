"""Framework-neutral agent tools over a :class:`~aleo_bridge.client.Bridge`.

``bridge_tools()`` returns tool definitions in the Claude API ``tools=`` shape
(name / description / input_schema); ``dispatch_tool(bridge, name, args)``
executes one and returns JSON-serializable data.  Reads are open.  Writes
require ``confirm: true`` — without it they return the quote (or recovered
progress / built call) plus ``how_to_confirm`` and move nothing.  Agents never
carry ``Plan`` objects: ``bridge_execute`` takes the quote inputs and
re-quotes internally; recovery tools take the checkpoint dict returned by the
previous write.  Amounts in and out are human units; ints stay ints.

Three rules keep a model from doing damage with this surface:

* **Secrets never leave the process.**  ``_serialize`` drops the private-mint
  secret, the Circle attestation body and the proved transaction bytes from any
  receipt it renders, a previewed program call renders record-shaped inputs as
  ``"<record>"``, and no tool ever echoes its own arguments back.  (The
  xReserve *hook data* — a public commitment, not the secret that opens it —
  stays on the quote and inside the checkpoint, because ``lifecycle.resume``
  refuses to resume a deposit whose checkpoint has lost it.)
* **A private mint never gets a default secret.**  ``mint_mode="private"``
  without a ``secret_nonce`` is a structured error, never a quiet ``"0scalar"``
  that commits to a hook nobody can reproduce.
* **Failures come back as data.**  Every :class:`~aleo_bridge.errors.BridgeError`
  is rendered ``{"error", "error_type", "how_to_fix"}`` so the model can fix the
  input or set up the missing connection instead of crashing its own loop; a
  send whose outcome is ambiguous also carries ``{"next": "recover"}`` and the
  last checkpoint.  Programming errors (``ValueError``, ``TypeError``) still
  propagate.
"""
from __future__ import annotations

import copy
import dataclasses
import enum
import re
from typing import Any, Callable

from . import lifecycle
from .checkpoint import Checkpoint, create_checkpoint
from .errors import BridgeError, ConfigurationError
from .registry import DEFAULT_REGISTRY, Registry
from .types import EvmXReserveQuote, Receipt
from .units import format_decimal_amount

_S = {"type": "string"}
_I = {"type": "integer"}
_B = {"type": "boolean"}
HOW_TO_CONFIRM = "re-call with confirm=true"

EVM_HOW_TO_FIX = ("set EVM_PRIVATE_KEY + ETHEREUM_RPC_URL in the environment (or pass "
                  "ethereum=Ethereum(...) to Bridge(...)) and retry")
SOLANA_HOW_TO_FIX = ("set SOLANA_PRIVATE_KEY (and optionally SOLANA_RPC_URL) in the environment (or pass "
                     "solana=Solana(...) to Bridge(...)) and retry")
NONCE_HOW_TO_FIX = ("re-call with secret_nonce set to the value the user kept from execute — the SDK "
                    "never stored it, and no other value can open the commitment")
RECOVER_HOW_TO_FIX = ("the source step may already be on the wire: call bridge_get_progress with the "
                      "checkpoint (or bridge_pending) before doing anything else — never bridge_execute again")

#: Receipt ``protocol_state`` keys that are dropped from every rendered receipt: the private-mint
#: secret and record plaintext (secret), the Circle attestation body (secret-adjacent and large),
#: the proved transaction bytes and the hook commitment (both live in the checkpoint, which is what
#: the recovery verbs actually consume).
_REDACTED_STATE_KEYS = frozenset({
    "payload", "attestation", "secretnonce", "record", "recordplaintext", "privatekey",
    "hookdata", "preparedtransaction", "prepareddestinationtransaction",
})


def _schema(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


# ── serialization ─────────────────────────────────────────────────────────────

def _max_fee_entry(quote: EvmXReserveQuote, registry: Registry | None) -> dict[str, Any] | None:
    """The xReserve max fee as a fee entry: ``EvmXReserveQuote.fees`` is empty, and an agent that
    cannot see the fee cannot tell the user what the transfer costs."""
    plan = quote.plan
    if plan is None:
        return None
    try:
        asset = (registry or DEFAULT_REGISTRY).asset(plan.source_asset_id)
        amount = format_decimal_amount(quote.max_fee_atomic, asset.decimals)
    except BridgeError:
        return None
    return {"kind": "protocol", "chain_id": asset.chain_id, "asset_id": asset.id, "amount": amount,
            "estimated": True, "label": "xReserve max fee"}


def _serialize(value: Any, registry: Registry | None = None) -> Any:
    """Dataclasses → dicts, bytes → 0x hex, enums → values, tuples → lists; ints stay ints.

    Checkpoints render through ``to_dict()`` (the camelCase form the recovery tools take back),
    receipts lose their secret / bulky ``protocol_state`` entries, and an ``EvmXReserveQuote``
    gains the synthetic max-fee entry its empty ``fees`` tuple would otherwise hide.
    """
    if isinstance(value, Checkpoint):
        return value.to_dict()
    if isinstance(value, Receipt):
        return {f.name: (_redacted_state(getattr(value, f.name)) if f.name == "protocol_state"
                         else _serialize(getattr(value, f.name), registry))
                for f in dataclasses.fields(value)}
    if isinstance(value, EvmXReserveQuote):
        out = {f.name: _serialize(getattr(value, f.name), registry) for f in dataclasses.fields(value)}
        entry = _max_fee_entry(value, registry)
        if entry is not None:
            out["fees"] = list(out.get("fees") or []) + [entry]
        return out
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _serialize(getattr(value, f.name), registry) for f in dataclasses.fields(value)}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (bytes, bytearray)):
        return "0x" + bytes(value).hex()
    if isinstance(value, dict):
        return {str(k): _serialize(v, registry) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize(v, registry) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


#: A record, in any form an Aleo call input can carry it: the plaintext ``{ owner: … }`` a record
#: selection returns (``privacy.py``'s ``select_record``), the ``record1…`` ciphertext, a
#: ``x.record`` locator, or any plaintext carrying a nonce / a credits amount.  A record IS the
#: private balance — showing one to a model (or writing it into a transcript) spends its privacy.
_RECORD_SHAPED = re.compile(r"""
    (^record1[a-z0-9]{8,})       # record ciphertext
  | (\.record\b)                 # a record locator
  | (^\{\s*owner\s*:)            # record plaintext, as select_record returns it
  | (\b_nonce\s*:)               # …or anything else carrying a record's nonce
  | (\bmicrocredits\s*:)         # …or a credits record's amount
""", re.IGNORECASE | re.VERBOSE)


def _summarize_input(value: Any) -> str:
    """One program-call input, rendered for a human or a model: record-shaped → ``"<record>"``.

    Everything public — the amount literal, the recipient address, a Merkle path — passes through,
    so the preview still says what the call does.
    """
    text = value if isinstance(value, str) else str(value)
    return "<record>" if _RECORD_SHAPED.search(text.strip()) else text


def _redacted_state(state: Any) -> Any:
    if not isinstance(state, dict):
        return _serialize(state)
    return {str(k): _serialize(v) for k, v in state.items()
            if str(k).replace("_", "").lower() not in _REDACTED_STATE_KEYS}


# ── errors ────────────────────────────────────────────────────────────────────

def _how_to_fix(exc: Exception) -> str | None:
    message = str(exc).lower()
    if "secret_nonce" in message or "secret nonce" in message:
        return NONCE_HOW_TO_FIX
    if "evm_private_key" in message or "ethereum_rpc_url" in message or "ethereum=ethereum" in message:
        return EVM_HOW_TO_FIX
    if "solana" in message and ("not configured" in message or "solana_private_key" in message):
        return SOLANA_HOW_TO_FIX
    return None


def _error_payload(exc: Exception, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": str(exc), "error_type": exc.__class__.__name__}
    fix = _how_to_fix(exc)
    if fix is not None:
        payload["how_to_fix"] = fix
    payload.update(extra)
    return payload


def _connection_gap(bridge: Any, source_chain: Any) -> dict[str, Any] | None:
    """Probe the connection a transfer out of *source_chain* would sign with (spec §10: a missing
    connection is a configuration answer the model can act on, not an exception)."""
    try:
        chain = bridge.registry.chain(source_chain)
        family = chain.family
    except (BridgeError, TypeError):
        return None                                   # let the real lookup produce the real error
    if family == "evm" and getattr(bridge, "ethereum", None) is None:
        return _error_payload(ConfigurationError(
            f"No Ethereum connection is configured, and transfers out of {chain.id} are signed "
            "there."), how_to_fix=EVM_HOW_TO_FIX)
    if family == "solana" and getattr(bridge, "solana", None) is None:
        return _error_payload(ConfigurationError(
            f"No Solana connection is configured, and transfers out of {chain.id} are signed "
            "there."), how_to_fix=SOLANA_HOW_TO_FIX)
    return None


def _missing_nonce(what: str) -> dict[str, Any]:
    return _error_payload(ConfigurationError(
        f"a secret_nonce is required for a private mint: {what} commits to (recipient, secret_nonce) "
        "and bridge_complete needs the same value again — keep it, the SDK never stores it"),
        how_to_fix=NONCE_HOW_TO_FIX)


# ── shared argument handling ──────────────────────────────────────────────────

_QUOTE_PROPS = {
    "source_chain": {**_S, "description": "Chain the funds leave: 'ethereum', 'solana', 'aleo' (testnet: 'sepolia', 'aleo-testnet')."},
    "source_asset": {**_S, "description": "Asset key on the source chain: 'usdc', 'eth', 'wbtc', 'usdt', 'sol', 'usdcx'."},
    "destination_chain": {**_S, "description": "Chain the funds arrive on: 'aleo', 'ethereum', 'solana'."},
    "destination_asset": {**_S, "description": "Asset key on the destination chain ('usdcx', 'eth', ...). Only needed when "
                                               "the source asset can arrive as more than one asset."},
    "amount": {**_S, "description": "Positive decimal amount in source-asset display units (e.g. '2', '0.001')."},
    "recipient": {**_S, "description": "Destination-chain address that receives the funds."},
    "sender": {**_S, "description": "Optional source-chain address; must be the configured connection's address."},
    "bridge_protocol": {**_S, "enum": ["xreserve", "hyperlane"], "description": "Only needed when both protocols serve the pair."},
    "mint_mode": {**_S, "enum": ["public", "record", "private"],
                  "description": "xReserve into Aleo only. 'private' requires the user to complete the mint later with the same secret_nonce."},
    "secret_nonce": {**_S, "description": "Private-mint commitment secret, REQUIRED when mint_mode is 'private' "
                                          "(there is no default). The user must keep it for bridge_complete; the SDK never stores it."},
}
_QUOTE_REQUIRED = ["source_chain", "source_asset", "destination_chain", "amount", "recipient"]
_CONFIRM = {"confirm": {**_B, "description": "Set true to move funds. Without it the quote is returned and nothing is submitted."}}
_CHECKPOINT = {"checkpoint": {"type": "object",
                              "description": "The checkpoint dict returned by bridge_get_progress, by an entry of "
                                             "bridge_pending, or by bridge_execute / bridge_resume / bridge_complete "
                                             "(including the one an interrupted write hands back with next='recover')."}}


def _quote_kwargs(args: dict[str, Any]) -> dict[str, Any]:
    """Quote inputs. ``secret_nonce`` defaults to ``"0scalar"`` only for a non-private mint —
    a private one must carry its own (checked by :func:`_private_nonce_gap` first)."""
    mint_mode = args.get("mint_mode") or "public"
    secret_nonce = args.get("secret_nonce")
    return dict(source_chain=args["source_chain"], source_asset=args["source_asset"],
                destination_chain=args["destination_chain"], destination_asset=args.get("destination_asset"),
                bridge_protocol=args.get("bridge_protocol"), amount=str(args["amount"]),
                recipient=args["recipient"], sender=args.get("sender"), mint_mode=mint_mode,
                secret_nonce=secret_nonce if mint_mode == "private" else (secret_nonce or "0scalar"))


def _private_nonce_gap(args: dict[str, Any]) -> dict[str, Any] | None:
    if (args.get("mint_mode") or "public") == "private" and not args.get("secret_nonce"):
        return _missing_nonce("the deposit")
    return None


def _with_checkpoint(bridge: Any, progress: Any) -> dict[str, Any]:
    checkpoint = create_checkpoint(progress.plan, progress.receipt, bridge.registry)
    return {"progress": _serialize(progress, bridge.registry), "checkpoint": checkpoint.to_dict()}


def _confirmation(**payload: Any) -> dict[str, Any]:
    return {"confirmation_required": True, **payload, "how_to_confirm": HOW_TO_CONFIRM}


def _write(b: Any, call: Callable[[list[Any]], Any]) -> dict[str, Any]:
    """Run one fund-moving verb, collecting its checkpoints, and render either outcome.

    Success is ``{"progress", "checkpoint"}``.  A failure is never a retry cue: every write here is
    single-use and a lost RPC answer is ambiguous, so the error comes back with ``next: "recover"``,
    how-to-fix pointing at ``bridge_get_progress``, and the last checkpoint that made it out —
    which, for a write interrupted after proving, is the only copy of those bytes.
    """
    seen: list[Any] = []
    try:
        progress = call(seen)
    except BridgeError as exc:
        payload = _error_payload(exc, next="recover")
        payload["how_to_fix"] = RECOVER_HOW_TO_FIX
        if seen:
            payload["checkpoint"] = _serialize(seen[-1], b.registry)
        return payload
    return _with_checkpoint(b, progress)


# ── reads ─────────────────────────────────────────────────────────────────────

def _h_status(b, a):
    return _serialize(b.status(), b.registry)


def _h_list_assets(b, a):
    return _serialize(b.registry.assets(chain=a.get("chain"), symbol=a.get("symbol"),
                                        environment=a.get("environment", b.environment)), b.registry)


def _h_list_routes(b, a):
    return _serialize(b.registry.routes(source_chain=a.get("source_chain"), source_asset=a.get("source_asset"),
                                        destination_chain=a.get("destination_chain"),
                                        destination_asset=a.get("destination_asset"),
                                        bridge_protocol=a.get("bridge_protocol"), symbol=a.get("symbol"),
                                        include_unavailable=bool(a.get("include_unavailable", False)),
                                        environment=a.get("environment", b.environment)), b.registry)


def _h_quote(b, a):
    gap = _private_nonce_gap(a) or _connection_gap(b, a.get("source_chain"))
    if gap is not None:
        return gap
    return _serialize(lifecycle.quote(b, **_quote_kwargs(a)), b.registry)


def _h_get_progress(b, a):
    # A checkpoint comes back with the progress: an agent that started from a stale one (or from
    # bridge_pending) can hand this one straight to bridge_resume / bridge_complete.
    return _with_checkpoint(b, lifecycle.recover(b, a["checkpoint"]))


def _h_pending(b, a):
    """Every stored checkpoint with the ``Progress`` reconstructed for it — offline, one entry each.

    Reconstruction is :func:`lifecycle.progress_from_checkpoint`, exactly what ``Bridge.pending()``
    runs per record (no network read, so one unreachable chain can never hide the others). It is
    called here rather than through ``Bridge.pending()`` because that verb returns bare ``Progress``
    objects: it does not pair each one with the checkpoint that produced it — which is what the
    recovery tools take back, and must be the STORED record, not one re-derived from a receipt an
    offline reconstruction may have flattened.

    Nothing is dropped. A record this build cannot interpret at all becomes one error entry naming
    its ``checkpoint_id``, and a file the store could not even read back as a checkpoint becomes one
    naming its ``path`` — the healthy entries still come back beside them, so a corrupt or stale
    file can never make a transfer that is still on the wire invisible. Both carry
    ``"next": "failed"``, the shape ``Bridge.pending()`` also reports, so one reader handles either.
    """
    store = getattr(b, "checkpoints", None)
    if store is None:
        return []
    lister = getattr(store, "list_with_problems", None)
    checkpoints, problems = lister() if callable(lister) else (store.list(), [])
    out: list[dict[str, Any]] = []
    for cp in checkpoints:
        try:
            progress = lifecycle.progress_from_checkpoint(b.registry, cp)
        except BridgeError as exc:
            out.append(_error_payload(exc, next="failed", checkpoint_id=cp.id))
            continue
        out.append({"progress": _serialize(progress, b.registry), "checkpoint": cp.to_dict()})
    out.extend({"next": "failed", **problem.to_dict()} for problem in problems)
    return out


# ── writes (confirm-gated) ────────────────────────────────────────────────────

def _h_execute(b, a):
    gap = _private_nonce_gap(a) or _connection_gap(b, a.get("source_chain"))
    if gap is not None:
        return gap
    quote = lifecycle.quote(b, **_quote_kwargs(a))
    if not a.get("confirm"):
        return _confirmation(quote=_serialize(quote, b.registry))
    return _write(b, lambda seen: lifecycle.execute(
        b, quote.plan, on_checkpoint=seen.append, mode=a.get("mode"), proving=a.get("proving", "delegate"),
        gas_payment_microcredits=a.get("gas_payment_microcredits"), secret_nonce=a.get("secret_nonce")))


def _resume_needs_nonce(bridge: Any, progress: Any) -> bool:
    """True when ``lifecycle.resume`` would demand a ``secret_nonce`` for this progress.

    Only an EVM-source xReserve deposit with ``mint_mode == "private"``: it re-quotes the hook the
    approval committed to, which is derived from the secret. An Aleo leg rebroadcasts proved bytes
    and needs nothing.
    """
    plan = progress.plan
    if plan.protocol != "xreserve" or getattr(plan, "mint_mode", "public") != "private":
        return False
    try:
        resolved = lifecycle.resolve_route(bridge.registry, plan)
    except BridgeError:
        return False                       # an unresolvable route has a louder problem than this
    return resolved.source_chain.family == "evm"


def _h_resume(b, a):
    progress = lifecycle.recover(b, a["checkpoint"])                 # reads only
    if _resume_needs_nonce(b, progress) and not a.get("secret_nonce"):
        # Pre-checked like _h_complete: lifecycle.resume refuses this before any RPC, but routed
        # through _write that refusal would come back as next: "recover" — "it may already be on
        # the wire" — for a transfer that never left the process. Say what is actually missing.
        return _missing_nonce("the deposit this resume finishes")
    if not a.get("confirm"):
        return _confirmation(progress=_serialize(progress, b.registry))
    return _write(b, lambda seen: lifecycle.resume(b, progress, on_checkpoint=seen.append,
                                                   secret_nonce=a.get("secret_nonce")))


def _has_prepared_destination(progress: Any) -> bool:
    """A mint whose bytes are already proved rebroadcasts without the secret."""
    state = getattr(progress.receipt, "protocol_state", {}) or {}
    return bool(state.get("preparedDestinationTransaction"))


def _h_complete(b, a):
    progress = lifecycle.recover(b, a["checkpoint"])                 # reads only
    secret_nonce = a.get("secret_nonce")
    if not secret_nonce and not _has_prepared_destination(progress):
        return _missing_nonce("the deposit this mint finishes")
    if not a.get("confirm"):
        return _confirmation(progress=_serialize(progress, b.registry))
    return _write(b, lambda seen: lifecycle.complete(b, progress, on_checkpoint=seen.append,
                                                     secret_nonce=secret_nonce))


def _privacy(b, a, direction: str):
    kwargs = dict(asset=a["asset"], amount=a.get("amount"), amount_atomic=a.get("amount_atomic"))
    call = b.shield(**kwargs) if direction == "shield" else b.unshield(**kwargs)
    if not a.get("confirm"):
        # An unshield's inputs carry the selected record's plaintext — the private balance itself.
        return _confirmation(call={"program": call.program_id, "function": call.function_name,
                                   "inputs": [_summarize_input(i) for i in call.inputs]})
    return _serialize(call.delegate(), b.registry)


def _h_shield(b, a):
    return _privacy(b, a, "shield")


def _h_unshield(b, a):
    return _privacy(b, a, "unshield")


# ── tool table ────────────────────────────────────────────────────────────────

_READ_TOOLS: list[tuple[str, str, dict[str, Any], Callable[[Any, dict[str, Any]], Any]]] = [
    ("bridge_status",
     "Re-orient: environment, registry version, the configured Aleo/Ethereum/Solana addresses with balances of "
     "every bridge asset (atomic units), and pending (in-flight) transfers. Run this FIRST in any session.",
     _schema({}, []), _h_status),
    ("bridge_list_assets",
     "Assets that can be bridged, with chain, symbol, decimals and on-chain locator. Filter by chain id "
     "(aleo, ethereum, solana, aleo-testnet, sepolia) or symbol.",
     _schema({"chain": _S, "symbol": _S, "environment": {**_S, "enum": ["mainnet", "testnet"]}}, []), _h_list_assets),
    ("bridge_list_routes",
     "Supported directions and their protocol (xreserve = USDC<->USDCx via Circle; hyperlane = ETH/WBTC/USDT/SOL). "
     "Active routes move funds; metadata-required ones are listed but refused by quote/execute.",
     _schema({"source_chain": _S, "source_asset": _S, "destination_chain": _S, "destination_asset": _S,
              "bridge_protocol": {**_S, "enum": ["xreserve", "hyperlane"]}, "symbol": _S,
              "include_unavailable": _B, "environment": {**_S, "enum": ["mainnet", "testnet"]}}, []), _h_list_routes),
    ("bridge_quote",
     "Validate and price a transfer: route, fees (human units), amount_out, approval needs. Reads chain state, "
     "never signs. ALWAYS quote before bridge_execute and show the user fees and amount_out. A private mint "
     "(mint_mode='private') must carry the user's own secret_nonce — there is no default.",
     _schema(_QUOTE_PROPS, _QUOTE_REQUIRED), _h_quote),
    ("bridge_get_progress",
     "Recover a transfer's state from a checkpoint (reads only). Returns {progress, checkpoint}: progress.next tells "
     "what to do — wait (call again later), resume (bridge_resume), complete (bridge_complete), done, failed — and "
     "the checkpoint is the fresh one to pass to whichever of those you call.",
     _schema(_CHECKPOINT, ["checkpoint"]), _h_get_progress),
    ("bridge_pending",
     "Every in-flight transfer in this profile's checkpoint store, one {progress, checkpoint} entry each, "
     "reconstructed offline (no chain read, so its progress can lag: bridge_get_progress refreshes one against live "
     "state). A record too damaged to interpret comes back as {error, checkpoint_id} in its place.",
     _schema({}, []), _h_pending),
]

_WRITE_TOOLS: list[tuple[str, str, dict[str, Any], Callable[[Any, dict[str, Any]], Any]]] = [
    ("bridge_execute",
     "Start a transfer: re-quotes the same inputs, then commits funds on the source chain. Requires confirm=true; "
     "without it returns the quote and moves nothing. The source step is IRREVERSIBLE once broadcast — afterwards "
     "use bridge_get_progress with the returned checkpoint, never bridge_execute again. For mint_mode=private the "
     "user must supply secret_nonce here (no default) and keep it for bridge_complete.",
     _schema({**_QUOTE_PROPS, "mode": {**_S, "description": "Aleo-origin only: caller|signer (Hyperlane) or "
                                                            "private|public|public-as-signer (xReserve burn)."},
              "gas_payment_microcredits": _I, "proving": {**_S, "enum": ["delegate", "local"]}, **_CONFIRM},
             _QUOTE_REQUIRED), _h_execute),
    ("bridge_resume",
     "Finish an interrupted source submission (progress.next == 'resume'): rebroadcasts the identical proved Aleo "
     "transaction or authorizes the single missing EVM step after re-scanning history. Requires confirm=true. "
     "An interrupted EVM xReserve deposit needs the same secret_nonce used at execute.",
     _schema({**_CHECKPOINT, "secret_nonce": {**_S, "description": "The value used at bridge_execute; required to "
                                                                    "resume a private-mint xReserve deposit."},
              **_CONFIRM}, ["checkpoint"]), _h_resume),
    ("bridge_complete",
     "Submit the private USDCx mint (progress.next == 'complete') with the secret_nonce used at execute. "
     "Requires confirm=true. Submits exactly one Aleo transaction.",
     _schema({**_CHECKPOINT,
              "secret_nonce": {**_S, "description": "The private-mint secret used at bridge_execute — required "
                                                    "(there is no default); only an already-proved mint can be "
                                                    "rebroadcast without it."},
              **_CONFIRM}, ["checkpoint"]), _h_complete),
    ("bridge_shield",
     "Move a public Aleo balance of a bridged asset (aleo/eth, aleo/wbtc, aleo/usdt, aleo/sol, aleo/usdcx) into a "
     "private record. Requires confirm=true; without it returns the program call for review.",
     _schema({"asset": _S, "amount": _S, "amount_atomic": _I, **_CONFIRM}, ["asset"]), _h_shield),
    ("bridge_unshield",
     "Move a private record of a bridged asset back to the public balance (needed before an Aleo-origin Hyperlane "
     "transfer). Requires confirm=true.",
     _schema({"asset": _S, "amount": _S, "amount_atomic": _I, **_CONFIRM}, ["asset"]), _h_unshield),
]

_HANDLERS: dict[str, Callable[[Any, dict[str, Any]], Any]] = {
    name: handler for name, _, _, handler in _READ_TOOLS + _WRITE_TOOLS}


def bridge_tools(include_writes: bool = True) -> list[dict[str, Any]]:
    """Tool definitions (Claude API ``tools=`` shape); ``include_writes=False`` keeps only reads.

    Every call returns a fresh deep copy: a caller that tailors a schema (or a framework that
    annotates one in place) can never edit the module's own table out from under everyone else.
    """
    tools = _READ_TOOLS + (_WRITE_TOOLS if include_writes else [])
    return [{"name": name, "description": desc, "input_schema": copy.deepcopy(schema)}
            for name, desc, schema, _ in tools]


def dispatch_tool(bridge: Any, name: str, args: dict[str, Any] | None = None) -> Any:
    """Execute one tool against *bridge*; returns JSON-serializable data.

    A :class:`~aleo_bridge.errors.BridgeError` becomes ``{"error", "error_type", ...}`` — the model
    gets to fix the input rather than lose its loop.  An unknown tool name is a ``ValueError``.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown bridge tool: {name!r}")
    try:
        return handler(bridge, dict(args or {}))
    except BridgeError as exc:
        return _error_payload(exc)


__all__ = ["bridge_tools", "dispatch_tool", "HOW_TO_CONFIRM"]
