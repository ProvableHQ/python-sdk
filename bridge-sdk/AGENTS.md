# aleo-bridge — agent guide

> GENERATED from SDK docstrings by `codegen/gen_context.py` — do not
> edit by hand; edit the docstrings and regenerate.

Typed Python client that moves assets between Aleo, Ethereum and Solana
over the reviewed Hyperlane warp routes and Circle xReserve deployments
(`pip install aleo-bridge-sdk[evm,solana]`, imports as `aleo_bridge`).
MCP alternative: `python -m aleo_bridge.mcp` exposes the same lifecycle as
tools; `aleo_bridge.agent.bridge_tools()` gives Claude-shape tool schemas.
Registry version `2026-08-31.solana-deposits.1`.

## Tier 1 — the lifecycle (quote → execute → wait, then resume / complete as asked)

```python
from aleo_bridge import Bridge

bridge = Bridge.from_env()                      # BRIDGE_PRIVATE_KEY (+ EVM/Solana keys) from the environment
print(bridge.status())                          # addresses, balances, pending transfers
quote = bridge.quote("ethereum/wbtc", "aleo/wbtc", amount="0.001", recipient=bridge.aleo_address())
print(quote.fees, quote.amount_out)             # show these to the user BEFORE executing
progress = bridge.execute(quote.plan)           # source step; checkpoints saved to the bound store
progress = bridge.wait(progress)                # stops at resume / complete / done / failed
if progress.next == "resume":   progress = bridge.wait(bridge.resume(progress))
if progress.next == "complete": progress = bridge.wait(bridge.complete(progress, secret_nonce=nonce))
assert progress.next == "done", progress.error
```

### `from_env(**overrides: 'Any') -> "'Bridge'"`

Everything from the environment (spec §3.3); writes nothing to disk. Overrides: ethereum, solana, registry, checkpoints.

### `from_profile(home: 'Any' = None, *, network: 'str | None' = None, endpoint: 'str | None' = None, ethereum: 'Any' = None, solana: 'Any' = None) -> "'Bridge'"`

The client for the local profile (spec §3.4), created on first use. *network*/*endpoint* apply only when
creating. Side-chain connections come from the arguments or the same env variables as ``from_env``.

### `status(self) -> 'BridgeStatus'`

Read-only re-orientation: addresses and public balances of every registry asset per configured chain.

``pending`` is every in-flight transfer of the bound checkpoint store, reconstructed offline by
:meth:`pending` — no chain is read for it, and a record that cannot be interpreted comes back as a
``Progress`` with ``next == "failed"`` instead of hiding the others. It is empty when no store is
bound. Finish any entry with ``recover`` → ``wait`` / ``resume`` / ``complete``, never by starting
a new transfer.

### `quote(self, source, destination, *, amount=None, amount_atomic=None, recipient: 'str', sender: 'str | None' = None, protocol: 'str | None' = None, mint_mode: 'str' = 'public', secret_nonce: 'str' = '0scalar')`

Price a transfer and get the plan that ``execute`` takes. Nothing is signed.

``source`` / ``destination`` are ``"chain/key"`` strings or ``(chain, key)``
tuples (``"ethereum/usdc"``, ``"aleo/usdcx"``); give exactly one of
``amount`` (human units, str) or ``amount_atomic`` (int).  ``recipient`` is
the destination-chain address.  ``mint_mode`` (xReserve into Aleo only):
``"public"`` balance, ``"record"`` minted by the relayer, or ``"private"``
— you finish it yourself with ``complete`` and must keep ``secret_nonce``.
Returns a kind-specific ``Quote`` (``quote.kind`` in evm-hyperlane /
solana-hyperlane / aleo-hyperlane / evm-xreserve / aleo-xreserve) with
``fees`` and ``amount_out`` in human units and ``quote.plan``.  Show the
user fees + amount before ``execute``.

### `execute(self, plan, *, on_checkpoint=None, proving: 'str' = 'delegate', mode: 'str | None' = None, record: 'str | None' = None, merkle_proof: 'str | None' = None, gas_payment_microcredits: 'int | None' = None, secret_nonce: 'str | None' = None, poll_seconds: 'float' = 1.0, timeout_seconds: 'float' = 120.0)`

Commit funds on the source chain for ``quote.plan``; returns ``Progress``.

Runs approval(s) → deposit / dispatch / burn, emitting a ``Checkpoint`` to
``on_checkpoint`` (and the bound store) at every boundary — including
AFTER proving and BEFORE broadcast for Aleo legs, so a crash there is
resumable without proving twice.  ``proving`` is ``"delegate"`` (DPS) or
``"local"``; ``mode`` is ``"caller"|"signer"`` (Aleo Hyperlane) or
``"private"|"public"|"public-as-signer"`` (Aleo xReserve burn, default
private; ``record``/``merkle_proof`` optional — the SDK selects a record
and computes the exclusion proof).  The Hyperlane hook payment is
re-quoted right before proving unless ``gas_payment_microcredits`` is
pinned.  Irreversible once the source step is broadcast: afterwards use
``wait`` / ``recover``, never ``execute`` again.

### `wait(self, progress, *, until=None, poll_seconds: 'float' = 15.0, timeout_seconds: 'float' = 1200.0, on_update=None, on_error=None, max_consecutive_errors: 'int' = 5)`

Poll until the transfer finishes or needs you: stops at ``progress.next``
in resume / complete / done / failed, or at any status in ``until``.

A ``PollingTimeoutError`` is NOT a failure — the transfer is still in
flight; call ``wait`` again or ``recover`` later.  ``on_update`` receives
each changed ``Progress``.  A transient error (flaky RPC/HTTP transport)
is retried up to ``max_consecutive_errors`` times, calling ``on_error``
on each tolerated retry; a non-transient error propagates immediately.

### `recover(self, checkpoint)`

Rebuild ``Progress`` from a saved checkpoint (``Checkpoint``, dict or JSON) — reads only.

Re-resolves the route from the live registry and reads chain state once;
``progress.next`` then says what to do: ``wait``, ``resume``, ``complete``,
``done`` or ``failed``.

### `resume(self, progress, *, on_checkpoint=None, secret_nonce: 'str | None' = None, poll_seconds: 'float' = 1.0, timeout_seconds: 'float' = 120.0, proving: 'str' = 'delegate')`

Finish an interrupted source submission (``progress.next == "resume"``).

Rebroadcasts the identical proved Aleo transaction (a duplicate response is
success) or, on EVM, re-scans history and only then authorizes the single
missing deposit/dispatch.  Never repeats a confirmed step.

### `complete(self, progress, *, secret_nonce: 'str', on_checkpoint=None, proving: 'str' = 'delegate')`

Submit the private USDCx mint (``progress.next == "complete"``).

Requires the same ``secret_nonce`` given to ``execute``; the SDK never
stored it.  Submits exactly one ``private_mint`` and returns
``DESTINATION_CONFIRMING`` progress to ``wait`` on.

### `pending(self) -> 'list'`

The in-flight transfers of this profile — every checkpoint in the bound store,
reconstructed offline (:func:`lifecycle.progress_from_checkpoint`): no network read, so one
unreachable chain can never hide the others. A malformed checkpoint yields a ``Progress``
with ``next == "failed"`` and ``error`` set instead of raising; call ``wait()``/``recover()``
on any entry to refresh it against live chain state.

Nothing is ever dropped silently. A record this client cannot interpret at all — a route
that no longer exists, a registry version this build did not write — and a file the store
could not even read back come back as ``{"next": "failed", "error", "error_type"}`` entries
(naming the ``checkpoint_id`` or the ``path``) alongside the healthy ``Progress`` objects,
so a stale or corrupt file can never make a transfer that is still on the wire invisible.

## Serving a chatting user (the conversation pattern)

### Keys and identity

1. **NEVER ask the user to paste a private key into the conversation.**  Keys
   come from the environment only: `BRIDGE_PRIVATE_KEY` (Aleo),
   `EVM_PRIVATE_KEY` + `ETHEREUM_RPC_URL`, `SOLANA_PRIVATE_KEY` (+ optional
   `SOLANA_RPC_URL`), set in the user's own shell before the process starts.
   `Bridge.from_profile()` creates an Aleo key on first use and never writes
   EVM/Solana keys to disk.
2. `status()` first in any session: which chains are configured, balances of
   every bridge asset, and the pending transfers in the checkpoint store.  A
   pending transfer is finished with `recover` → `wait`/`resume`/`complete`,
   never by starting a new one.

### Quote first, always

3. **Always `quote` before `execute`** and show the user the route, the fees
   and `amount_out` in human units with symbols ("2 USDC → 2 USDCx; Hyperlane
   hook payment 8.17 ALEO"), never raw atomic units.  Minimums: xReserve
   needs at least 2 USDC in and strictly more than the 2 USDCx withdrawal fee
   out; Hyperlane moves one atomic unit but network fees and the relayer
   payment cost more than that — say so.
4. Only `execute` after the user confirms.  Through the agent tools every
   write requires `confirm=true`; without it the tool returns the quote and
   moves nothing.  A live mainnet execution additionally needs the user's
   own `BRIDGE_LIVE_MAINNET_EXECUTE` acknowledgement — never set it yourself;
   without it, treat any mainnet run as a rehearsal.

### The source step is irreversible

5. Once the deposit / dispatch / burn is broadcast the funds are committed.
   A timeout, an RPC error or a crash after that point is an UNKNOWN outcome,
   not a failure: recover from the last checkpoint (`recover(checkpoint)` or
   `pending()`) — never run `execute` again for the same transfer.  This is
   the funds-safety rule above all others: never resend after an ambiguous
   broadcast.

### What `progress.next` means for the user

| `progress.next` | Status | Tell the user | Do |
| --- | --- | --- | --- |
| `wait` | source confirming, attestation pending, delivery pending | "In flight; I'll keep checking." | `wait(progress)` (or re-check later from the checkpoint) |
| `resume` | `SOURCE_SUBMISSION_PENDING` | "An approval confirmed / a proof was built but the transfer itself was not submitted; I can submit it now." | confirm, then `resume(progress)` |
| `complete` | `DESTINATION_ACTION_REQUIRED` | "Circle attested your deposit; your private mint needs your signature (and the secret nonce)." | confirm, then `complete(progress, secret_nonce=...)` |
| `done` | `COMPLETED` | "Delivered." Report source and destination transaction ids. | nothing |
| `failed` | `FAILED` / `EXPIRED` | Relay `progress.error`; the source step did not commit funds or was rejected. | nothing — a new transfer needs a new quote |

`wait` raising `PollingTimeoutError` is NOT a failure — say the transfer is
still in flight and check again later.

### Private mints and the secret nonce

6. `mint_mode="private"` (USDC → USDCx) commits `(recipient, secret_nonce)` on
   Ethereum.  The same `secret_nonce` is required by `complete`; the SDK
   **never stores** it and checkpoints exclude it (and every other secret).
   Tell the user to keep it (the default `0scalar` needs no storage but adds
   no entropy).  Only the recipient's Aleo key can complete a private mint —
   make sure the recipient IS the configured Aleo address before depositing.
7. Aleo-origin Hyperlane transfers spend PUBLIC balances: `unshield` a private
   record first.  Hyperlane delivers into public balances; `shield` afterwards
   if the user wants privacy.  Private xReserve burns spend records directly.

### While acting

8. Writes are slow (proving + confirmation ≈ a minute or two on Aleo; Circle
   attestation and Hyperlane relay take minutes).  Never re-submit because a
   call seems slow — `status()` / `recover` first.
9. Confirm, act, report ids.  Errors name their own fix — read the exception
   message and do what it says.


## Tier 2 — the protocol modules (building your own flows)

Every Aleo write returns an `AleoCall`: nothing touches the network until
`.simulate()` (free), `.prove()` / `.delegate_prepared()` (proved, not
broadcast — checkpoint it), `.submit_prepared()`, `.transact()` (local
proving + broadcast) or `.delegate()` (DPS + broadcast).  EVM and Solana
writes return `EvmCall` / `SolCall` with `.build()` (unsigned) and `.send()`.
The lifecycle verbs above compose these; use them directly only when you
need a single leg.  Confirm-gated writes and the never-resend rule above
apply here too — these are the same broadcasts, just one leg at a time.

### `hyperlane.transfer_remote(self, asset: 'Any', recipient: 'str', *, amount: 'Any' = None, amount_atomic: 'int | None' = None, as_signer: 'bool' = False, gas_payment_microcredits: 'int | None' = None) -> 'AleoCall[DispatchReceipt]'`

Withdraw an Aleo warp asset to Ethereum/Solana. Quotes the IGP payment now unless pinned; the
lifecycle layer (plan 4) re-quotes at the last responsible moment by calling this again.

### `hyperlane.quote_gas_payment(self, asset: 'Any') -> 'GasQuote'`

Live relayer payment for the route (the exact u64 the hook asserts); quote right before proving.

### `xreserve.burn(self, recipient: 'str', *, amount: 'Any' = None, amount_atomic: 'int | None' = None, mode: 'str' = 'private', record: 'str | None' = None, merkle_proof: 'str | None' = None) -> 'AleoCall[BurnReceipt]'`

Burn USDCx for USDC on Ethereum. ``private`` (default) spends a Token record via the wrapper and needs a
freeze-list exclusion proof — both are resolved from chain state when not supplied. Minimum: more than
the 2 USDCx withdrawal fee. The Aleo burn-attestation service forwards accepted burns to Circle.

### `xreserve.private_mint(self, attestation: 'Attestation', recipient: 'str', *, secret_nonce: 'str' = '0scalar', route: 'Route | None' = None) -> 'AleoCall[MintReceipt]'`

Finish a private-mode deposit: the only user-signed Aleo step of the inbound flow (``wrapper.private_mint``).

### `xreserve.get_attestation(self, message_hash: "'str | bytes'", *, route: 'Route | None' = None) -> 'Attestation | None'`

One Circle request for *message_hash*; ``None`` while pending (404).

### `shield(self, asset: 'Any', *, amount: 'Any' = None, amount_atomic: 'int | None' = None, recipient: 'str | None' = None) -> 'AleoCall[PrivacyReceipt]'`



### `unshield(self, asset: 'Any', *, amount: 'Any' = None, amount_atomic: 'int | None' = None, record: 'str | None' = None, merkle_proof: 'str | None' = None, recipient: 'str | None' = None) -> 'AleoCall[PrivacyReceipt]'`



### `freezelist.exclusion_proof(self, address: 'str', program: 'str') -> 'str'`

``[MerkleProof; 2]`` proving *address* is not frozen on *program*; veil's empty pair when the list is empty.

### `eth.transfer_remote(self, asset: 'Any' = None, recipient: 'str | None' = None, *, amount: 'Any' = None, amount_atomic: 'int | None' = None, plan: 'Plan | None' = None) -> 'EvmCall[DispatchReceipt]'`

Send ETH, WBTC or USDT to Aleo through its Hyperlane Warp Route.

Re-quotes ``quoteTransferRemote`` at send time. Collateral routes approve exactly the
quoted token amount only when the allowance is short (USDT: a non-zero allowance is
reset to 0 first). Native ETH sends amount + fee as ``msg.value``; collateral routes
send the fee only. Each hash is checkpointed before polling; a timeout returns a
pending ``DispatchReceipt``. The message id comes from the Mailbox ``DispatchId`` log.

``plan=`` executes a plan prepared earlier (typically ``quote.plan``): the route is
re-resolved by id against the live registry, the sender must be the connected account, and
the plan must equal what this call would have prepared itself. Mutually exclusive with ``asset=``.

### `eth.deposit_usdc(self, recipient: 'str | None' = None, *, amount: 'Any' = None, amount_atomic: 'int | None' = None, mint_mode: 'str | None' = None, secret_nonce: 'str' = '0scalar', plan: 'Plan | None' = None) -> 'EvmCall[DepositReceipt]'`

Deposit USDC into Circle xReserve for USDCx on Aleo (minimum 2 USDC; irreversible once confirmed).

``mint_mode``: ``public`` (public USDCx balance), ``record`` (protocol-minted private
record), or ``private`` (deposit addressed to the shielded wrapper program; you must later
run ``bridge.xreserve.private_mint`` / plan 4's ``complete`` with the same ``secret_nonce``,
which the SDK never stores). Approves exactly the amount only when the allowance is
short, then ``depositToRemote`` with no ``msg.value``. The confirmed ``DepositReceipt``
carries Circle's message hash (receipt id) and the deposit nonce. ``mint_mode`` defaults to
``plan.mint_mode`` when a plan is given, else ``"public"``.

``plan=`` executes a plan prepared earlier (typically ``quote.plan``): the route is
re-resolved by id against the live registry, the sender must be the connected account, and
the plan must equal what this call would have prepared itself. ``secret_nonce`` is never
part of a plan, so a private deposit must still pass the same one it was quoted with.

### `eth.quote_transfer_remote(self, asset: 'Any' = None, recipient: 'str | None' = None, *, amount: 'Any' = None, amount_atomic: 'int | None' = None, route: 'Route | None' = None, sender: 'str | None' = None, plan: 'Plan | None' = None) -> 'EvmHyperlaneQuote'`

Quote an Ethereum → Aleo Hyperlane transfer without signing.

Native routes (ETH): ``msg.value`` carries the asset and the relayer fee, so
``native_fee_atomic = native_value_atomic - amount``. Collateral routes (WBTC, USDT):
``msg.value`` is fee only and ``approval_required`` reflects the router's ERC-20
allowance for ``sender`` (or the connection's account); it is ``None`` when no account is known.

``plan=`` re-quotes a plan prepared earlier: it supplies the route, sender, recipient and
amount, and is validated against the live registry. It is mutually exclusive with
``asset=``/``route=``/``sender=``.

### `sol.transfer_remote(self, recipient: 'str | None' = None, *, amount: 'str | None' = None, amount_atomic: 'int | None' = None, plan: 'Plan | None' = None) -> 'SolCall[DispatchReceipt]'`

Send native SOL to an Aleo address over the Hyperlane warp route (spec §6).

Returns a :class:`SolCall`: ``build()`` previews the partially signed transaction,
``send()`` moves funds (amount + IGP payment + network fee + rent leave the wallet).

``plan`` (from ``Bridge.execute``) supplies recipient and amount and must have been prepared for
the connected wallet; its registry version and route id are re-checked against the live registry
when the call runs. An ``amount``/``amount_atomic`` that disagrees with the plan is a
``ValueError``. Without a plan, ``recipient`` is required.

### `sol.quote_transfer_remote(self, recipient: 'str | None' = None, *, amount: 'str | None' = None, amount_atomic: 'int | None' = None, sender: 'str | None' = None, plan: 'Plan | None' = None) -> 'SolanaHyperlaneQuote'`

Lamports required for a SOL → Aleo transfer: amount + IGP payment + network fee + rent (spec §5 kind
``solana-hyperlane``). Reads Solana; never signs. ``sender`` defaults to the connected wallet and is required
for the fee estimate.

``plan`` (from ``Bridge.quote``) supplies recipient, amount and sender, and must match the live registry
version and route; like ``EthModule`` it is mutually exclusive with ``sender=``, and an ``amount``/
``amount_atomic`` that disagrees with the plan is a ``ValueError`` (an identical one is tolerated, so
re-stating the plan's own amount is harmless). Without a plan, ``recipient`` is required.

### Routes in the pinned registry

| Route id | Protocol | Environment | Availability |
| --- | --- | --- | --- |
| `xreserve:ethereum/usdc->aleo/usdcx` | xreserve | mainnet | active |
| `xreserve:aleo/usdcx->ethereum/usdc` | xreserve | mainnet | active |
| `xreserve:sepolia/usdc->aleo-testnet/usdcx` | xreserve | testnet | active |
| `xreserve:aleo-testnet/usdcx->sepolia/usdc` | xreserve | testnet | active |
| `hyperlane:ethereum/eth->aleo/eth` | hyperlane | mainnet | active |
| `hyperlane:aleo/eth->ethereum/eth` | hyperlane | mainnet | active |
| `hyperlane:ethereum/wbtc->aleo/wbtc` | hyperlane | mainnet | active |
| `hyperlane:aleo/wbtc->ethereum/wbtc` | hyperlane | mainnet | active |
| `hyperlane:ethereum/usdt->aleo/usdt` | hyperlane | mainnet | active |
| `hyperlane:aleo/usdt->ethereum/usdt` | hyperlane | mainnet | active |
| `hyperlane:solana/sol->aleo/sol` | hyperlane | mainnet | active |
| `hyperlane:aleo/sol->solana/sol` | hyperlane | mainnet | active |
| `hyperlane:aleo/aleo->ethereum/aleo` | hyperlane | mainnet | metadata-required |
| `hyperlane:ethereum/aleo->aleo/aleo` | hyperlane | mainnet | metadata-required |
| `hyperlane:aleo/aleo->solana/aleo` | hyperlane | mainnet | metadata-required |
| `hyperlane:solana/aleo->aleo/aleo` | hyperlane | mainnet | metadata-required |
| `hyperlane:aleo/aleo->base/aleo` | hyperlane | mainnet | metadata-required |
| `hyperlane:base/aleo->aleo/aleo` | hyperlane | mainnet | metadata-required |
| `hyperlane:aleo/aleo->hyperevm/aleo` | hyperlane | mainnet | metadata-required |
| `hyperlane:hyperevm/aleo->aleo/aleo` | hyperlane | mainnet | metadata-required |
| `hyperlane:ethereum/usad->aleo/usad` | hyperlane | mainnet | metadata-required |
| `hyperlane:aleo/usad->ethereum/usad` | hyperlane | mainnet | metadata-required |

`metadata-required` routes are listed but refused by `quote`/`execute`
until their deployments are reviewed upstream.

