# shield-swap — agent guide

> GENERATED from SDK docstrings by `codegen/gen_context.py` — do not
> edit by hand; edit the docstrings and regenerate.

Typed Python client for the shield_swap AMM on Aleo
(`pip install shield-swap-sdk`, imports as `aleo_shield_swap`).
MCP alternative: `python -m aleo_shield_swap.mcp` exposes the same
lifecycle as tools.

## Tier 1 — the lifecycle (six lines end to end)

```python
from aleo_shield_swap import ShieldSwap

dex = ShieldSwap.from_profile()          # key material auto-managed on disk
dex.onboard(invite_code="...")           # first run only; no-op afterwards
pools = dex.api.get_pools()
report = dex.swap_many(pool_key=pools[0].key, token_in_id=pools[0].token0,
                       amount_in=10**6, count=5)
dex.collect_all()                        # any session, any time
```

### `from_profile(home: 'Any' = None) -> "'ShieldSwap'"`

The client for the local participant profile (created on first use).

Wires endpoint, network, signer, and (when present) delegated-proving
credentials from ``$SHIELD_SWAP_HOME``/``~/.shield-swap``.  Run
``onboard()`` next on a fresh profile.

### `onboard(self, invite_code: 'Optional[str]' = None) -> 'OnboardReport'`

Register this profile end to end — safe to re-run any time.

Runs only the registration stages not already satisfied (see
``lifecycle.REGISTRATION_STAGES``); a registered, funded account is
a no-op.  The one thing it may need from you: *invite_code*, on the
first run.  Requires a profile-bound client (``from_profile()``).

### `status(self) -> 'SessionStatus'`

One re-orientation call: identity, access, holdings, pending work.

Run this first in any session — it answers "is this account already
registered, what do I hold, what is in flight" from the profile,
journal, chain, and API without changing anything.

### `get_positions(self, account: 'Any' = None) -> 'list[PositionView]'`

Every open position — journaled ones plus a record scan.

The scan catches positions the journal never saw (account used from
another machine, journal lost); it needs a registered record
provider and is skipped silently without one.

### `swap_many(self, *, pool_key: 'str', token_in_id: 'str', amount_in: 'int', count: 'int', slippage_bps: 'int' = 50, record_wait_seconds: 'float' = 120.0, account: 'Any' = None) -> 'SwapBatchReport'`

*count* private swaps of *amount_in* each, with reserved counters.

Counters come from the journal (no probe races); every handle is
journaled as soon as its broadcast is accepted (no confirmation
wait), so a crash mid-batch loses nothing — ``collect_all()`` later
claims whatever finalized.  A swap the network rejects simply never
becomes claimable (it stays in ``still_pending``).  A failed
broadcast burns its counter and the batch continues; failures are
reported, not raised.  Requires ``from_profile()``.

### `collect_all(self, account: 'Any' = None) -> 'CollectReport'`

Claim every finalized swap and collect owed fees on open positions.

Safe to run any time, from any session: works off the journal, skips
swaps whose finalize hasn't landed (they stay pending for next time),
never double-claims, and requests exactly the owed amounts the chain
reports.  Requires ``from_profile()``.

## Serving a chatting user (the conversation pattern)

### Before doing anything: two questions

1. **Existing account?**  When `status()` shows a brand-new, unregistered
   profile, ask whether the user already has a shield-swap account before
   creating anything — their funds and access live on the old key.
   **NEVER ask the user to paste a private key into the conversation.**
   They supply it out-of-band: `export SHIELD_SWAP_PRIVATE_KEY=...` (or
   `SHIELD_SWAP_PRIVATE_KEY_FILE=path`) in their own shell before the
   profile is first created.
2. **Invite code.**  Access is invite-gated per account; `onboard()` stops
   with `NotRedeemedError` until one is supplied.  Ask the user for their
   code; codes are one-time — never guess or reuse.

### After startup: ask what's next

When onboarding reports funded, STOP and ask the user what they want to do
— never launch into a journey unprompted.  Present the options WITH their
context, in plain language (identities, records, and journals are your
business, not the user's):

1. **Their own playbook.**  Ask whether they have instructions of their
   own — a strategy file, notes, a memory store, output from a previous
   session.  If so, read it and treat it as the plan: their document
   decides what to do, the verbs here describe how each step works.

2. **A suggested journey.**  Frame the setting first — Shield Swap is a
   private exchange on Aleo's test network: trading uses test tokens, and
   what is traded, and by whom, stays hidden on the public chain — then
   offer:

   - *Swap tokens* — trade one token for another.  It settles in two
     steps — placing the trade, then collecting what was bought — and you
     do both, so the proceeds arrive without a separate trip.  The
     natural first move.
   - *Several swaps at once* — place a handful of trades and watch them
     all land (`swap_many`); the busiest way to exercise the exchange.
     First show which trades are possible right now (tokens held x live
     pools) and ask how many — and which — they want.
   - *Open a liquidity position* — instead of trading, become the market:
     deposit a pair of tokens so others can trade against them (`mint`).
     The user picks the price range; while the market price sits inside
     it they earn a cut of every trade passing through.
   - *Add or remove liquidity* — grow a position or take some back out
     (`increase_liquidity`/`decrease_liquidity`); whatever comes out
     becomes earnings to collect.
   - *Collect earnings* — sweep everything the account is owed (tokens
     bought in earlier swaps, fees its liquidity earned) into the wallet
     (`collect_all`); good after any trading session.

3. **Developing a trading application or agent?**  Ask whether they are
   building on Shield Swap — a dApp, a trading bot, a server or agent
   integration — rather than (or besides) trading here.  The chat
   journeys above are one way to use the DEX; consumers also build on
   the SDK directly — route builders to Tier 2 below, which opens with
   the client-choice table (bot/server, agent integration, browser dApp)
   and the integration checklist.

4. **A free-form prompt.**  Whatever they describe, map it onto the
   verbs and journeys above before improvising against the SDK.

### While acting

1. `status()` first in any session — never onboard an account that is
   already set up; state lives in `~/.shield-swap/`, not in your context.
2. Ground every proposal in ACTUAL holdings crossed with
   `dex.api.get_pools()`; recommend tick ranges from `get_slot(pool_key)`
   (see `SlotView.tick_range`) instead of waiting for exact parameters.
3. **Never show raw base units to the user** — render amounts in human
   units with the symbol via the token registry's `decimals`
   ("0.0534 ETH", never "53,369,000,000,000 raw").
4. Writes are slow (delegated proving + confirmation ≈ a minute or two).
   Never re-submit because a call seems slow — check `status()` first.
5. Confirm, act, report ids.  Errors name their own fix — read the
   exception message and do what it says.

## Tier 2 — the development guide (building your own tools)

### Building a trading application or agent?

Start by asking what they are building — the client choice follows from
where the signing keys live:

| Building | Stack | Keys live |
| --- | --- | --- |
| Bot / server / CLI / notebook | `aleo-sdk` facade + `shield-swap-sdk` (this package) | A local private key (`ShieldSwap.from_profile()` manages it); delegated proving through the Provable prover — fees covered by default. |
| Agent integration | `aleo_shield_swap.agent` (Claude-shape tool schemas + `dispatch_tool`) or `python -m aleo_shield_swap.mcp` (MCP server), over the same client | Same as the underlying client; the tools bind to it. |
| Browser dApp (wallet-signed) | The TypeScript stack: `@provablehq/shield-swap-sdk` + Veil react hooks — not this package | The user's wallet signs and proves. |

What every integration must handle (each enforced or automated by the
verbs above — this list is the review checklist for code that bypasses
them):

- **Auth is layered**: a bearer credential (24h session JWT from the
  challenge/verify handshake, or a durable `ss_…` API token — data/trading
  endpoints only) AND a one-time invite redemption per account.
- **Dynamic-dispatch imports**: every record-spending write must register
  the involved token programs with the prover (the verbs resolve this via
  the token registry; pass `imports=`/`token_*_program=` to override).
- **Tokens are private records**: spendable balances do not appear in
  public reads; one covering record funds an amount — no aggregation.
- **Amounts obey the no-dust rule** both directions (`amount % scale == 0`);
  quote in canonical decimals, transact in raw base units, display human.
- **A `SwapHandle` is the only key to a swap's output** — persist before
  anything else (the journal does this); claim after finalize with retry.
- **Concurrency needs partitioned blinded-identity counters AND disjoint
  input records** — `swap_many` implements the recipe; copy it, don't
  improvise.

Suggested path for a new integrator: (1) `onboard()` a profile — it
doubles as a test fixture; (2) walk swap → `collect_all()` once with the
Tier 1 verbs so the mechanics are concrete; (3) read the reference below
for the surface your app needs; (4) `tests/integration/` and
`scripts/rehearsal.py` in the repo are working reference implementations
of the full journey.

Every write verb returns a prepared `DexCall`: nothing touches the
network until a terminal verb — `.simulate()` (local, free),
`.transact()` (local proving, slow), or `.delegate()` (delegated
proving — the practical path).

### Registration, unbundled

`onboard()` is a stage list, and the steps WILL change over time —
introspect `lifecycle.REGISTRATION_STAGES`, never hard-code the
sequence.  Current stages:

- `authenticate`
- `redeem`
- `credentials`
- `airdrop`
- `funded`

Apps that own their onboarding call the same `dex.api` endpoints
the stages use:

### `api.authenticate(self, address: 'str', sign: 'Any') -> 'str'`

Challenge/verify handshake; stores and returns the JWT.

*sign* is a callable taking the challenge message string and
returning an Aleo signature literal (``sign1…``) — e.g.::

    pk = aleo.testnet.PrivateKey.from_string(key)
    api.authenticate(str(pk.address),
                     lambda msg: str(pk.sign(msg.encode())))

### `api.access_status(self) -> 'models.AccessStatusResponse'`

Whether this authenticated account has redeemed an invite code.

### `api.redeem_code(self, code: 'str') -> 'models.AccessRedeemResponse'`

Redeem an invite code; adopts the fresh token the API returns.

### `api.request_airdrop(self, address: 'str') -> 'models.AirdropStartResult'`

Start the test-token airdrop job for *address* (private records).

One claim per address per 15 minutes — raises
:class:`AirdropRateLimitedError` on 429.  Poll the returned
``job_id`` with :meth:`get_airdrop_job`.

### `api.get_airdrop_job(self, job_id: 'str') -> 'models.AirdropJob'`

Progress of an airdrop job — ``running`` until every transfer lands.

### `api.create_api_token(self, name: 'str', expires_in_days: "'int | None'" = None) -> 'models.ApiTokenCreatedResponse'`

Mint a long-lived DEX API token (the secret is returned ONCE).

JWTs from :meth:`authenticate` expire in 24h; persist the returned
``.token`` for durable access.  Tiering (verified live): ``ss_…``
tokens work on data/trading endpoints; ``/access/*`` and token
management still require a session JWT.

### `api.get_pools(self) -> 'list[PoolEntry]'`



### `api.get_tokens(self) -> 'list[models.TokenDoc]'`



### `api.get_route(self, *, token_in: 'str', token_out: 'str', amount_in: 'Any' = None) -> 'models.RouteResultDoc'`

Best route between two tokens.  *amount_in* is a CANONICAL
decimal amount (human units, e.g. ``1.5``) — not base units —
and the returned ``estimated_amount_out`` is decimal too.

### Counters & blinding

Blinded identities derive deterministically from (view key, counter,
program).  Counters must NEVER be reused: reserve them via
`dex.journal.reserve_counters(n)` (what `swap_many` does), or probe
on-chain when no journal exists.  Persist `SwapHandle`s — the
blinding factor is the claim secret.

### `blinded_identity_at(aleo: 'Any', account: 'Any', program: 'str', counter: 'int') -> 'BlindedIdentity'`

The identity at an exact *counter* — no on-chain probing.

Use with journal-reserved counters for concurrent swaps;
:func:`next_blinded_identity` (probe-based) remains the recovery path
when no journal exists.

### `next_blinded_identity(aleo: 'Any', account: 'Any', program: 'str' = 'shield_swap_v3.aleo', *, start_counter: 'int' = 0, max_scan: 'int' = 64) -> 'BlindedIdentity'`

First unused single-use identity for *account*.

Derives at ``start_counter, +1, …`` and probes the program's
``used_blinded_addresses`` mapping until one is free.  ``max_scan`` fails
fast when something is systematically wrong (e.g. wrong program).

### Chain verbs

### `swap(self, *, pool_key: 'str', token_in_id: 'str', amount_in: 'int', slippage_bps: 'int' = 50, expected_out: 'Optional[int]' = None, sqrt_price_limit: 'Optional[int]' = None, deadline_offset_blocks: 'int' = 10000, nonce: 'Optional[int]' = None, identity: 'Optional[BlindedIdentity]' = None, token_in_program: 'Optional[str]' = None, token_record: 'Optional[str]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[SwapHandle]'`

Request a private swap — phase one of the two-transaction flow.

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

### `claim_swap_output(self, handle: 'SwapHandle', *, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[ClaimResult]'`

Claim a private swap's output — phase two of the lifecycle.

Reads the chain-computed result from ``swap_outputs`` (never an
off-chain service — these amounts gate money movement), proves
ownership of the blinded identity, and prepares ``claim_swap_output``.
The output and any refund arrive as private records owned by the
signer; the mapping entry is consumed.

Raises :class:`SwapOutputNotFinalizedError` **at prepare time** when
the output is not readable yet (retry after a few blocks) or was
already claimed.

### `create_pool(self, *, token0_id: 'str', token1_id: 'str', fee: 'int', initial_tick: 'int', tick_spacing: 'Optional[int]' = None, initial_sqrt_price: 'Optional[int]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[TxResult]'`

Create a pool — a single public transaction, no records involved.

The fee tier must be registered with the program (validated before
submission); tick spacing defaults to the tier's on-chain binding and
the opening price to the tick's sqrt price.

### `mint(self, *, pool_key: 'str', tick_lower: 'int', tick_upper: 'int', amount0_desired: 'int', amount1_desired: 'int', amount0_min: 'int' = 0, amount1_min: 'int' = 0, token0_program: 'Optional[str]' = None, token1_program: 'Optional[str]' = None, token0_record: 'Optional[str]' = None, token1_record: 'Optional[str]' = None, tick_lower_hint: 'Optional[int]' = None, tick_upper_hint: 'Optional[int]' = None, recipient: 'Optional[str]' = None, nonce: 'Optional[str]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[MintResult]'`

Mint a concentrated-liquidity position as a private PositionNFT.

Tick bounds are rounded to the pool's spacing; insert hints derive
from the slot's neighbors unless given explicitly.

### `increase_liquidity(self, *, pool_key: 'str', amount0_desired: 'int', amount1_desired: 'int', amount0_min: 'int' = 0, amount1_min: 'int' = 0, token0_program: 'Optional[str]' = None, token1_program: 'Optional[str]' = None, token0_record: 'Optional[str]' = None, token1_record: 'Optional[str]' = None, position_record: 'Optional[str]' = None, tick_lower_hint: 'Optional[int]' = None, tick_upper_hint: 'Optional[int]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[TxResult]'`

Add funds to an existing position (range fixed at mint).

### `decrease_liquidity(self, *, pool_key: 'str', liquidity_to_remove: 'int', amount0_min: 'int' = 0, amount1_min: 'int' = 0, position_record: 'Optional[str]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[TxResult]'`

Remove liquidity from a position; owed amounts become collectable.

### `collect(self, *, pool_key: 'str', amount0_requested: 'int', amount1_requested: 'int', recipient: 'Optional[str]' = None, position_record: 'Optional[str]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[TxResult]'`

Collect owed token amounts from a position.

### `burn(self, *, pool_key: 'str', position_record: 'Optional[str]' = None, account: 'Any' = None) -> 'DexCall[TxResult]'`

Burn an empty position NFT.

### `get_pool(self, pool_key: 'str') -> 'g.PoolState'`

Static pool configuration (token pair, fee, decimal scales).

### `get_slot(self, pool_key: 'str') -> 'SlotView'`

Live trading state (sqrt price, tick, in-range liquidity).

Raises :class:`PoolNotFoundError` when the pool does not exist, or
:class:`PoolNotInitializedError` when it exists but has no slot yet.

### `get_swap_output(self, swap: "'SwapHandle | str'") -> 'g.SwapOutput'`

Chain-computed output of a finalized swap request.

Accepts the :class:`SwapHandle` from ``swap()`` or a bare swap id.
Raises :class:`SwapOutputNotFinalizedError` when the entry is absent —
not finalized yet (retry after a few blocks) or already claimed.

### `get_balances(self, address: 'Optional[str]' = None, account: 'Any' = None) -> 'dict[str, dict[str, Any]]'`

Public + private + total per token id, joined via the API's
token registry.  Defaults to the bound account's address; returns
only tokens actually held.

Private balances can only be scanned for the bound account's view
key — when *address* names someone else, ``private`` is 0 for every
token (their records are not scannable) rather than silently mixing
in the caller's own private holdings.

### `get_private_balances(self, programs: 'list[str]', account: 'Any' = None) -> 'dict[str, int]'`

Sum of unspent record amounts per wrapper program (spendable
privately).  Requires a configured record provider.

### `derive_pool_key(self, token0: 'str', token1: 'str', fee: 'int') -> 'str'`



### `derive_tick_key(self, pool_key: 'str', tick: 'int') -> 'str'`



