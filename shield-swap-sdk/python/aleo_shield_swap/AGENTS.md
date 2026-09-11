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
dex.onboard()                            # first run registers; no-op afterwards
pools = dex.api.get_pools()
report = dex.swap_many(pool_key=pools[0].key, token_in_id=pools[0].token0,
                       amount_in=10**6, count=5)
dex.collect_all()                        # any session, any time
```

### `from_profile(home: 'Any' = None, *, network: 'Optional[str]' = None, endpoint: 'Optional[str]' = None) -> "'ShieldSwap'"`

The client for the local participant profile (created on first use).

Wires endpoint, network, signer, and (when present) delegated-proving
credentials from ``$SHIELD_SWAP_HOME``/``~/.shield-swap``.  Run
``onboard()`` next on a fresh profile.

*network* and *endpoint* apply only when the profile is being created —
an existing one keeps what it was created with, because its derived pool
keys and blinded identities are network-scoped and would not transfer.
Give each network its own home directory.

Args:
    home: Profile directory; defaults to ``$SHIELD_SWAP_HOME`` or
        ``~/.shield-swap``.
    network: ``"mainnet"`` or ``"testnet"`` for a NEW profile; defaults
        to testnet.
    endpoint: Node API origin for a NEW profile.

### `onboard(self, referral_code: 'Optional[str]' = None) -> 'OnboardReport'`

Register this profile end to end — safe to re-run any time.

Runs only the registration stages not already satisfied (see
``lifecycle.REGISTRATION_STAGES``); a registered, funded account is
a no-op.  Needs nothing from you: access is granted by
authentication alone.  *referral_code* is optional — pass one a
friend shared to credit them as the referrer (recorded once; later
calls ignore it).  Requires a profile-bound client
(``from_profile()``).

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

### `swap_many(self, *, pool_key: 'str', token_in_id: 'str', amount_in: 'int', count: 'int', slippage_bps: 'int' = 50, expected_out: 'Optional[int]' = None, record_wait_seconds: 'float' = 120.0, account: 'Any' = None) -> 'SwapBatchReport'`

*count* private swaps of *amount_in* each, with reserved counters.

Counters come from the journal (no probe races); every handle is
journaled as soon as its broadcast is accepted (no confirmation
wait), so a crash mid-batch loses nothing — ``collect_all()`` later
claims whatever finalized.  A swap the network rejects simply never
becomes claimable (it stays in ``still_pending``).  A failed
broadcast burns its counter and the batch continues; failures are
reported, not raised.  Requires ``from_profile()``.

*expected_out* (base units) skips the route quote.  Without it the batch
quotes once and refuses rather than falling back to a spot estimate,
which ignores the pool fee and would revert every swap after paying for
its proof.

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
2. **Referral code — optional.**  Access is granted by authentication
   alone; `onboard()` needs nothing from the user.  Mention that a
   referral code from a friend can be passed (`onboard(referral_code=)`)
   to credit them, then proceed whether or not one is offered.  Never
   block on it, never guess one.  The account gets its own code to share
   (`dex.api.my_referral_code()`).

### After startup: ask what's next

When onboarding reports funded, STOP and ask the user what they want to do
— never launch into a journey unprompted.  Present the options WITH their
context, in plain language (identities, records, and journals are your
business, not the user's):

1. **Their own playbook.**  Ask whether they have instructions of their
   own — a strategy file, notes, a memory store, output from a previous
   session.  If so, read it and treat it as the plan: their document
   decides what to do, the methods here describe how each step works.

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
   methods and journeys above before improvising against the SDK.

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
methods above — this list is the review checklist for code that bypasses
them):

- **Auth is by signature, and it is the whole gate**: a session from the
  challenge/verify handshake (cookie + CSRF, or a legacy JWT), or a durable
  `ss_…` API token for data/trading endpoints.  No code is required; a
  referral code is optional attribution.
- **Dynamic-dispatch imports**: every record-spending write must register
  the involved token programs with the prover (the methods resolve this via
  the token registry; pass `imports=`/`token_*_program=` to override).
- **Tokens are private records**: spendable balances do not appear in
  public reads; one covering record funds an amount — no aggregation.
- **Amounts are raw native units end to end** (the AMM does no decimal
  scaling); quote in canonical decimals, transact in raw base units,
  display human.
- **Wrapped assets route automatically** (swap/claim/LP methods dispatch to
  the routers per token shape) — fund them with UNDERLYING records; never
  hand wrapper records around.
- **A `SwapHandle` is the only key to a swap's output** — persist before
  anything else (the journal does this); claim after finalize with retry.
  A swap with nothing left over claims through the no-refund entrypoints
  automatically; `get_swap_execution` reads the per-hop fill receipt (fees
  paid, price after) at any later time.
- **Rebalancing is one transaction, testnet only for now** —
  `plan_rebalance` quotes the close-and-remint (what comes back, what the
  new range needs, funding vs refund per token) and `rebalance_position`
  submits it through `shield_swap_rebalance_router.aleo`.  Every amount is
  asserted at the execution price: a trade in between reverts the whole
  transaction (fee paid, no funds moved) — re-plan and resubmit.
- **Concurrency needs partitioned blinded-identity counters AND disjoint
  input records** — `swap_many` implements the recipe; copy it, don't
  improvise.

Suggested path for a new integrator: (1) `onboard()` a profile — it
doubles as a test fixture; (2) walk swap → `collect_all()` once with the
Tier 1 methods so the mechanics are concrete; (3) read the reference below
for the surface your app needs; (4) `tests/integration/` and
`scripts/rehearsal.py` in the repo are working reference implementations
of the full journey.

Every write method returns a prepared `DexCall`: nothing touches the
network until a terminal method — `.simulate()` (local, free),
`.transact()` (local proving, slow), or `.delegate()` (delegated
proving — the practical path).

### Registration, unbundled

`onboard()` is a stage list, and the steps WILL change over time —
introspect `lifecycle.REGISTRATION_STAGES`, never hard-code the
sequence.  Current stages:

- `authenticate`
- `referral`
- `credentials`
- `airdrop`
- `funded`

Apps that own their onboarding call the same `dex.api` endpoints
the stages use:

### `api.authenticate(self, address: 'str', sign: 'Any') -> 'str'`

Challenge/verify handshake; establishes the session.

The staging API issues the session as httpOnly cookies on this
client's HTTP session plus a CSRF token (stored and echoed as
``X-CSRF-Token``); older deployments return a bearer JWT in the
body — both are handled.  Returns the stored credential (CSRF token
or JWT).  Sessions are short-lived — mint a durable ``ss_…`` token
via :meth:`create_api_token` for anything long-running.

*sign* is a callable taking the challenge message string and
returning an Aleo signature literal (``sign1…``) — e.g.::

    pk = aleo.testnet.PrivateKey.from_string(key)
    api.authenticate(str(pk.address),
                     lambda msg: str(pk.sign(msg.encode())))

### `api.referral_status(self) -> 'models.ReferralStatusResponse'`

This account's referral picture: ``referred_by`` (the referrer's
address once a code was redeemed, else None), ``my_code`` (the code
this account shares), and ``has_access``.  Network read.

``has_access`` is always true for an authenticated account — access
is granted by authentication alone, no code required — and the call
raises :class:`NotAuthenticatedError` when the session is missing or
expired, which makes this the session liveness probe (the dedicated
``/access/status`` route was retired in 2026-09).

### `api.my_referral_code(self) -> 'Optional[str]'`

The referral code this account hands to others.

The API issues the code on the first request, so this normally
returns a value; None means issuance is disabled for the account.
Network read.

### `api.redeem_code(self, code: 'str') -> 'models.ReferralRedeemResponse'`

Redeem a referral code (``POST /referral/redeem``) — optional.

Access does not depend on this: authentication alone unlocks every
endpoint.  Redeeming records who referred the account, once: a
repeat returns ``status="already_redeemed"`` without changing
anything, while an unknown code or the account's own code is a 400
(:class:`DexApiError`).

Sessions live on the ``/auth/*`` endpoints, so no token comes
back (one is still adopted if the API resurrects the legacy
body-JWT).

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

Every pool the DEX lists, each with its two tokens' metadata.

An entry exposes the pool's own fields directly — ``entry.key`` is the
``pool_key`` that ``swap``, ``mint``, and ``collect`` take.  Its
``token0_info`` / ``token1_info`` carry that token's ``symbol`` and
``decimals``, but the API does not guarantee them — check for ``None``
before reading.

### `api.get_tokens(self) -> 'list[models.TokenDoc]'`

Every token the DEX lists, with its id, symbol, and decimals.

``decimals`` converts between the two amount conventions.
The API returns canonical decimal amounts (``"1.5"``), if using this
value to call on-chain methods — ``swap(amount_in=…)``, ``mint``,
``collect`` — conversion to raw base units is necessary.

### `api.get_route(self, *, token_in: 'str', token_out: 'str', amount_in: 'Any' = None, pool_key: 'Optional[str]' = None) -> 'models.RouteResultDoc'`

Best route between two tokens.  *amount_in* is a CANONICAL
decimal amount (human units, e.g. ``1.5``) — not base units —
and the returned ``estimated_amount_out`` is decimal too.  *pool_key*
pins the quote to one pool instead of the router's best path.

### `api.get_route_topology(self) -> 'models.RouteTopologyDoc'`

The routable token graph: every (token0, token1) edge with an
enabled pool and the router's ``max_hops``.  Lets a client enumerate
reachable pairs without probing ``/route`` per pair.

### `api.get_unclaimed(self) -> 'models.UnclaimedPayloadDoc'`

Everything the authenticated account can still collect, as the
indexer sees it: swaps with finalized-but-unclaimed output and
positions with owed balances.  A cross-check for a local journal —
the chain, not this, gates the claim amounts.

### `api.get_protocol_state(self, *, minimum_revision: 'Optional[int]' = None) -> 'models.ProtocolStateResponse'`

The indexer's view of protocol configuration and its own freshness.

``freshness.ready_for_quote`` says whether quotes reflect the chain
head; ``revision`` increments on every config change and is echoed by
``/route`` as ``protocol_revision`` — pass *minimum_revision* to wait
for the indexer to reach one.  Returned unwrapped (no ``data``).

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

### `next_blinded_identity(aleo: 'Any', account: 'Any', program: 'str' = 'shield_swap.aleo', *, start_counter: 'int' = 0, max_scan: 'int' = 64, gallop: 'bool' = True) -> 'BlindedIdentity'`

An unused single-use identity for *account*.

Derives at ``start_counter, +1, …`` and probes the program's
``used_blinded_addresses`` mapping until one is free.  When the whole
linear window is used — an account that has swapped more than *max_scan*
times without a journal — *gallop* extends the search in O(log n) probes:
double the stride past the window until a free counter appears, then
bisect back to the lowest free one in that span.  Any free counter is a
valid identity (a gap left by a failed swap is fine), so the search only
needs SOME unused address, not the exact end of the used run.

With ``gallop=False`` the linear window is the whole search and
exhausting it raises — the fail-fast for a systematically wrong program.

### Chain methods

### `swap(self, *, pool_key: 'str', token_in_id: 'str', amount_in: 'int', slippage_bps: 'int' = 50, expected_out: 'Optional[int]' = None, sqrt_price_limit: 'Optional[int]' = None, deadline_offset_blocks: 'int' = 10000, nonce: 'Optional[int]' = None, identity: 'Optional[BlindedIdentity]' = None, token_in_program: 'Optional[str]' = None, token_record: 'Optional[str]' = None, wrapper_proofs: 'Optional[str]' = None, track: 'bool' = True, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[SwapHandle]'`

Request a private swap — phase one of the two-transaction flow.

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
**Building is not free with a journal.**  The blinded address is a
transition input, so a counter is reserved *here*, not at the terminal
method — discarding the call, or only simulating, still spends it.  That
reservation is what makes concurrent swaps safe: it serializes under a
file lock where the probe it replaces could hand two callers the same
counter.  The handle is journaled once the broadcast is accepted, so a
crash before the claim keeps the blinding factor.  ``track=False`` builds
on the racing probe instead; *identity* supplies your own.

The default
*deadline_offset_blocks* (~8h at ~3s blocks) absorbs delegated-
proving latency; a tight deadline aborts at finalize when proving
outlives it.

### `claim_swap_output(self, handle: 'SwapHandle', *, wrapper_proofs: 'Optional[str]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[ClaimResult]'`

Claim a private swap's output — phase two of the lifecycle.

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

### `create_pool(self, *, token0_id: 'str', token1_id: 'str', fee: 'int', initial_tick: 'int', tick_spacing: 'Optional[int]' = None, initial_sqrt_price: 'Optional[int]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[TxResult]'`

Create a pool — a single public transaction, no records involved.

The fee tier must be registered with the program (validated before
submission); tick spacing defaults to the tier's on-chain binding and
the opening price to the tick's sqrt price.

### `mint(self, *, pool_key: 'str', tick_lower: 'int', tick_upper: 'int', amount0_desired: 'int', amount1_desired: 'int', amount0_min: 'int' = 0, amount1_min: 'int' = 0, token0_program: 'Optional[str]' = None, token1_program: 'Optional[str]' = None, token0_record: 'Optional[str]' = None, token1_record: 'Optional[str]' = None, tick_lower_hint: 'Optional[int]' = None, tick_upper_hint: 'Optional[int]' = None, recipient: 'Optional[str]' = None, withdrawal: 'Optional[str]' = None, nonce: 'Optional[str]' = None, wrapper_proofs: 'Optional[str]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[MintResult]'`

Mint a concentrated-liquidity position as a private PositionNFT.

Tick bounds are rounded to the pool's spacing; insert hints derive
from the slot's neighbors unless given explicitly.  *withdrawal* is
the immutable payout address stored on the NFT — ``collect`` always
pays it and it can never be changed; defaults to *recipient*.
Wrapped pool sides route via the LP router; fund with UNDERLYING records.

### `increase_liquidity(self, *, pool_key: 'str', amount0_desired: 'int', amount1_desired: 'int', amount0_min: 'int' = 0, amount1_min: 'int' = 0, token0_program: 'Optional[str]' = None, token1_program: 'Optional[str]' = None, token0_record: 'Optional[str]' = None, token1_record: 'Optional[str]' = None, position_record: 'Optional[str]' = None, tick_lower_hint: 'Optional[int]' = None, tick_upper_hint: 'Optional[int]' = None, wrapper_proofs: 'Optional[str]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[TxResult]'`

Add funds to an existing position (range fixed at mint).
Wrapped pool sides route via the LP router; fund with UNDERLYING records.

### `decrease_liquidity(self, *, pool_key: 'str', liquidity_to_remove: 'int', amount0_min: 'int' = 0, amount1_min: 'int' = 0, position_record: 'Optional[str]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[TxResult]'`

Remove liquidity from a position; owed amounts become collectable.

### `collect(self, *, pool_key: 'str', amount0_requested: 'int', amount1_requested: 'int', position_record: 'Optional[str]' = None, wrapper_proofs: 'Optional[str]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[TxResult]'`

Collect owed token amounts from a position.

The payout always goes to the position's immutable ``withdrawal``
address — set at mint, not redirectable here.  Wrapped pool sides
route via the LP router, unwrapping to that address in-transaction.

### `burn(self, *, pool_key: 'str', position_record: 'Optional[str]' = None, account: 'Any' = None) -> 'DexCall[TxResult]'`

Burn an empty position NFT.

### `plan_rebalance(self, *, pool_key: 'str', position_token_id: 'str', tick_lower: 'int', tick_upper: 'int', liquidity_target: 'Optional[int]' = None, max_funding0: 'Optional[int]' = None, max_funding1: 'Optional[int]' = None) -> 'RebalancePlan'`

Quote a close-and-remint of one position into a new range.  Reads only.

Reads the pool, slot, ``positions`` entry, both current boundary ticks,
and each side's wrapped-ness, then derives what the close returns, what
the successor range needs, and per side the funding to add or surplus
to refund.  Size the successor with exactly one of *liquidity_target*
(exact) or *max_funding0*/*max_funding1* (a budget the planner solves
for; ``0, 0`` rebalances on recovered funds alone).

The plan is only valid at the pool price it was built against — the
contract asserts every amount at finalize — so build it right before
:meth:`rebalance_position` and never cache one.

Raises:
    PoolNotFoundError / PoolNotInitializedError: For an unknown pool.
    ValueError: If the position or a boundary tick does not exist, the
        aligned range is empty, the sizing is ambiguous, or the budget
        supports no liquidity.

### `rebalance_position(self, *, plan: 'Optional[RebalancePlan]' = None, pool_key: 'Optional[str]' = None, position_token_id: 'Optional[str]' = None, tick_lower: 'Optional[int]' = None, tick_upper: 'Optional[int]' = None, liquidity_target: 'Optional[int]' = None, max_funding0: 'Optional[int]' = None, max_funding1: 'Optional[int]' = None, position_record: 'Optional[str]' = None, token0_program: 'Optional[str]' = None, token1_program: 'Optional[str]' = None, token0_record: 'Optional[str]' = None, token1_record: 'Optional[str]' = None, tick_lower_hint: 'Optional[int]' = None, tick_upper_hint: 'Optional[int]' = None, deadline_offset_blocks: 'int' = 20, nonce: 'Optional[str]' = None, wrapper_proofs: 'Optional[str]' = None, imports: 'Optional[dict[str, str]]' = None, account: 'Any' = None) -> 'DexCall[RebalanceResult]'`

Close a position and mint its successor range in ONE transaction.

Burns the old position, settles its principal and every fee it earned,
adds funding from the signer's records where the new range needs more,
mints the successor with the same owner and withdrawal address, and
pays any surplus to the withdrawal address — atomically, through
``shield_swap_rebalance_router.aleo`` (deployed on testnet).  Either
pass a *plan* from :meth:`plan_rebalance` (submitted verbatim), or the
pool, range, and one sizing mode and the plan is built here.

Every amount is asserted against the pool price at execution: a trade
that moves the price between planning and finalize reverts the whole
transaction (fee paid, no funds moved).  Rebuild and resubmit when
that happens; the short default deadline fails stale requests cheaply.
Funding records for a wrapped side are the UNDERLYING asset's records.

Raises:
    ValueError: Without a plan or a complete (pool, range, sizing).
    InsufficientRecordsError: If no record covers a funded side.

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

### `get_swap_execution(self, swap: "'SwapHandle | str'") -> 'Optional[SwapExecution]'`

Per-hop fill receipt of an executed swap — what each pool leg paid.

Reads ``swap_execution_headers`` then one ``swap_execution_hops`` entry
per hop.  Returns None while the swap has not finalized.  Unlike
:meth:`get_swap_output` the receipt survives the claim, so it answers
"what did this trade actually cost" at any later time.  Reads
``1 + hop_count`` mapping entries.

Raises:
    ValueError: If the header names a hop the node did not return —
        a node lagging its own finalize; retry.

### `get_balances(self, address: 'Optional[str]' = None, account: 'Any' = None) -> 'dict[str, dict[str, Any]]'`

Public + private + total per token id, joined via the API's
token registry.  Defaults to the bound account's address; returns
only tokens actually held.

Private balances can only be scanned for the bound account's view
key — when *address* names someone else, ``private`` is 0 for every
token (their records are not scannable) rather than silently mixing
in the caller's own private holdings.

### `get_public_balances(self, programs: 'list[str]', address: 'Optional[str]' = None) -> 'dict[str, int]'`

Public balances per token program, read from each program's
on-chain ``balances`` mapping (keyed by plain address — one mapping
read per program, any address).  The public counterpart to
:meth:`get_private_balances`.  Pass the registry's
``amm_token_program`` values; an absent entry reads as ``0``.

Args:
    programs: Token programs to read; duplicates are read once.
    address: Whose balances; defaults to the bound account's.

Returns:
    Raw base-unit balances keyed by program.

Raises:
    ValueError: No address available, or a value that is not an
        unsigned-integer literal (the mapping is not ARC-20 shaped).

### `get_private_balances(self, programs: 'list[str]', account: 'Any' = None) -> 'dict[str, int]'`

Sum of unspent record amounts per wrapper program (spendable
privately).  Requires a configured record provider.

### `derive_pool_key(self, token0: 'str', token1: 'str', fee: 'int') -> 'str'`

Compute the pool key for a token pair and fee tier. Local — no network.

Deriving a key never implies the pool exists; pass the result to
:meth:`is_pool_initialized` before quoting or trading against it.  The
derivation is sensitive to token order and is network-scoped, so swapping
*token0* and *token1*, or reusing a key across mainnet and testnet, yields
a valid-looking ``field`` that matches nothing on chain.  *fee* is the
contract's ``u16`` fee tier.

### `derive_tick_key(self, pool_key: 'str', tick: 'int') -> 'str'`

Compute the key of one tick within a pool. Local — no network.

*tick* is a signed index, and the returned ``field`` is what reads that
tick's on-chain state — the initialized-tick list ``mint`` validates its
hints against.  Network-scoped like :meth:`derive_pool_key`.

