# shield-swap — agent guide

> GENERATED from SDK docstrings by `codegen/gen_context.py` — do not
> edit by hand; edit the docstrings and regenerate.

Typed Python client for the shield_swap AMM on Aleo
(`pip install shield-swap-sdk`, imports as `aleo_shield_swap`).
MCP alternative: `python -m aleo_shield_swap.mcp` exposes the same
lifecycle as tools.

## Tier 1 — start and resume a trading session

```python
from aleo_shield_swap import Profile, ShieldSwap

Profile.load_or_create(storage="encrypted")  # Protect a new key; existing storage stays unchanged.
dex = ShieldSwap.from_profile()              # Resume with the same account and journal.
dex.onboard()                            # first run registers; no-op afterwards
pools = dex.api.get_pools()
report = dex.swap_many(pool_key=pools[0].key, token_in_id=pools[0].token0,
                       amount_in=10**6, count=5)
dex.collect_all()                        # any session, any time
```

### `from_profile(home=None, *, network=None, endpoint=None)`

Continue trading with the same account, credentials, and swap journal.

Use the same *home* on each run to retain pending swaps and saved API
credentials. On first use, the client uses an existing Aleo account
or creates one. To keep an existing address and its funds, import
that account before starting; see :meth:`Profile.load_or_create`.

For a new account, use macOS Keychain or password encryption to
protect the private key. Set this up before building the client::

    from aleo_shield_swap import Profile, ShieldSwap

    Profile.load_or_create(storage="encrypted")
    dex = ShieldSwap.from_profile()

This prompts for a password if none is supplied by the environment
or Keychain. It does not change an account's existing storage.
Without this setup, a generated key is saved as plaintext: anyone
who can read the account file can spend its funds. Use
``storage="keychain"`` on macOS to keep the key outside the file.

An encrypted account needs its password each time the client starts.
For unattended runs, set ``ALEO_ACCOUNT_PASSWORD`` or save the password
in the macOS Keychain. A key supplied through ``ALEO_PRIVATE_KEY`` or
``ALEO_PRIVATE_KEY_FILE`` must also be supplied each run; it is never
saved by the profile.

The client attempts to register with the hosted record scanner,
sharing the account's view key so the scanner can read its records.
For a custom record provider, construct ``ShieldSwap(aleo)`` instead.
Call ``onboard()`` to register a new account with Shield Swap before
trading. Client construction does not submit a trade.

Args:
    home: Directory holding the profile and its saved work. Defaults to
        ``SHIELD_SWAP_HOME``, then ``~/.shield-swap``. Use a separate
        directory for each network.
    network: ``"mainnet"`` or ``"testnet"``; defaults to testnet for a
        new profile. An existing profile keeps its recorded network.
    endpoint: API service root for a new profile; defaults to
        ``https://edge.provable.com/api``. Ignored for an existing profile.

Raises:
    aleo.keystore.KeystoreError: The account cannot be opened. Check
        ``code``: ``PASSWORD_REQUIRED`` means a password must be supplied;
        ``ACCOUNT_NOT_FOUND`` means an existing profile's key is missing;
        ``ADDRESS_MISMATCH`` means the selected key belongs to another
        address. Restore the intended key before trying again.

### `onboard(referral_code=None)`

Register this profile end to end — safe to re-run any time.

Runs only the registration stages not already satisfied (see
``lifecycle.REGISTRATION_STAGES``); a registered, funded account is
a no-op.  Needs nothing from you: access is granted by
authentication alone.  *referral_code* is optional — pass one a
friend shared to credit them as the referrer (recorded once; later
calls ignore it).  Requires a profile-bound client
(``from_profile()``).

### `status()`

One re-orientation call: identity, access, holdings, pending work.

Run this first in any session — it answers "is this account already
registered, what do I hold, what is in flight" from the profile,
journal, chain, and API without changing anything.

### `get_positions(account=None)`

Every open position — journaled ones plus a record scan.

The scan catches positions the journal never saw (account used from
another machine, journal lost); it needs a registered record
provider and is skipped silently without one.

### `swap_many(*, pool_key, token_in_id, amount_in, count, slippage_bps=50, expected_out=None, record_wait_seconds=120.0, account=None)`

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

### `collect_all(account=None)`

Claim every finalized swap and collect owed fees on open positions.

Safe to run any time, from any session: works off the journal, skips
swaps whose finalize hasn't landed (they stay pending for next time),
never double-claims, and requests exactly the owed amounts the chain
reports.  Requires ``from_profile()``.

## Serving a chatting user (the conversation pattern)

### Before doing anything: two questions

1. **Existing account?** Ask before creating a profile: an existing
   account's funds stay with its original key. Have the user import that key
   in their own terminal with
   `python -m aleo.keystore import --storage encrypted`, or use
   `--storage keychain` on macOS. These protect the saved key; plaintext
   storage leaves it readable to anyone with access to the account file.
   After import, `Profile.load_or_create(create=False)` requires that account
   rather than generating another key if setup is missing.
   **Never ask for a private key in the conversation.** Do not run
   `python -m aleo.keystore export` or read account files to inspect an
   address; use `python -m aleo.keystore show` instead. An account supplied
   through `ALEO_PRIVATE_KEY` or `ALEO_PRIVATE_KEY_FILE` must be supplied
   each run; the profile does not save that key.
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
| Bot / server / CLI / notebook | `aleo-sdk` facade + `shield-swap-sdk` (this package) | A saved Aleo account. Import an existing key to retain its funds, or create an account with Keychain or encrypted storage before calling `ShieldSwap.from_profile()`. Delegated proving uses the Provable prover; fees are covered by default. |
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
- **Rebalancing replaces the complete position in one transaction** —
  choose `plan_rebalance_exact_amounts` for exact quotes,
  `plan_rebalance_to_liquidity` for exact liquidity with actual recovery, or
  `plan_rebalance_from_amounts` for execution-time sizing from token caps.
  Review the plan, then pass `plan=` to the matching `rebalance_position_*`
  method. Flexible funding/refunds are fixed; surplus stays collectible on
  the successor. `estimated_liquidity` is advisory, and token minimums do not
  guarantee minimum liquidity. The router is supported on testnet; flexible
  calls require an upgraded contract. A rejection rolls back funds movements
  but can still cost a fee — refresh state before retrying.
- **Concurrency needs partitioned blinded-identity counters AND disjoint
  input records** — `swap_many` implements the recipe; copy it, don't
  improvise.

For the funding, swap, and claim flow, see
`examples/first-swap/`.

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

### `api.authenticate(address, sign)`

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

### `api.referral_status()`

This account's referral picture: ``referred_by`` (the referrer's
address once a code was redeemed, else None), ``my_code`` (the code
this account shares), and ``has_access``.  Network read.

``has_access`` is always true for an authenticated account — access
is granted by authentication alone, no code required — and the call
raises :class:`NotAuthenticatedError` when the session is missing or
expired, which makes this the session liveness probe (the dedicated
``/access/status`` route was retired in 2026-09).

### `api.my_referral_code()`

The referral code this account hands to others.

The API issues the code on the first request, so this normally
returns a value; None means issuance is disabled for the account.
Network read.

### `api.redeem_code(code)`

Redeem a referral code (``POST /referral/redeem``) — optional.

Access does not depend on this: authentication alone unlocks every
endpoint.  Redeeming records who referred the account, once: a
repeat returns ``status="already_redeemed"`` without changing
anything, while an unknown code or the account's own code is a 400
(:class:`DexApiError`).

Sessions live on the ``/auth/*`` endpoints, so no token comes
back (one is still adopted if the API resurrects the legacy
body-JWT).

### `api.request_airdrop(address)`

Start the test-token airdrop job for *address* (private records).

One claim per address per 15 minutes — raises
:class:`AirdropRateLimitedError` on 429.  Poll the returned
``job_id`` with :meth:`get_airdrop_job`.

### `api.get_airdrop_job(job_id)`

Progress of an airdrop job — ``running`` until every transfer lands.

### `api.create_api_token(name, expires_in_days=None)`

Mint a long-lived DEX API token (the secret is returned ONCE).

JWTs from :meth:`authenticate` expire in 24h; persist the returned
``.token`` for durable access.  Tiering (verified live): ``ss_…``
tokens work on data/trading endpoints; ``/access/*`` and token
management still require a session JWT.

### `api.get_pools()`

Every pool the DEX lists, each with its two tokens' metadata.

An entry exposes the pool's own fields directly — ``entry.key`` is the
``pool_key`` that ``swap``, ``mint``, and ``collect`` take.  Its
``token0_info`` / ``token1_info`` carry that token's ``symbol`` and
``decimals``, but the API does not guarantee them — check for ``None``
before reading.

### `api.get_tokens()`

Every token the DEX lists, with its id, symbol, and decimals.

``decimals`` converts between the two amount conventions.
The API returns canonical decimal amounts (``"1.5"``), if using this
value to call on-chain methods — ``swap(amount_in=…)``, ``mint``,
``collect`` — conversion to raw base units is necessary.

### `api.get_route(*, token_in, token_out, amount_in=None, pool_key=None)`

Best route between two tokens.  *amount_in* is a CANONICAL
decimal amount (human units, e.g. ``1.5``) — not base units —
and the returned ``estimated_amount_out`` is decimal too.  *pool_key*
pins the quote to one pool instead of the router's best path.

### `api.get_route_topology()`

The routable token graph: every (token0, token1) edge with an
enabled pool and the router's ``max_hops``.  Lets a client enumerate
reachable pairs without probing ``/route`` per pair.

### `api.get_unclaimed()`

Everything the authenticated account can still collect, as the
indexer sees it: swaps with finalized-but-unclaimed output and
positions with owed balances.  A cross-check for a local journal —
the chain, not this, gates the claim amounts.

### `api.get_protocol_state(*, minimum_revision=None)`

The indexer's view of protocol configuration and its own freshness.

``freshness.ready_for_quote`` says whether quotes reflect the chain
head; ``revision`` increments on every config change and is echoed by
``/route`` as ``protocol_revision`` — pass *minimum_revision* to wait
for the indexer to reach one.  Returned unwrapped (no ``data``).

### Counters & blinding

Blinded identities derive deterministically from (view key, counter,
program).  Counters must NEVER be reused (the finalize rejects a reused
blinded address after the proof is paid for).  Let `swap`/`swap_many`
pick them: with a journal they reserve under its lock AND verify each
counter on chain; without one they probe the chain.  Persist
`SwapHandle`s — the blinding factor is the claim secret.

### `blinded_identity_at(aleo, account, program, counter)`

The identity at an exact *counter* — no on-chain probing.

Use with journal-reserved counters for concurrent swaps;
:func:`next_blinded_identity` (probe-based) remains the recovery path
when no journal exists.

### `next_blinded_identity(aleo, account, program='shield_swap.aleo', *, start_counter=0, max_scan=64, gallop=True)`

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

### `confirm_airdrop(*, timeout=600.0, poll_interval=5.0, account=None)`

Fund the account and wait for its airdrop records to become spendable.

Requires API authentication and scanner registration. ``timeout`` covers
faucet settlement and scanning. Check ``success`` and ``error`` before
swapping. A record timeout retains the transfer results; inspect those
transactions before requesting another airdrop. Scanner errors propagate.

### `quote(*, token_in, token_out, amount_in, slippage_bps=50)`

Quote a direct or multi-hop swap from two symbols and a token amount.

Returns the API's best route and final slippage floor; pass it to
``swap(quote)``. Requires API authentication. Quoting spends no funds
and needs no record scanning or transaction signature.

### `swap(quote=None, *, pool_key=None, token_in_id=None, amount_in=None, slippage_bps=None, expected_out=None, sqrt_price_limit=None, deadline_offset_blocks=10000, nonce=None, identity=None, token_in_program=None, token_record=None, compliance_proofs=None, wrapper_proofs=None, track=True, imports=None, account=None)`

Prepare a direct or multi-hop swap; submit with transact/delegate.

Pass a ``SwapQuote`` to execute its full route and final slippage floor.
Alternatively pass pool_key, token_in_id, and amount_in for one pool;
trade parameters cannot override a quote. Legacy strings/Decimal use
token units, integers base units. Returned handle amounts are base units.

Wrapped inputs spend underlying records. ``token_record`` bypasses scanning.
Journaled calls reserve a counter during preparation and retain the
submitted handle. Without a journal, save the handle and avoid concurrent
swaps; use an explicit identity for concurrency. ``track=False`` bypasses
journaling. ``deadline_offset_blocks`` defaults to 10,000 (~8 hours).
Freeze-list proofs: leave ``compliance_proofs`` unset and the SDK builds
them, or pass 0–1 bundles (the signer's, for a wrapped input).

### `claim_swap_output(handle, *, timeout=0.0, compliance_proofs=None, wrapper_proofs=None, imports=None, account=None)`

Claim a private swap's output.

With a journal, ``delegate(wait=True)`` or ``transact(wait=True)``
records the confirmed claim automatically.

Reads the chain-computed result from ``swap_outputs`` (never an
off-chain service — these amounts gate money movement), proves
ownership of the blinded identity, and claims. Wrapped tokens unwrap
automatically. Output and refunds arrive as private records owned by
the signer; claiming consumes the mapping entry.

After swap confirmation, ``timeout`` sets the maximum wait in seconds
(default 0). Missing output is polled every 2s without submitting. Timeout raises
``SwapOutputNotFinalizedError``; other errors propagate.
Freeze-list proofs: leave ``compliance_proofs`` unset and the SDK builds
them, or pass 1–3 bundles (AMM signer, then each wrapped leg paid).

### `create_pool(*, token0_id, token1_id, fee, initial_tick, tick_spacing=None, initial_sqrt_price=None, imports=None, account=None)`

Create a pool — a single public transaction, no records involved.

The fee tier must be registered with the program (validated before
submission); tick spacing defaults to the tier's on-chain binding and
the opening price to the tick's sqrt price.

### `mint(*, pool_key, tick_lower, tick_upper, amount0_desired, amount1_desired, amount0_min=0, amount1_min=0, token0_program=None, token1_program=None, token0_record=None, token1_record=None, tick_lower_hint=None, tick_upper_hint=None, recipient=None, withdrawal=None, nonce=None, compliance_proofs=None, wrapper_proofs=None, imports=None, account=None)`

Mint a concentrated-liquidity position as a private PositionNFT.

Tick bounds are rounded to the pool's spacing; insert hints derive
from the slot's neighbors unless given explicitly.  *withdrawal* is
the immutable payout address stored on the NFT — ``collect`` always
pays it and it can never be changed; defaults to *recipient*.
Wrapped pool sides route via the LP router; fund with UNDERLYING records.
Freeze-list proofs: leave ``compliance_proofs`` unset and the SDK builds
them, or pass 3–5 bundles (wrapped senders, then signer, recipient, withdrawal).

### `increase_liquidity(*, pool_key, amount0_desired, amount1_desired, amount0_min=0, amount1_min=0, token0_program=None, token1_program=None, token0_record=None, token1_record=None, position_record=None, tick_lower_hint=None, tick_upper_hint=None, compliance_proofs=None, wrapper_proofs=None, imports=None, account=None)`

Add funds to an existing position (range fixed at mint).
Wrapped pool sides route via the LP router; fund with UNDERLYING records.
Freeze-list proofs: leave ``compliance_proofs`` unset and the SDK builds
them, or pass 0–2 bundles (the signer's, per wrapped side).

### `decrease_liquidity(*, pool_key, liquidity_to_remove, amount0_min=0, amount1_min=0, position_record=None, imports=None, account=None)`

Remove liquidity from a position; owed amounts become collectable.

### `collect(*, pool_key, amount0_requested, amount1_requested, position_record=None, compliance_proofs=None, wrapper_proofs=None, imports=None, account=None)`

Collect owed token amounts from a position.

The payout always goes to the position's immutable ``withdrawal``
address — set at mint, not redirectable here.  Wrapped pool sides
route via the LP router, unwrapping to that address in-transaction.
Freeze-list proofs: leave ``compliance_proofs`` unset and the SDK builds
them, or pass 2–4 bundles (owner, withdrawal, then a receiver per wrapped side).

### `burn(*, pool_key, position_record=None, account=None)`

Burn an empty position NFT.

### `plan_rebalance(*, pool_key, position_token_id, tick_lower, tick_upper, liquidity_target=None, max_funding0=None, max_funding1=None, amount0_desired=None, amount1_desired=None, amount0_min=None, amount1_min=None, funded0=None, funded1=None, refund0=None, refund1=None)`

Plan a complete position replacement without signing or submitting.

A positive target with omitted bounds preserves exact quoted recovery
and deposits. ``max_funding0/1`` also produce an exact quote sized at
the current price. Explicit ``amount0/1_desired`` cap token consumption;
``amount0/1_min`` default to the caps. Widen either bound to use actual
recovery, or set ``liquidity_target=0`` to size liquidity at execution.
Flexible funding and refunds are fixed amounts, defaulting to zero;
surplus remains collectible on the successor and earns no LP fees.

Amounts are integer token base units, from 0 through 2**128 - 1.
``estimated_liquidity`` is advisory; token floors do not guarantee a
minimum liquidity. Flexible plans must fit the quoted token
bounds and available funds; funding errors name the amount to supply.
Named planners supply mode-specific defaults. Requires an existing pool, position, and initialized boundary ticks.

### `rebalance_position(*, plan=None, pool_key=None, position_token_id=None, tick_lower=None, tick_upper=None, liquidity_target=None, max_funding0=None, max_funding1=None, amount0_desired=None, amount1_desired=None, amount0_min=None, amount1_min=None, funded0=None, funded1=None, refund0=None, refund1=None, position_record=None, token0_program=None, token1_program=None, token0_record=None, token1_record=None, tick_lower_hint=None, tick_upper_hint=None, deadline_offset_blocks=20, nonce=None, compliance_proofs=None, wrapper_proofs=None, imports=None, account=None)`

Prepare an atomic replacement in the same pool, keeping owner and withdrawal.

Pass a reviewed ``plan`` verbatim, or pool/range/sizing arguments as in
``plan_rebalance``. Pool/position arguments must match the plan, and
supplied ticks must align to its range. Sizing/amount arguments
are rejected. Exact bounds enforce the quote; wider bounds use actual
recovery and a zero target sizes liquidity at execution. Funding and
immediate refunds stay fixed; flexible surplus remains collectible on
the successor. Wrapped funding uses the underlying asset's records.

Preparation reads chain state and resolves records and proofs. Submit
with ``delegate()`` or ``transact()``. A rejection rolls back all funds
movements but can still cost a fee; refresh state before retrying.
The default deadline is 20 blocks. Flexible calls require a deployment
supporting execution-time recovery and amount-driven rebalancing.
Freeze-list proofs are built automatically unless supplied explicitly.

### `plan_rebalance_exact_amounts(*, pool_key, position_token_id, tick_lower, tick_upper, liquidity_target=None, max_funding0=None, max_funding1=None, amount0_desired=None, amount1_desired=None, amount0_min=None, amount1_min=None, funded0=None, funded1=None, refund0=None, refund1=None)`

Quote a complete replacement with fixed recovery and deposits.

Supply a positive target or paired ``max_funding0/1`` budgets. Minimums
must equal caps. Price or fee changes can reject execution. Reads only.

### `rebalance_position_exact_amounts(*, plan=None, pool_key=None, position_token_id=None, tick_lower=None, tick_upper=None, liquidity_target=None, max_funding0=None, max_funding1=None, amount0_desired=None, amount1_desired=None, amount0_min=None, amount1_min=None, funded0=None, funded1=None, refund0=None, refund1=None, position_record=None, token0_program=None, token1_program=None, token0_record=None, token1_record=None, tick_lower_hint=None, tick_upper_hint=None, deadline_offset_blocks=20, nonce=None, compliance_proofs=None, wrapper_proofs=None, imports=None, account=None)`

Prepare an exact quote using target-or-budget sizing, or a reviewed plan.

Uses ``plan_rebalance_exact_amounts`` constraints. Price or fee changes
can reject execution. Submit with ``delegate()`` or ``transact()``.

### `plan_rebalance_to_liquidity(*, pool_key, position_token_id, tick_lower, tick_upper, liquidity_target=None, max_funding0=None, max_funding1=None, amount0_desired=None, amount1_desired=None, amount0_min=None, amount1_min=None, funded0=None, funded1=None, refund0=None, refund1=None)`

Quote fixed successor liquidity with actual recovered funds.

Supply a positive target and both caps; one bound must be flexible.
Minimums and fixed funding/refunds default to zero. Rejects quotes
outside the bounds or available funds. Later price changes can still
reject execution. Reads only.

### `rebalance_position_to_liquidity(*, plan=None, pool_key=None, position_token_id=None, tick_lower=None, tick_upper=None, liquidity_target=None, max_funding0=None, max_funding1=None, amount0_desired=None, amount1_desired=None, amount0_min=None, amount1_min=None, funded0=None, funded1=None, refund0=None, refund1=None, position_record=None, token0_program=None, token1_program=None, token0_record=None, token1_record=None, tick_lower_hint=None, tick_upper_hint=None, deadline_offset_blocks=20, nonce=None, compliance_proofs=None, wrapper_proofs=None, imports=None, account=None)`

Prepare fixed successor liquidity with bounded token consumption.

Uses ``plan_rebalance_to_liquidity`` constraints, or a reviewed plan.
Insufficient execution-time funds reject rather than minting less.
Submit with ``delegate()`` or ``transact()``.

### `plan_rebalance_from_amounts(*, pool_key, position_token_id, tick_lower, tick_upper, liquidity_target=None, max_funding0=None, max_funding1=None, amount0_desired=None, amount1_desired=None, amount0_min=None, amount1_min=None, funded0=None, funded1=None, refund0=None, refund1=None)`

Estimate replacement liquidity from both token caps.

Omit the target or set zero. Minimums and fixed funding/refunds default
to zero. Estimates and token minimums do not guarantee liquidity.
Surplus stays collectible; no swap occurs. Reads only; integer base units.

### `rebalance_position_from_amounts(*, plan=None, pool_key=None, position_token_id=None, tick_lower=None, tick_upper=None, liquidity_target=None, max_funding0=None, max_funding1=None, amount0_desired=None, amount1_desired=None, amount0_min=None, amount1_min=None, funded0=None, funded1=None, refund0=None, refund1=None, position_record=None, token0_program=None, token1_program=None, token0_record=None, token1_record=None, tick_lower_hint=None, tick_upper_hint=None, deadline_offset_blocks=20, nonce=None, compliance_proofs=None, wrapper_proofs=None, imports=None, account=None)`

Prepare a replacement sized from available tokens at execution.

Uses ``plan_rebalance_from_amounts`` constraints, or a reviewed plan.
Estimated liquidity is advisory; surplus stays collectible.
Submit with ``delegate()`` or ``transact()``.

### `plan_inventory(token, target, *, max_transactions=None, account=None)`

Plan the joins and splits that give *token* the record layout trades need.

One trade spends one record, so several trades sold at the same time
need several records that each cover a trade. The plan targets the
token's underlying record program (the records a trader holds), not
the AMM wrapper. Reads the token registry, the program, and the
scanner; signs and reserves nothing. Review ``plan.steps`` and
``plan.deduction`` before executing with :meth:`rebalance_inventory`.

Args:
    token: Symbol such as ``"USDCx"`` or the on-chain token id.
    target: Desired count, minimum amount in base units, and distribution.
    max_transactions: Operation ceiling, 1 through 10000. Defaults to 100.
    account: Signer; defaults to ``aleo.default_account``.

Raises:
    ShieldSwapError: The token is unknown or has no underlying program.
    InventoryPlanError: The target cannot be reached from the available records.

Example::

    plan = dex.plan_inventory("USDCx", InventoryTarget(records=6, min_record_amount=100_000, distribution="balanced"))

### `rebalance_inventory(token, target, *, max_transactions=None, timeout=120.0, poll_interval=1.0, max_fee_microcredits=None, cancel=None, account=None)`

Reshape *token*'s underlying records to *target*, one confirmed transaction at a time.

Plans as :meth:`plan_inventory` does, then executes through the
facade's record inventory: each step reserves its inputs, proves,
broadcasts, and waits for the scanner to show its outputs before the
next step. Requires the facade to be built with record management
enabled. No accepted step is rolled back; an interrupted result names
the reason and every submitted transaction.

Example::

    result = dex.rebalance_inventory("USDCx", InventoryTarget(records=6), max_fee_microcredits=5_000_000)
    if result.status == "interrupted":
        dex.inventory.reconcile_record_inventory()

### `get_pool(pool_key)`

Static pool configuration (token pair, fee, decimal scales).

### `get_slot(pool_key)`

Live trading state (sqrt price, tick, in-range liquidity).

Raises :class:`PoolNotFoundError` when the pool does not exist, or
:class:`PoolNotInitializedError` when it exists but has no slot yet.

### `get_swap_output(swap)`

Chain-computed output of a finalized swap request.

Accepts the :class:`SwapHandle` from ``swap()`` or a bare swap id.
Raises :class:`SwapOutputNotFinalizedError` when the entry is absent —
not finalized yet (retry after a few blocks) or already claimed.

### `get_swap_execution(swap)`

Per-hop fill receipt of an executed swap — what each pool leg paid.

Reads ``swap_execution_headers`` then one ``swap_execution_hops`` entry
per hop.  Returns None while the swap has not finalized.  Unlike
:meth:`get_swap_output` the receipt survives the claim, so it answers
"what did this trade actually cost" at any later time.  Reads
``1 + hop_count`` mapping entries.

Raises:
    ValueError: If the header names a hop the node did not return —
        a node lagging its own finalize; retry.

### `get_balances(address=None, account=None, *, include_private=True)`

Public + private + total per token id, joined via the API's
token registry.  Defaults to the bound account's address; returns
only tokens actually held.

Private balances can only be scanned for the bound account's view
key — when *address* names someone else, ``private`` is 0 for every
token (their records are not scannable) rather than silently mixing
in the caller's own private holdings.  ``include_private=False``
skips the record scan (public only — before scanner credentials).

### `get_public_balances(programs, address=None)`

Public balances per token program from each program's on-chain
``balances`` mapping (plain-address key; one read per program, any
address) — the public counterpart to :meth:`get_private_balances`.
Pass the registry's ``amm_token_program`` values.  Raw base units
keyed by program; an absent entry reads as ``0``.  A program this
network lacks, or a non-ARC-20 value, is skipped with a warning
rather than failing every other token.  *address* defaults to the
bound account's (ValueError when there is none).

### `get_private_balances(programs, account=None)`

Sum of unspent record amounts per wrapper program (spendable
privately).  Requires a configured record provider.

### `derive_pool_key(token0, token1, fee)`

Compute the pool key for a token pair and fee tier. Local — no network.

Deriving a key never implies the pool exists; pass the result to
:meth:`is_pool_initialized` before quoting or trading against it.  The
derivation is sensitive to token order and is network-scoped, so swapping
*token0* and *token1*, or reusing a key across mainnet and testnet, yields
a valid-looking ``field`` that matches nothing on chain.  *fee* is the
contract's ``u16`` fee tier.

### `derive_tick_key(pool_key, tick)`

Compute the key of one tick within a pool. Local — no network.

*tick* is a signed index, and the returned ``field`` is what reads that
tick's on-chain state — the initialized-tick list ``mint`` validates its
hints against.  Network-scoped like :meth:`derive_pool_key`.

