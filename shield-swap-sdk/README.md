# shield-swap-sdk

Typed Python client for the **shield_swap** AMM on Aleo. Sits on top of the
Aleo Python SDK's facade (`aleo.Aleo`): signer, record provider, proving
configuration, and network all come from the client you bind — this package
adds the DEX verbs, the typed results, and the off-chain DEX API, nothing
else.

```python
from aleo import Aleo
from aleo_shield_swap import ShieldSwap

aleo = Aleo(Aleo.HTTPProvider("https://api.provable.com"))
aleo.default_account = account
dex = ShieldSwap(aleo)

pools = dex.api.get_pools()                      # requires a live DEX API
handle = dex.swap(pool_key=pools[0].key,
                  token_in_id=pools[0].token0,
                  amount_in=10**9).delegate()    # broadcasts; spends funds
out = dex.claim_swap_output(handle).delegate()   # broadcasts; spends funds
```

## Install

```bash
pip install -e shield-swap-sdk                 # from the repo root
pip install -e "shield-swap-sdk[async]"        # + AsyncShieldSwap (httpx)
pip install -e "shield-swap-sdk[mcp]"          # + the MCP server
```

Requires `aleo-sdk>=0.2` (this repo's SDK; imports as `aleo`) and Python 3.10+.

## Agents

`AGENTS.md` (generated from the SDK's docstrings — always current) is the
one page an agent needs: the five-verb lifecycle, the conversation pattern,
and the building-block reference. Claude Code users get it via the
`shield-swap` skill; any MCP client can run the same lifecycle through
`python -m aleo_shield_swap.mcp`.

## How calls work

Every read returns a value immediately. Every write returns a prepared
`DexCall` — nothing touches the network until you invoke a terminal verb:

```python
call = dex.swap(pool_key=key, token_in_id=token, amount_in=10**9)

call.simulate()             # runs locally; no broadcast, no fee
call.transact(account)      # proves locally, broadcasts, pays the fee
call.delegate(account)      # proves via the delegated proving service
```

`transact` and `delegate` return the verb's *typed result* (a `SwapHandle`,
`MintResult`, `ClaimResult`, …) built from the transaction's root-transition
outputs — not a bare transaction id. Local proving downloads SNARK parameters
on first use and takes minutes for the larger entrypoints; `delegate` is the
practical path and requires DPS credentials on the provider
(`api_key=`, `network_client.consumer_id`).

Chain reads and writes live directly on `ShieldSwap`; the off-chain DEX API
is namespaced under `.api`, so a call site always shows whether a value came
from the chain or the service.

## The verb surface

**Chain reads** (node REST API):

| Verb | Returns |
|---|---|
| `get_pool(pool_key)` | The pool struct from the `pools` mapping. |
| `get_slot(pool_key)` | `SlotView` — current tick, sqrt price, liquidity, spacing. |
| `get_swap_output(swap_id)` | The finalized swap outcome; raises `SwapOutputNotFinalizedError` until the finalize lands (and again after the claim consumes it). |
| `is_pool_initialized(pool_key)` | Whether the pool exists on chain. |
| `get_private_balances(programs)` | Summed unspent record amounts per token program (needs a registered record provider). |
| `get_balances()` | Public + private balances in one shape. |
| `derive_pool_key(token0, token1, fee)` / `derive_tick_key(pool_key, tick)` | Mapping keys derived locally — no network. |

**Writes** (each returns a `DexCall`):

| Verb | What it does |
|---|---|
| `swap(...)` | Phase one of the two-transaction private swap: locks the input record against a blinded identity. Returns a `SwapHandle` — persist it if the process might die before the claim. |
| `claim_swap_output(handle)` | Phase two: claims the finalized output as a private record. |
| `create_pool(...)` | Initializes a pool for a token pair + fee tier. |
| `mint(...)` | Opens a position in a tick range; returns a `MintResult` with the position's token id. |
| `increase_liquidity(...)` / `decrease_liquidity(...)` | Resizes a position (spends the position NFT record and returns a fresh one). |
| `collect(...)` | Pays out `tokens_owed` as private records. |
| `burn(...)` | Closes an emptied position and removes it from the `positions` mapping. |

Quote before you swap: pass `expected_out` from `dex.api.get_route(...)` —
without it a spot estimate is used, which ignores fees and price impact.
Amounts are `u128` base units of the token; fees are microcredits.

**DEX API** (`dex.api`, standalone as `ApiClient`): `get_pools`,
`get_tokens`, `get_route`, `get_swap`, `get_ohlcv`, `get_public_balances`.
Route quoting, OHLCV, and balances are auth-gated — call
`api.authenticate(address, sign)` once (challenge/verify by signature, no
funds required); some deployments additionally gate them behind an invite
code.

## Privacy

The swap flow never puts your address on chain next to the output. `swap`
derives a single-use **blinded identity** from the signer's view key
(`derive_blinding_factor` / `derive_blinded_address` are exported for
verification against the TS SDK's vectors); the claim proves knowledge of the
blinding factor instead of revealing the owner.

Two conveniences trade secret material for service:

- `delegate` sends the transaction *authorization* to the proving service —
  it can see the transaction's contents (not your private key). Prove
  locally with `transact` if that is unacceptable.
- The hosted record scanner behind `get_private_balances` /
  `aleo.records.register` shares the account's **view key** with the
  scanning service, which can then see everything the account owns.
  Self-host the scanner if that is unacceptable.

## Async

`AsyncShieldSwap` / `AsyncApiClient` mirror the sync surface verb-for-verb on
`aleo.AsyncAleo` (install the `[async]` extra):

```python
from aleo_shield_swap import AsyncShieldSwap

dex = AsyncShieldSwap(async_aleo)
handle = await (await dex.swap(pool_key=key, token_in_id=token,
                               amount_in=10**9)).delegate()
```

## Agent tools and MCP

`shield_swap_tools()` returns JSON-schema tool definitions for the whole verb
surface; `dispatch_tool(dex, name, args)` executes one. For MCP hosts, the
`[mcp]` extra ships a stdio server over the same definitions:

```bash
ALEO_PRIVATE_KEY=APrivateKey1... python -m aleo_shield_swap.mcp
```

Omit `ALEO_PRIVATE_KEY` for a read-only server. See the module docstring for
the full environment (`ALEO_ENDPOINT`, `ALEO_NETWORK`, DPS credentials).

## Generated bindings

The contract surface is pinned, not hand-written: `codegen/` holds the
deployed program source, its ABI, and the DEX API's OpenAPI document.
`_generated.py` (program structs + entrypoints via `aleo.codegen`) and
`_api_models.py` (API response models) are built from those pins. When the
deployed contract or API drifts, rerun `codegen/regen-abi.sh` /
`codegen/regen-openapi.sh` and reconcile.

## Tests

```bash
python -m pytest                        # hermetic tier — no network
python -m pytest -m live               # read-only against the REAL testnet + DEX API
python -m pytest -m "live and slow"    # spends real testnet funds (DPS proving)
ALEO_DEVNODE_UNPROVEN=1 \
python -m pytest -m devnode            # full AMM lifecycle on a local aleo-devnode
```

The devnode tier deploys the vendored `shield_swap_v3.aleo` stack and drives
pool creation, liquidity, swaps, and burn end-to-end, hermetically. It needs
the `aleo-devnode` binary (`ALEO_DEVNODE_BIN` or on `PATH`) and skips
otherwise. Deployments are proofless (dummy verifying keys — the devnode
skips certificate verification); `ALEO_DEVNODE_UNPROVEN=1` extends that to
executions and is the fast path (~5 minutes). Without it, executions are
fully proven locally: expect SNARK parameter downloads and key synthesis on
first use.
