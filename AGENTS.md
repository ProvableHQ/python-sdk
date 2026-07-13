# Aleo Python SDK — Agent Guide

Python SDK for Aleo: PyO3 bindings over **snarkvm v4.8.1** plus pure-Python
client / record-scanner / delegated-proving layers and a Web3.py-style facade.
Two shipped packages:

| Path | Package | What |
| --- | --- | --- |
| `sdk/` | `aleo` | the PyO3 crate (`_aleolib_mainnet`) + the `aleo` Python package (accounts, algebra, programs, records, algorithms, network client, record scanner, facade) |
| `sdk-abi/` | `aleo-abi` | a **separate** package that generates contract ABIs from Aleo bytecode via Leo's `leo-abi` crate; kept isolated so Leo's snarkvm feature flags never reach the main crate |

## Build & test

```sh
cd sdk && python3 -m venv .env && . .env/bin/activate && pip install maturin
maturin develop --features mainnet          # build + install the extension
python -m pytest python/tests -m "not slow"  # fast suite (what CI runs)
python -m pytest python/tests -m slow         # network/proving tests (run locally)
npx -y pyright@latest                         # strict type check (must be 0)
cargo fmt --check && cargo clippy --no-default-features --features mainnet -- -D warnings
cargo check --no-default-features --features testnet   # guard the testnet cfg seam
```

Run pytest from the package dir so it uses that package's `pytest.ini`
(`sdk/pytest.ini`, `sdk-abi/pytest.ini`) — a legacy root `setup.cfg` injects
addopts if you invoke pytest from the repo root.

## Architecture conventions (follow these)

- **Two-build, no macro.** One flat codebase; `CurrentNetwork`/`CurrentAleo`
  are cfg-selected (`--features mainnet|testnet`) and compiled once per network
  into `_aleolib_mainnet` / `_aleolib_testnet`, exposed as `aleo.mainnet.*` /
  `aleo.testnet.*`. The Web3.py facade sits on top; network comes from the
  provider.
- **Sync + async pairs.** Every HTTP client ships both (`AleoNetworkClient` /
  `AsyncAleoNetworkClient`, scanner likewise, `Aleo` / `AsyncAleo`): sync on
  `requests`, async on `httpx`, shared pure logic in a common module — never
  duplicate orchestration.
- **Pythonic (anti-viem) surface.** snake_case; pure getters are `@property`;
  Python dunders (`__str__`/`__bytes__`/`__eq__`/`__hash__`) over
  `to_string()/to_bytes()/equals()`; explicit `PrivateKey.random()` (no
  implicit-RNG constructor). Do NOT mirror the TS/viem method names where a
  Python idiom exists.
- **Algebraic types expose BOTH** a dunder and a named method (`__add__`+`add`,
  `__mul__`+`multiply`, Group `__mul__`+`scalar_multiply`, `__neg__`+`negate`).
  snarkvm's `inverse()` is negation → expose as `negate`, never `inverse`.
- **Both stubs.** Update `python/aleo/_aleolib_mainnet.pyi` (`list[...]` style)
  AND `python/aleo/__init__.pyi` (`List[...]` style) for every binding change.
  Pure-Python modules need no stub — annotate inline; pyright strict must pass.
- **aleo-abi isolation.** Never add a Leo crate to `sdk/`. The CI `test-abi`
  lane asserts `dev_skip_checks` never appears in `sdk/Cargo.lock`.

## Testing discipline

- Known-answer vectors are vendored from `ProvableHQ/sdk` (and its `wasm/`
  crate) with a `_source` note (path + commit). Copy constants character-exact.
- **Never weaken an assertion to make a test pass.** A KAT mismatch is a real
  signal — report it, don't paper over it.
- Proving / live-network tests are `@pytest.mark.slow` (SRS downloads, live
  REST); CI runs `-m "not slow"`.

## Delegated services (DPS + record scanner)

**Auth, proving, and scanning are all Provable *services* on `api.provable.com`,
hosted at the API ORIGIN — NOT under the read node's `/v2/{network}` base.** The
read/RPC endpoints live at `https://api.provable.com/v2/{network}/…`; the
services hang off the bare origin (`https://api.provable.com`) at their own path
prefixes. Each is confirmed working against live testnet (see
`tests/e2e/test_testnet_e2e.py`):

- **JWT auth** — origin, no prefix: `POST {origin}/jwts/{consumerId}`. Derive the
  origin with `jwt_origin(base_url)` (`scheme://host`, path stripped).
- **Delegated proving** — `{origin}/prove/{network}` prefix:
  - `GET {origin}/prove/{network}/pubkey` — ephemeral X25519 key + key id + a
    `Set-Cookie` **affinity** session. The ephemeral private key lives only on
    the backend that served this call, so the follow-up `/prove` MUST stick to
    it: rely on the client cookie **jar** (requests.Session / httpx.AsyncClient)
    to re-send the cookie — do NOT hand-build a `Cookie` header from
    `set-cookie` (comma-joins multiple cookies, carries attributes, and bypasses
    the jar).
  - `POST {origin}/prove/{network}/prove/authorization` (or `/prove/request`) —
    sealed-box `{key_id, ciphertext}`; JWT + the affinity cookie; SDK retries
    500/503.
- **Record scanner** — `{origin}/scanner/{network}` prefix (env
  `RECORD_SCANNER_URL=https://api.provable.com/scanner`).

Documented endpoints (docs describe paths *relative to the service base*; the
base is the origin + service prefix above):
- Register → API key: `POST /consumers` — https://docs.provable.com/docs/api/services/get-auth-register
- Issue/refresh JWT: `POST /jwts/:consumerId` (send `X-Provable-API-Key`; JWT
  returns in the `Authorization: Bearer …` header) — https://docs.provable.com/docs/api/services/issue-jwt
- Ephemeral prover pubkey: `GET /pubkey` — https://docs.provable.com/docs/api/services/get-prove-pubkey
- Submit proving: `POST /prove/authorization` (the authorization endpoint; the
  docs also describe a `/prove/encrypted` variant). https://docs.provable.com/docs/api/services/post-prove-encrypted

Tests read `ALEO_CONSUMER_ID` / `ALEO_DPS_API_KEY` from the env; the client mints
and refreshes JWTs automatically. The prover base defaults to
`{jwt_origin(base_url)}/prove/{network}` — override via `HTTPProvider(prover_uri=…)`
only to point at a non-default prover host.

## Working process

Multi-step work is tracked in `.superpowers/sdd/progress.md` (git-ignored) with
a review gate per task. Specs/plans live under `docs/superpowers/` (git-ignored,
kept local). Documentation voice: see `.agents/voice.md`.
