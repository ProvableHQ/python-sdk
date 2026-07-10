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

Delegated proving and the hosted record scanner authenticate with a consumer id
+ API key from the Provable API (`POST /consumers`, then `POST /jwts/<id>` with
`X-Provable-API-Key`; JWT returns in the `Authorization` header). Tests read
`ALEO_CONSUMER_ID` and `ALEO_DPS_API_KEY` from the environment; the client mints
and refreshes JWTs automatically. Prover host: `accelerate.provable.com`
(sandbox: `accelerate-sandbox.provable.com`). API/JWT host: `api.provable.com`.

## Working process

Multi-step work is tracked in `.superpowers/sdd/progress.md` (git-ignored) with
a review gate per task. Specs/plans live under `docs/superpowers/` (git-ignored,
kept local). Documentation voice: see `.agents/voice.md`.
