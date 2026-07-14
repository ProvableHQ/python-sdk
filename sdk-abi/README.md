# aleo-contract-abi-generator

Generates a JSON ABI from deployed Aleo bytecode, and checks one ABI against
another for compatibility. Rust bindings around Leo's ABI generation with
full snarkVM validation — a program that does not validate does not get an
ABI. Everything runs locally; nothing touches the network.

The distribution is `aleo-contract-abi-generator`; the import module is
`aleo_abi`.

```python
import aleo_abi

abi_json = aleo_abi.generate_abi("simple.aleo", bytecode, "testnet", None)
violations = aleo_abi.check_compatibility(candidate_json, standard_json)
```

## Install

```bash
pip install aleo-contract-abi-generator
```

Most callers want the friendlier hook in the main SDK instead —
`aleo.abi.generate_abi` accepts a `Program` object or a raw bytecode string,
infers the program name, and returns a dict:

```python
from aleo import abi
from aleo.mainnet import Program

result = abi.generate_abi(Program.credits())
result["program"]        # "credits.aleo"
```

The hook imports this package lazily and raises `ImportError` with the
install command if it is absent — the main SDK does not depend on it.

## Generating an ABI

`generate_abi(program_name, bytecode, network, imports)` returns a
pretty-printed JSON string describing the program: its `structs`, `records`,
`mappings`, and `functions`, each field carrying a structured type
(`{"Primitive": {"Int": "I32"}}`, record ownership, visibility).

snarkVM validation is **contextual**: a program that declares imports is
rejected unless those imports are supplied. Pass them as
`(program_id, bytecode)` pairs in topological order — dependencies before
dependents:

```python
abi_json = aleo_abi.generate_abi(
    "shield_swap_v3.aleo", amm_bytecode, "testnet",
    [("test_shield_swap_multisig_core.aleo", multisig_bytecode)],
)
```

`network` is `"mainnet"`, `"testnet"`, or `"canary"` — it selects the
validation rules, not a connection.

## Checking compatibility

`check_compatibility(candidate_abi_json, standard_abi_json)` returns a list
of violation strings — empty means the candidate satisfies the standard.
Use it to pin a deployed contract's surface and fail *your* CI when the
deployment drifts, instead of your consumers:

```python
violations = aleo_abi.check_compatibility(deployed_abi, pinned_abi)
assert not violations, "\n".join(violations)
```

This is how `shield-swap-sdk`'s live drift test guards its committed
bindings (`codegen/regen-abi.sh` regenerates the pin from the deployed
program).

## Downstream: generated Python bindings

The ABI JSON is the input format for the main SDK's `aleo.codegen`, which
emits typed dataclasses and entrypoint stubs from it:

```bash
python -m aleo.codegen --abi shield_swap.abi.json --out _generated.py
```

## Tests

```bash
cd sdk-abi && python -m pytest python/tests -v   # hermetic — no network
```

Fixtures and expected ABIs are vendored from Leo's own test suite, so the
output shape is pinned to upstream.
