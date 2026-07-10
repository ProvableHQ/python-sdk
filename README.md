# Aleo Python SDK

Welcome to the Aleo Python SDK! This SDK provides a set of libraries aimed at empowering Python developers with zk (zero-knowledge) capabilities.

## Quick Start

```python
from aleo import Aleo

# Connect to mainnet
aleo = Aleo(Aleo.HTTPProvider("https://api.provable.com/v2"))

print(aleo.is_connected())          # True
print(aleo.network_name)            # "mainnet"

# Check a public balance (microcredits; 1 credit == 1_000_000 microcredits)
address = "aleo1..."
balance = aleo.get_balance(address)
print(aleo.from_microcredits(balance), "credits")
```

### Call a program — the verb ladder

```python
# Fetch a live program and build a call
credits = aleo.programs.get("credits.aleo")

# Inspect without touching the network
call = credits.functions.transfer_public(
    "aleo1recipient...",   # address
    1_000_000,             # u64 microcredits (auto-coerced)
)
print(call.signature)   # "transfer_public(address, u64)"

# Dry-run locally (no proof, no send)
result = call.simulate(account)     # AuthorizationResult

# Build + broadcast (proves locally; needs a funded account)
# requires a live node + funded key
tx_id = call.transact(account)
print("tx:", tx_id)
```

### Delegate — the flagship frictionless path

```python
# The prover's fee master pays by default — no credits needed on your side.
# requires prover credentials; fee master pays
tx = aleo.programs.get("my_app.aleo") \
         .functions.my_function("arg1", 42) \
         .delegate(account)
```

No fee record. No public balance. The DPS handles proving and fee payment.

### Low-level primitives

For direct access to `PrivateKey`, `Signature`, `Program`, and the raw network client, see [sdk/Readme.md](./sdk/Readme.md).

---

Built with snarkvm 4.8.1 (MainnetV0). For build instructions, see [sdk/Readme.md](./sdk/Readme.md).

## Codebases Included

- [**sdk**](./sdk/): A library that brings Aleo MainnetV0 functionalities to Python developers.
- [**zkml**](./zkml/): A transpiler library that converts Python machine learning models into Leo code.
- [**zkml-research**](./zkml-research/): Research on accurate/constraint-efficient zkML techniques, mostly for internal purposes.

For detailed information on each codebase, please navigate to their respective folders.
