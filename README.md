# Aleo Python SDK

Welcome to the Aleo Python SDK! This SDK provides a set of libraries aimed at empowering Python developers with zk (zero-knowledge) capabilities.

## Quick Start

```python
from aleo.mainnet import PrivateKey, Signature

# Generate a random private key
pk = PrivateKey.random()
print(f"Address: {pk.address}")
print(f"View Key: {pk.view_key}")

# Sign and verify a message
message = b"hello"
sig = Signature.sign(pk, message)
assert sig.verify(pk.address, message)
```

Built with snarkvm 4.7.3 (MainnetV0). For build instructions, see [sdk/Readme.md](./sdk/Readme.md).

## Codebases Included

- [**sdk**](./sdk/): A library that brings Aleo MainnetV0 functionalities to Python developers.
- [**zkml**](./zkml/): A transpiler library that converts Python machine learning models into Leo code.
- [**zkml-research**](./zkml-research/): Research on accurate/constraint-efficient zkML techniques, mostly for internal purposes.

For detailed information on each codebase, please navigate to their respective folders.
