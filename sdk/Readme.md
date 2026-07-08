# Aleo Python SDK (MainnetV0)

The Aleo Python SDK provides Python bindings to Aleo's zero-knowledge cryptographic primitives, built with snarkvm 4.7.3.

## Quick Start

```python
from aleo.mainnet import PrivateKey, Signature

# Generate a random private key
pk = PrivateKey.random()
address = pk.address
view_key = pk.view_key

# Sign a message
message = b"hello"
signature = Signature.sign(pk, message)
assert signature.verify(address, message)
```

## Build & Install

```bash
bash install.sh
```

This creates a development environment and installs the `aleo` package with MainnetV0 bindings.

## Contributing
If you wish to contribute, please follow the contribution guidelines outlined on [GitHub](https://github.com/AleoHQ/python-sdk/blob/master/sdk/CONTRIBUTING.md).