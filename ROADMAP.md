# Python SDK Roadmap

* **Last Updated:** Mar 21, 2025
* **Last Updated By:** Michael Turner, Staff Engineer at Provable.

# Overview
The Provable Python SDK provides python libraries to build applications that interact with the Aleo Network
or use utilize the cryptographic tools available in `SnarkVM`.

This document provides a public and tentative Roadmap for future changes to the Python SDK. Since this repo is open 
source, users interested.

# Q2 Roadmap

## April 2025

### General SDK Enhancements
  * Update & Maintain the SDK to keep current with the latest versions of SnarkVM.
  * Add support for Mainnet and Testnet.
  * Build Python SDK packages for all available architectures.
  * Build extensive tests for all SDK components.
  * Republish Python SDK package to PyPi.

### Account Management
  * Add support for signing messages over field elements.
  * Add the graph key and compute key.
  * Add the same methods that exist within the JS SDK for account management.

### Serialization
  * Add support for Aleo data and grammatical types (identifiers, etc.) for the following serialization methods:
    * to_bytes/from_bytes
    * to_bits/from_bits
    * to_fields/from_fields`

### Support for Cryptographic Types and Hash Functions
* Build support for all supported operations on algebraic types (Field, Scalar, and Group).
* Build arithmetic operations into python dunder methods where applicable (`__ge__`, `__rmul__`, `__rand__`, etc.).
* Support for Aleo hash functions.
* Add support for converting between the `Plaintext` monadic type and Python types.
* Add support for ciphertext types and mirror the operations available in the JS SDK.

## May 2025 

### Transaction Lifecycle Management
  * Enhance the `Transaction` class with more helper methods from the `Transaction` struct in SnarkVM to enable inspection
    of the contents of a transaction.
  * Enhance the `Transition` class with more helper methods from the `Transition` struct in SnarkVM to enable inspection
    of the contents of a transaction.
    * Support for decryption of private inputs within transactions.
    * Support
  * Provide a python `REST API` library for interacting with SnarkOS nodes and expanded functionality provided by
    the `Provable API` in both blocking and async flavors.

### Program Management and Execution
  * Create `WebVM` class that mirrors the `VM` struct that has the execution methods supported by the `VM` struct. 
  * Expand `Process` class to support all methods available in the `Process` struct in SnarkVM.
  * Expose all convenience methods on the `Program` class that the corresponding `Program` struct has in SnarkVM.
  * Create a `Function` class that mirrors the `FunctionCore` struct in SnarkVM to enable inspection of functions.

## June 2025

### Support Semantics for building function inputs directly in Python
  * Allow users to create `Plaintext` style objects directly in Python semantics.
    * Allow python `dictionaries` and `dataclasses` to be converted directly to `Plaintext` structs or `Records`.
    * Allow python `lists` to be converted directly to `Plaintext` arrays objects.
  * Support export of existing plaintext literals to Python and Numpy types.
  * Support explicit conversion between Numpy types and `Literal` types.
  * Support conversion of floats to fixed-point types for usage in function inputs.

### Expand Proving and Verification Tools
  * Add support for prove_vk and verify_vk methods.
  * Add support for downloading and managing parameters manually.



