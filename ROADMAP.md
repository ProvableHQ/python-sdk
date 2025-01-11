# Python SDK Roadmap

* **Last Updated:** Jan 10, 2025
* **Last Updated By:** Michael Turner, Staff Engineer at Provable.

## Overview
The Provable Python SDK provides python libraries to build applications that interact with the Aleo Network
or use utilize the cryptographic tools available in `SnarkVM`.

This document provides a public and tentative Roadmap for future changes to the Python SDK. Since this repo is open 
source, users interested.

## Roadmap

### 2025 Q1

* **General SDK Enhancements**:
  * Update & Maintain the SDK to keep current with the latest versions of SnarkVM.
  * Add support for Mainnet and Testnet.
  * Update usage documentation.


* **Account Management**
  * Add support for signing messages over field elements.
  * Add the graph key and compute key.
  

* **Program Management and Execution**: 
  * Speed up program execution by encapsulating program execution into a single process.
  * Expose all convenience methods on the `Program` class that the corresponding `Program` struct has in SnarkVM.
  * Create a `Function` class that mirrors the `FunctionCore` struct in SnarkVM to enable inspection of functions.

* **Transaction Lifecycle Management**:
  * Enhance the `Transaction` class with more helper methods from the `Transaction` struct in SnarkVM to enable inspection 
  of the contents of a transaction.
  * Enhance the `Transition` class with more helper methods from the `Transition` struct in SnarkVM to enable inspection
    of the contents of a transaction.
  * Add support for converting between the `Plaintext` monadic type and Python types.
  * Support for decryption of private inputs within transactions.
  * Provide a python `REST API` library for interacting with SnarkOS nodes and expanded functionality provided by 
  the `Provable API` in both blocking and async flavors.


* **Support for Cryptographic Types and Hash Functions**
  * Support for field and group operations Field, Scalar, and Group elements.
  * Support for Aleo native hash functions.
