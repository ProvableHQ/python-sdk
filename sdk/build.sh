#!/bin/bash

## This script assumes you have already installed Rust 1.6+ and Python 3.8.5+
## The script below will install the Aleo Python bindings and create the aleo library with MainnetV0 support.
python3 -m venv .env
source .env/bin/activate
pip install maturin
maturin build --features mainnet
