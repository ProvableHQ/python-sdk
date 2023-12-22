#!/bin/bash

## This script assumes you have already installed Rust 1.6+ and Python 3.8.5+
## The script below will install the Aleo Python bindings and create the aleo_python library allowing you to use
## the Aleo implementation of the Poseidon hash function in Python.
python3 -m venv .env
source .env/bin/activate
pip install maturin
maturin develop
python python/test.py
