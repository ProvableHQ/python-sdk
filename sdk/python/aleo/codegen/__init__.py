"""aleo.codegen — build-time ABI→Python emitter.

Turns an ``aleo-abi`` JSON description of a program into a module of frozen
dataclasses with ``to_plaintext()`` encoders and ``from_plaintext()``
decoders.  Build-time only: nothing in the ``aleo`` runtime imports this
package, and generated modules import only :mod:`aleo.codegen.runtime`.

Usage: ``python -m aleo.codegen --abi abi.json --out generated.py``.
"""
