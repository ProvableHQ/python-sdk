"""Run the account commands: ``python -m aleo.keystore <command>``.

``create``, ``import``, ``show``, ``protect``, ``export`` and ``migrate``
store and inspect an Aleo account on this machine; ``--help`` lists the flags
of each.
"""
from __future__ import annotations

from ._cli import main

if __name__ == "__main__":
    raise SystemExit(main())
