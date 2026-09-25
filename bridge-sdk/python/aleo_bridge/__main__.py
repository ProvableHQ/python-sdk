"""``python -m aleo_bridge`` prints the agent guide; ``[status|routes|assets]`` run those checks
(``status`` needs BRIDGE_PRIVATE_KEY, read-only)."""
from __future__ import annotations

import dataclasses
import json
import sys

from .registry import DEFAULT_REGISTRY

USAGE = "usage: python -m aleo_bridge [status|routes|assets]"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        from . import agent_guide
        print(agent_guide(), end="")
        return 0
    command = args[0]
    if command == "routes":
        print(json.dumps([r.id for r in DEFAULT_REGISTRY.routes(include_unavailable=True)], indent=1))
        return 0
    if command == "assets":
        print(json.dumps([a.id for a in DEFAULT_REGISTRY.assets()], indent=1))
        return 0
    if command == "status":
        from .client import Bridge
        print(json.dumps(dataclasses.asdict(Bridge.from_env().status()), indent=1, default=str))
        return 0
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
