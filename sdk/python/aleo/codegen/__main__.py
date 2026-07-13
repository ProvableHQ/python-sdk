"""CLI: python -m aleo.codegen --abi abi.json --out generated.py [--config cfg.json]

Config mode drives multiple programs from one JSON file
(``{"programs": [{"abi": "...", "out": "..."}]}``); paths inside a config
resolve relative to the config file's own location.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._emit import emit_module


def _generate(abi_path: Path, out_path: Path) -> None:
    abi = json.loads(abi_path.read_text())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(emit_module(abi))
    print(f"generated {len(abi.get('structs', []))} structs, "
          f"{len(abi.get('records', []))} records, "
          f"{len(abi.get('mappings', []))} mapping decoders -> {out_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="aleo.codegen")
    p.add_argument("--abi", type=Path, help="path to ABI JSON")
    p.add_argument("--out", type=Path, help="output .py path")
    p.add_argument("--config", type=Path, help="config JSON with a programs list")
    args = p.parse_args(argv)
    try:
        if args.config:
            cfg = json.loads(args.config.read_text())
            base = args.config.parent
            for entry in cfg["programs"]:
                _generate((base / entry["abi"]).resolve(), (base / entry["out"]).resolve())
        elif args.abi and args.out:
            _generate(args.abi, args.out)
        else:
            p.error("provide --abi and --out, or --config")
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        print(f"aleo.codegen: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
