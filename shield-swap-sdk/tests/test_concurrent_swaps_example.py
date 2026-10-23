"""Offline checks of the concurrent-swaps example: it parses, documents its flags, and refuses to run without a key."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).parents[1] / "examples" / "concurrent-swaps" / "concurrent_swaps.py"


def run(*args: str, key: str | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "SHIELD_SWAP_PRIVATE_KEY"}
    if key is not None:
        env["SHIELD_SWAP_PRIVATE_KEY"] = key
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "python")
    return subprocess.run([sys.executable, str(EXAMPLE), *args], capture_output=True, text=True, env=env, timeout=120)


def test_help_documents_the_execute_opt_in() -> None:
    result = run("--help")
    assert result.returncode == 0
    assert "--execute" in result.stdout and "--count" in result.stdout and "--amount" in result.stdout


def test_refuses_to_run_without_a_key_before_touching_the_network() -> None:
    result = run()
    assert result.returncode != 0
    assert "SHIELD_SWAP_PRIVATE_KEY" in result.stderr


def test_rejects_an_out_of_range_count_before_touching_the_network() -> None:
    result = run("--count", "1", key="unused")
    assert result.returncode != 0 and "--count must be from 2 through 16" in result.stderr
