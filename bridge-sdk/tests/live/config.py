"""Gates and paths for the funded live cases — a port of veil's `test/integration/live/config.ts`.

Every function here only READS the environment.  Nothing in this repository sets, defaults or
repairs a gate variable: the acknowledgement strings exist so that a human types them into their
own shell for one command, and an agent that exports them has defeated the only safeguard between
a test run and somebody's money.  Values are never logged — errors name the VARIABLE, never what
it contained.

Gates (veil config.ts:17-36):

* ``BRIDGE_LIVE_FUNDS=1`` **and** ``BRIDGE_LIVE_STATE_DIR=<dir outside the repo>`` — live-funds
  tests exist at all.
* ``BRIDGE_LIVE_MAINNET_ACK=I_ACKNOWLEDGE_BRIDGE_MAINNET_FUNDS`` **and**
  ``BRIDGE_LIVE_MAINNET_CASES=<comma list>`` — the named mainnet case may run.
* ``BRIDGE_LIVE_MAINNET_EXECUTE=I_ACKNOWLEDGE_THIS_SUBMITS_MAINNET_TRANSACTIONS`` — the wallet may
  actually submit.  Without it a case runs to the quote and returns (veil's
  ``if (!mainnetExecutionEnabled()) return``).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping

#: The five mainnet cases veil ships, in the order the funding table lists them.
CASE_NAMES = ("evm-hyperlane", "evm-xreserve", "aleo-hyperlane", "aleo-xreserve", "solana-hyperlane")

FUNDS_VAR = "BRIDGE_LIVE_FUNDS"
STATE_DIR_VAR = "BRIDGE_LIVE_STATE_DIR"
MAINNET_ACK_VAR = "BRIDGE_LIVE_MAINNET_ACK"
MAINNET_ACK = "I_ACKNOWLEDGE_BRIDGE_MAINNET_FUNDS"
MAINNET_CASES_VAR = "BRIDGE_LIVE_MAINNET_CASES"
MAINNET_EXECUTE_VAR = "BRIDGE_LIVE_MAINNET_EXECUTE"
MAINNET_EXECUTE_ACK = "I_ACKNOWLEDGE_THIS_SUBMITS_MAINNET_TRANSACTIONS"

ENVIRONMENTS = ("mainnet", "testnet")

#: Endpoints used when the operator's shell names none. Public, read-mostly, no credentials.
DEFAULT_ALEO_ENDPOINT = "https://edge.provable.com/api"
DEFAULT_ETHEREUM_RPC_URL = "https://ethereum-rpc.publicnode.com"
DEFAULT_SEPOLIA_RPC_URL = "https://ethereum-sepolia-rpc.publicnode.com"

#: Key and RPC variables per environment, most specific first.  The live suite resolves connection
#: material ONLY through these lists, so a testnet run can never pick up the mainnet Aleo key.
ALEO_KEY_VARS = {
    "mainnet": ("BRIDGE_PRIVATE_KEY",),
    "testnet": ("BRIDGE_LIVE_ALEO_TESTNET_PRIVATE_KEY", "ALEO_E2E_PRIVATE_KEY"),
}
EVM_KEY_VARS = {
    "mainnet": ("EVM_PRIVATE_KEY", "BRIDGE_EVM_PRIVATE_KEY"),
    "testnet": ("BRIDGE_LIVE_EVM_TESTNET_PRIVATE_KEY", "EVM_PRIVATE_KEY", "BRIDGE_EVM_PRIVATE_KEY"),
}
EVM_RPC_VARS = {
    "mainnet": ("ETHEREUM_RPC_URL", "BRIDGE_LIVE_ETHEREUM_RPC_URL"),
    "testnet": ("SEPOLIA_RPC_URL", "BRIDGE_LIVE_SEPOLIA_RPC_URL"),
}
DEFAULT_EVM_RPC_URL = {"mainnet": DEFAULT_ETHEREUM_RPC_URL, "testnet": DEFAULT_SEPOLIA_RPC_URL}
ALEO_ENDPOINT_VARS = ("BRIDGE_LIVE_ALEO_ENDPOINT", "ALEO_ENDPOINT")

#: The Aleo network name each bridge environment runs on.
ALEO_NETWORKS = {"mainnet": "mainnet", "testnet": "testnet"}

#: Recipient overrides per destination-chain family; the default is our own address on that chain.
RECIPIENT_VARS = {
    "aleo": "BRIDGE_LIVE_ALEO_MAINNET_RECIPIENT",
    "evm": "BRIDGE_LIVE_EVM_RECIPIENT",
    "solana": "BRIDGE_LIVE_SOLANA_RECIPIENT",
}

_EVM_KEY_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class LiveConfigError(Exception):
    """A live case was explicitly enabled but is not configured (never carries a secret value)."""


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def value(name: str, env: Mapping[str, str] | None = None) -> str | None:
    """The trimmed value of *name*, or None when unset or blank. Never logs."""
    raw = _env(env).get(name)
    trimmed = raw.strip() if isinstance(raw, str) else None
    return trimmed or None


def required(name: str, env: Mapping[str, str] | None = None) -> str:
    """veil ``required()``: the value of *name*, or a clear error naming only the variable."""
    found = value(name, env)
    if not found:
        raise LiveConfigError(f"Missing {name}; the live bridge case was explicitly enabled but is not configured")
    return found


def required_evm_private_key(name: str, env: Mapping[str, str] | None = None) -> str:
    """veil ``requiredEvmPrivateKey()``: one 32-byte key normalised to ``0x…``; the value never appears."""
    raw = required(name, env)
    body = raw[2:] if raw[:2].lower() == "0x" else raw
    if not _EVM_KEY_RE.match(body):
        raise LiveConfigError(f"{name} must contain exactly 32 hexadecimal bytes")
    return f"0x{body.lower()}"


def live_funds_enabled(env: Mapping[str, str] | None = None) -> bool:
    """veil ``liveFundsEnabled()``: funded cases exist at all (flag exactly ``"1"`` + a state dir)."""
    source = _env(env)
    return source.get(FUNDS_VAR) == "1" and bool(value(STATE_DIR_VAR, env))


def mainnet_case_enabled(name: str, env: Mapping[str, str] | None = None) -> bool:
    """veil ``mainnetCaseEnabled()``: funds gate + the exact acknowledgement + *name* in the case list."""
    if not live_funds_enabled(env):
        return False
    if _env(env).get(MAINNET_ACK_VAR) != MAINNET_ACK:
        return False
    listed = {entry.strip() for entry in (value(MAINNET_CASES_VAR, env) or "").split(",")}
    return name in (listed - {""})


def mainnet_execution_enabled(env: Mapping[str, str] | None = None) -> bool:
    """veil ``mainnetExecutionEnabled()``: the wallet may submit. Read here, typed by a human elsewhere."""
    return _env(env).get(MAINNET_EXECUTE_VAR) == MAINNET_EXECUTE_ACK


def one_atomic_unit(decimals: int) -> str:
    """veil ``oneAtomicUnit()``: the smallest positive display amount of an asset ("0.000001" at 6)."""
    if isinstance(decimals, bool) or not isinstance(decimals, int) or decimals < 0:
        raise LiveConfigError(f"Invalid asset decimals: {decimals!r}")
    return "1" if decimals == 0 else f"0.{'0' * (decimals - 1)}1"


def state_dir(env: Mapping[str, str] | None = None) -> Path:
    """``BRIDGE_LIVE_STATE_DIR`` as a path — the operator's own directory, outside this repository."""
    return Path(required(STATE_DIR_VAR, env)).expanduser()


def live_state_path(environment: str, name: str, env: Mapping[str, str] | None = None) -> Path:
    """veil ``liveStatePath()``: ``<BRIDGE_LIVE_STATE_DIR>/<environment>/<name>.json``."""
    if environment not in ENVIRONMENTS:
        raise LiveConfigError(f"environment must be one of {ENVIRONMENTS}, got {environment!r}")
    return state_dir(env) / environment / f"{name}.json"


def case_route_override(case: str, env: Mapping[str, str] | None = None) -> str | None:
    """``BRIDGE_LIVE_<CASE>_ROUTE_ID`` (veil's per-case route override), or None."""
    return value(f"BRIDGE_LIVE_{case.replace('-', '_').upper()}_ROUTE_ID", env)


def case_amount_override(case: str, env: Mapping[str, str] | None = None) -> str | None:
    """``BRIDGE_LIVE_<CASE>_AMOUNT``, or ``BRIDGE_LIVE_XRESERVE_AMOUNT`` for either xReserve case."""
    specific = value(f"BRIDGE_LIVE_{case.replace('-', '_').upper()}_AMOUNT", env)
    if specific:
        return specific
    return value("BRIDGE_LIVE_XRESERVE_AMOUNT", env) if case.endswith("xreserve") else None


def _environment(environment: str) -> str:
    if environment not in ENVIRONMENTS:
        raise LiveConfigError(f"environment must be one of {ENVIRONMENTS}, got {environment!r}")
    return environment


def first_value(names: tuple[str, ...], env: Mapping[str, str] | None = None) -> tuple[str, str] | None:
    """The first of *names* that is set, as ``(variable, value)`` — or None. Never logs the value."""
    for name in names:
        found = value(name, env)
        if found:
            return name, found
    return None


def aleo_private_key(environment: str, env: Mapping[str, str] | None = None) -> str:
    """The Aleo key for *environment* (``BRIDGE_PRIVATE_KEY`` on mainnet, ``ALEO_E2E_PRIVATE_KEY``
    — or its ``BRIDGE_LIVE_ALEO_TESTNET_PRIVATE_KEY`` alias — on testnet)."""
    names = ALEO_KEY_VARS[_environment(environment)]
    found = first_value(names, env)
    if found is None:
        raise LiveConfigError(f"Missing {' / '.join(names)}; the {environment} live case needs an Aleo key")
    return found[1]


def evm_private_key(environment: str, env: Mapping[str, str] | None = None) -> str:
    """The EVM key for *environment*, normalised to ``0x…`` (the value never appears in an error)."""
    names = EVM_KEY_VARS[_environment(environment)]
    found = first_value(names, env)
    if found is None:
        raise LiveConfigError(f"Missing {' / '.join(names)}; the {environment} live case needs an EVM key")
    return required_evm_private_key(found[0], env)


def evm_rpc_url(environment: str, env: Mapping[str, str] | None = None) -> str:
    """The Ethereum (mainnet) or Sepolia (testnet) RPC url, falling back to the public default."""
    found = first_value(EVM_RPC_VARS[_environment(environment)], env)
    return found[1] if found else DEFAULT_EVM_RPC_URL[environment]


def aleo_endpoint(env: Mapping[str, str] | None = None) -> str:
    """The Aleo API root, falling back to the open, credential-free edge host."""
    found = first_value(ALEO_ENDPOINT_VARS, env)
    return found[1] if found else DEFAULT_ALEO_ENDPOINT


def recipient_override(family: str, env: Mapping[str, str] | None = None) -> str | None:
    """The operator's recipient override for a destination-chain *family*, or None (use our own address)."""
    try:
        name = RECIPIENT_VARS[family]
    except KeyError:
        raise LiveConfigError(f"No recipient override variable for chain family {family!r}") from None
    return value(name, env)


__all__ = [
    "ALEO_ENDPOINT_VARS", "ALEO_KEY_VARS", "ALEO_NETWORKS", "CASE_NAMES", "DEFAULT_ALEO_ENDPOINT",
    "DEFAULT_ETHEREUM_RPC_URL", "DEFAULT_EVM_RPC_URL", "DEFAULT_SEPOLIA_RPC_URL", "ENVIRONMENTS",
    "EVM_KEY_VARS", "EVM_RPC_VARS", "FUNDS_VAR", "LiveConfigError", "MAINNET_ACK", "MAINNET_ACK_VAR",
    "MAINNET_CASES_VAR", "MAINNET_EXECUTE_ACK", "MAINNET_EXECUTE_VAR", "RECIPIENT_VARS", "STATE_DIR_VAR",
    "aleo_endpoint", "aleo_private_key", "case_amount_override", "case_route_override", "evm_private_key",
    "evm_rpc_url", "first_value", "live_funds_enabled", "live_state_path", "mainnet_case_enabled",
    "mainnet_execution_enabled", "one_atomic_unit", "recipient_override", "required",
    "required_evm_private_key", "state_dir", "value",
]
