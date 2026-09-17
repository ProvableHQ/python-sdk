"""On-disk Aleo identity for the bridge — one profile per home directory (``$ALEO_BRIDGE_HOME`` or ``~/.aleo-bridge``).

Holds ONLY the Aleo key (mode 600). EVM and Solana keys are never written by this package; they come from
``EVM_PRIVATE_KEY`` / ``SOLANA_PRIVATE_KEY`` or caller-built connections. Shares no code with shield-swap.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

DEFAULT_ENDPOINT = "https://edge.provable.com/api"
NETWORKS = ("mainnet", "testnet")
_PROFILE = "profile.json"
_CHECKPOINTS = "checkpoints"


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    """Owner-only file written atomically (no umask window, no torn reads)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def _initial_key(network: str) -> tuple[str, str]:
    """(private_key, address): imported from BRIDGE_PRIVATE_KEY / BRIDGE_PRIVATE_KEY_FILE, else freshly random."""
    import aleo
    net = getattr(aleo, network)
    key = (os.environ.get("BRIDGE_PRIVATE_KEY") or "").strip()
    key_file = os.environ.get("BRIDGE_PRIVATE_KEY_FILE")
    if not key and key_file:
        key = Path(key_file).expanduser().read_text().strip()
    private_key = net.PrivateKey.from_string(key) if key else net.PrivateKey.random()
    return str(private_key), str(private_key.address)


class Profile:
    """A persistent Aleo identity: created on first use, reused every session after.

    ``Profile.load_or_create()`` does both, so callers never branch on existence. *network* and
    *endpoint* apply only when creating; an existing profile keeps the values it was created with.
    """

    def __init__(self, home: Path, data: dict[str, Any]) -> None:
        self.home = home
        self._data = data

    def __repr__(self) -> str:
        return f"Profile({self.address!r}, network={self.network!r}, home={str(self.home)!r})"

    @staticmethod
    def default_home() -> Path:
        env = os.environ.get("ALEO_BRIDGE_HOME")
        return Path(env).expanduser() if env else Path.home() / ".aleo-bridge"

    @classmethod
    def load_or_create(cls, home: "Path | str | None" = None, *, network: str = "mainnet",
                       endpoint: str = DEFAULT_ENDPOINT) -> "Profile":
        if network not in NETWORKS:
            raise ConfigurationError(f"Profile network must be one of {NETWORKS}, got {network!r}")
        home_path = Path(home).expanduser() if home is not None else cls.default_home()
        path = home_path / _PROFILE
        if path.exists():
            path.chmod(0o600)                       # heal a loose mode on load
            profile = cls(home_path, json.loads(path.read_text()))
        else:
            home_path.mkdir(parents=True, exist_ok=True)
            private_key, address = _initial_key(network)
            data = {"address": address, "private_key": private_key, "network": network, "endpoint": endpoint}
            _write_private(path, data)
            profile = cls(home_path, data)
        profile.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return profile

    @property
    def address(self) -> str:
        return self._data["address"]

    @property
    def private_key(self) -> str:
        """Controls the account — never log it or send it to a service."""
        return self._data["private_key"]

    @property
    def network(self) -> str:
        return self._data["network"]

    @property
    def endpoint(self) -> str:
        return self._data.get("endpoint", DEFAULT_ENDPOINT)

    @property
    def checkpoint_dir(self) -> Path:
        """Directory plan 4 binds as ``FileCheckpointStore``; created with the profile."""
        return self.home / _CHECKPOINTS


__all__ = ["DEFAULT_ENDPOINT", "Profile"]
