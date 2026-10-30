"""On-disk Aleo identity for the bridge — one profile per home directory (``$ALEO_BRIDGE_HOME`` or ``~/.aleo-bridge``).

Holds ONLY the Aleo key (mode 600). EVM and Solana keys are never written by this package; they come from
``EVM_PRIVATE_KEY`` / ``SOLANA_PRIVATE_KEY`` or caller-built connections. Shares no code with shield-swap.
"""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

DEFAULT_ENDPOINT = "https://edge.provable.com/api"
NETWORKS = ("mainnet", "testnet")
_PROFILE = "profile.json"
_CHECKPOINTS = "checkpoints"


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
                       endpoint: str = DEFAULT_ENDPOINT, rpc_urls: "Mapping[str, str] | None" = None) -> "Profile":
        if network not in NETWORKS:
            raise ConfigurationError(f"Profile network must be one of {NETWORKS}, got {network!r}")
        for chain_id, url in (rpc_urls or {}).items():
            _check_rpc_url(chain_id, url, network)
        home_path = Path(home).expanduser() if home is not None else cls.default_home()
        path = home_path / _PROFILE
        if path.exists():
            path.chmod(0o600)                       # heal a loose mode on load
            profile = cls(home_path, json.loads(path.read_text()))
        else:
            if home_path.is_dir():
                home_path.chmod(0o700)              # heal a loose mode on an existing dir
            else:
                home_path.mkdir(parents=True, exist_ok=True, mode=0o700)
                home_path.chmod(0o700)              # mkdir's mode is subject to umask; make it exact
            private_key, address = _initial_key(network)
            data = {"address": address, "private_key": private_key, "network": network, "endpoint": endpoint}
            if rpc_urls:
                data["rpc_urls"] = dict(rpc_urls)
            try:
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                # Lost the creation race to another process — adopt the winner's key rather than
                # clobbering its (already in-use) profile.
                path.chmod(0o600)
                data = json.loads(path.read_text())
            else:
                with os.fdopen(fd, "w") as handle:
                    handle.write(json.dumps(data, indent=1))
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

    @property
    def rpc_urls(self) -> dict[str, str]:
        """Read-only RPC URLs per registry chain id, used by ``Bridge.from_profile`` for chains with no other
        connection (pause-switch reads on the far side of an Aleo-origin route). URLs only; never a key."""
        return dict(self._data.get("rpc_urls", {}))

    def set_rpc_url(self, chain_id: str, url: str) -> None:
        """Persist *url* for *chain_id* (explicit ``ethereum=``/``evm=`` arguments and env variables still win)."""
        _check_rpc_url(chain_id, url, self.network)
        self._data.setdefault("rpc_urls", {})[chain_id] = url
        path = self.home / _PROFILE
        tmp = path.with_suffix(".json.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(self._data, indent=1))
        os.replace(tmp, path)
        path.chmod(0o600)


def _check_rpc_url(chain_id: str, url: str, network: str) -> None:
    """Refuse what ``Bridge.from_profile`` could not load: an unknown, non-EVM or other-environment chain."""
    from .registry import DEFAULT_REGISTRY
    from .errors import RouteNotFoundError
    try:
        chain = DEFAULT_REGISTRY.chain(chain_id)
    except RouteNotFoundError as exc:
        raise ConfigurationError(f"RPC URL for unknown chain {chain_id!r}; registry chains: "
                                 f"{[c.id for c in DEFAULT_REGISTRY.chains() if c.family == 'evm']}") from exc
    if chain.family != "evm":
        raise ConfigurationError(f"{chain.id} is not an EVM chain; the profile stores RPC URLs for EVM chains only")
    if chain.environment != network:
        raise ConfigurationError(f"{chain.id} is a {chain.environment} chain but this profile is {network}")
    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
        raise ConfigurationError(f"RPC URL for {chain_id!r} must start with https:// (or http:// for a local node), got {url!r}")


__all__ = ["DEFAULT_ENDPOINT", "Profile"]
