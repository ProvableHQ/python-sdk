"""On-disk participant profile — key material, credentials, journal location.

One profile per home directory (``$SHIELD_SWAP_HOME`` or ``~/.shield-swap``).
``Profile.load_or_create()`` generates key material on first use and reuses
it forever after; credentials (JWT, delegated-proving keys) are stored
separately so they can be refreshed without touching the key.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

DEFAULT_ENDPOINT = "https://api.provable.com"
_PROFILE = "profile.json"
_CREDENTIALS = "credentials.json"


def _write_private(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=1))
    path.chmod(0o600)


def _generate_key(network: str) -> tuple[str, str]:
    """(private_key, address) for a fresh random account."""
    import aleo
    net = getattr(aleo, network)
    pk = net.PrivateKey.random()
    return str(pk), str(pk.address)


class Profile:
    """A participant's persistent identity and credentials.

    Load with :meth:`load_or_create`; the profile is created (with fresh key
    material) when the home directory has none yet.
    """

    def __init__(self, home: Path, data: dict[str, Any]) -> None:
        self.home = home
        self._data = data

    def __repr__(self) -> str:
        return f"Profile({self.address!r}, home={str(self.home)!r})"

    # ── Construction ────────────────────────────────────────────────────────

    @staticmethod
    def default_home() -> Path:
        env = os.environ.get("SHIELD_SWAP_HOME")
        return Path(env) if env else Path.home() / ".shield-swap"

    @classmethod
    def load_or_create(cls, home: "Path | str | None" = None, *,
                       network: str = "testnet",
                       endpoint: str = DEFAULT_ENDPOINT) -> "Profile":
        home = Path(home) if home is not None else cls.default_home()
        path = home / _PROFILE
        if path.exists():
            return cls(home, json.loads(path.read_text()))
        home.mkdir(parents=True, exist_ok=True)
        private_key, address = _generate_key(network)
        data = {"address": address, "private_key": private_key,
                "network": network, "endpoint": endpoint}
        _write_private(path, data)
        return cls(home, data)

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    def address(self) -> str:
        return self._data["address"]

    @property
    def private_key(self) -> str:
        return self._data["private_key"]

    @property
    def network(self) -> str:
        return self._data["network"]

    @property
    def endpoint(self) -> str:
        return self._data.get("endpoint", DEFAULT_ENDPOINT)

    # ── Credentials ──────────────────────────────────────────────────────────

    @property
    def credentials(self) -> dict[str, str]:
        path = self.home / _CREDENTIALS
        return json.loads(path.read_text()) if path.exists() else {}

    def save_credentials(self, **kv: Optional[str]) -> None:
        """Merge non-None values into ``credentials.json`` (mode 600)."""
        merged = {**self.credentials, **{k: v for k, v in kv.items() if v}}
        _write_private(self.home / _CREDENTIALS, merged)

    @property
    def journal_path(self) -> Path:
        return self.home / "journal.jsonl"
