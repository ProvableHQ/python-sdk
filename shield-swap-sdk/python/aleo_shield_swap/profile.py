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
    """Owner-only file, written atomically (no umask window, no torn reads)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(payload, indent=1))
    os.replace(tmp, path)


def _initial_key(network: str) -> tuple[str, str]:
    """(private_key, address) for a new profile.

    Imports ``SHIELD_SWAP_PRIVATE_KEY`` (or a path in
    ``SHIELD_SWAP_PRIVATE_KEY_FILE``) when set — users with an existing
    account supply their key out-of-band, never pasted into a conversation
    — and generates a fresh random account otherwise.
    """
    import aleo
    net = getattr(aleo, network)
    key = (os.environ.get("SHIELD_SWAP_PRIVATE_KEY") or "").strip()
    key_file = os.environ.get("SHIELD_SWAP_PRIVATE_KEY_FILE")
    if not key and key_file:
        key = Path(key_file).read_text().strip()
    pk = net.PrivateKey.from_string(key) if key else net.PrivateKey.random()
    return str(pk), str(pk.address)


class Profile:
    """A participant's persistent identity and credentials.

    Holds exactly one Aleo address, generated on first use and reused every
    session after.  For several addresses, create a profile per address — each
    needs its own home directory (``SHIELD_SWAP_HOME``, or *home* on
    :meth:`load_or_create`), since the journal and credentials are per-profile
    too.

    Load with :meth:`load_or_create`; the profile is created (with fresh key
    material) when the home directory has none yet.  The same call does both,
    so callers never branch on whether one exists::

        from pathlib import Path
        from aleo_shield_swap import Profile

        # First run: generates a key and writes it owner-only.
        profile = Profile.load_or_create()
        print(profile.address)                  # aleo1…

        # Every run after: the same call returns that key, unchanged.
        assert Profile.load_or_create().address == profile.address

        # A second address needs its own home; "~/..." is expanded.
        other = Profile.load_or_create("~/.shield-swap-alt")

    That last call generates a fresh key only when no key is being imported;
    with ``SHIELD_SWAP_PRIVATE_KEY`` set, every new home adopts that one key
    instead, so all of them share an address.
    """

    def __init__(self, home: Path, data: dict[str, Any]) -> None:
        self.home = home
        self._data = data

    def __repr__(self) -> str:
        return f"Profile({self.address!r}, home={str(self.home)!r})"

    # ── Construction ────────────────────────────────────────────────────────

    @staticmethod
    def default_home() -> Path:
        """Where profiles live by default: ``$SHIELD_SWAP_HOME`` or ``~/.shield-swap``.

        Local only — returns the path whether or not it exists.  A ``~`` in
        ``SHIELD_SWAP_HOME`` is expanded, so a value set from a config file that
        does not shell-expand still resolves to the home directory rather than a
        literal ``~`` folder in the working directory.
        """
        env = os.environ.get("SHIELD_SWAP_HOME")
        return Path(env).expanduser() if env else Path.home() / ".shield-swap"

    @classmethod
    def load_or_create(cls, home: "Path | str | None" = None, *,
                       network: str = "testnet",
                       endpoint: str = DEFAULT_ENDPOINT) -> "Profile":
        """Load the profile at *home*, creating one with fresh keys if absent.

        Creating a profile writes a private key to disk at mode 600; an existing
        profile file has its mode tightened to 600 on load, healing a loose one.
        The key comes from ``SHIELD_SWAP_PRIVATE_KEY`` (or the file named by
        ``SHIELD_SWAP_PRIVATE_KEY_FILE``) when set, so a user with an existing
        account supplies it out-of-band rather than pasting it into a chat.

        *network* and *endpoint* apply only when creating — they are ignored for an
        existing profile, which keeps the values it was created with.

        Args:
            home: Profile directory; defaults to :meth:`default_home`.
            network: Network to bind a NEW profile to.
            endpoint: API endpoint to record on a NEW profile.

        Returns:
            The loaded or newly created profile.
        """
        # expanduser: a "~/..." string is otherwise taken literally, creating a
        # directory named ~ in the cwd and writing the private key there.
        home = Path(home).expanduser() if home is not None else cls.default_home()
        path = home / _PROFILE
        if path.exists():
            path.chmod(0o600)             # heal a pre-existing loose mode
            return cls(home, json.loads(path.read_text()))
        home.mkdir(parents=True, exist_ok=True)
        private_key, address = _initial_key(network)
        data = {"address": address, "private_key": private_key,
                "network": network, "endpoint": endpoint}
        _write_private(path, data)
        return cls(home, data)

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    def address(self) -> str:
        """The profile's Aleo address (``aleo1…``) — the public half of its key."""
        return self._data["address"]

    @property
    def private_key(self) -> str:
        """The profile's private key, read from disk.

        Whoever holds this controls the account and can decrypt its records — do
        not log it, echo it into a conversation, or send it to a service.
        """
        return self._data["private_key"]

    @property
    def network(self) -> str:
        """Network this profile is bound to; fixed when the profile was created.

        Key derivations are network-scoped, so reusing a profile against the other
        network yields keys that do not match anything on chain.
        """
        return self._data["network"]

    @property
    def endpoint(self) -> str:
        """API endpoint recorded for this profile, or the package default.

        Older profiles predate the stored field and fall back to the default.
        """
        return self._data.get("endpoint", DEFAULT_ENDPOINT)

    # ── Credentials ──────────────────────────────────────────────────────────

    @property
    def credentials(self) -> dict[str, str]:
        """DEX credentials saved for this profile — API tokens and session data.

        Re-read from disk on every access, so a value written by another process
        is picked up. A missing file reads as an empty dict rather than raising.
        """
        path = self.home / _CREDENTIALS
        return json.loads(path.read_text()) if path.exists() else {}

    def save_credentials(self, **kv: Optional[str]) -> None:
        """Merge non-None values into ``credentials.json`` (mode 600)."""
        merged = {**self.credentials, **{k: v for k, v in kv.items() if v}}
        _write_private(self.home / _CREDENTIALS, merged)

    def forget_credentials(self, *keys: str) -> None:
        """Drop *keys* from ``credentials.json``; absent keys are ignored."""
        merged = {k: v for k, v in self.credentials.items() if k not in keys}
        _write_private(self.home / _CREDENTIALS, merged)

    @property
    def journal_path(self) -> Path:
        """Where this profile's :class:`~aleo_shield_swap.journal.Journal` lives.

        Returns the path whether or not the file exists — the journal creates it
        on first append.
        """
        return self.home / "journal.jsonl"
