"""Private USDCx mint identities: Shield-compatible nonce derivation (veil PR #166).

A private xReserve deposit commits the Aleo recipient to a secret scalar on the source chain, and
only that scalar lets the recipient run ``private_mint`` later. Shield derives the scalar from the
account's view key and a counter; this module reproduces that derivation exactly (shield-core
``cc29a60``, vendored vectors in ``tests/fixtures/shield-core-private-mint.json``) so a local Aleo
account never has to invent, pass or safeguard a nonce itself.

Nothing here contacts a network. The view key and the derived scalar never appear in a ``repr``,
a checkpoint or an error message.
"""
from __future__ import annotations

import json
import re
import threading
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager, Iterable, Iterator, Protocol, cast

from . import encoding
from ._fs import atomic_write_text, counter_lock
from .errors import ConfigurationError

#: Field literals of the little-endian ASCII labels ``shielded_usdcx_mainnet`` / ``_testnet`` —
#: Shield's per-network domain separators, copied character-exact from veil.
SHIELDED_USDCX_DOMAIN_MAINNET = "43548994738795619508419071049301721479887331556681843field"
SHIELDED_USDCX_DOMAIN_TESTNET = "43548994873472583992984066613247233935881976249149555field"
_DOMAINS = {"mainnet": SHIELDED_USDCX_DOMAIN_MAINNET, "testnet": SHIELDED_USDCX_DOMAIN_TESTNET}

DEFAULT_WRAPPER_PROGRAM = "shielded_usdcx_wrapper.aleo"
MAX_COUNTER = 0xFFFFFFFF


def domain_field(network: str) -> str:
    """The Poseidon domain for *network*: ``shielded_usdcx_<network>`` as a little-endian field literal."""
    try:
        return _DOMAINS[network]
    except KeyError:
        raise ConfigurationError(f"network must be one of {tuple(_DOMAINS)}, got {network!r}") from None


def derive_secret_nonce(view_key: Any, counter: int, network: str, program: str = DEFAULT_WRAPPER_PROGRAM) -> str:
    """Shield's ``shielded-usdcx-secret-nonce``: ``Poseidon8.hash_to_scalar([wrapper_x, domain, view_key, counter])``.

    *view_key* is a ``ViewKey`` or its ``AViewKey1…`` string; *counter* is the identity's u32.
    """
    if isinstance(counter, bool) or not isinstance(counter, int) or not 0 <= counter <= MAX_COUNTER:
        raise ConfigurationError(f"Private mint identity counter must be a u32 (0..{MAX_COUNTER}), got {counter!r}")
    domain = domain_field(network)
    net = encoding.network_module(network)
    try:
        vk = net.ViewKey.from_string(str(view_key))
    except Exception as exc:
        raise ConfigurationError("Invalid Aleo view key for private mint derivation") from exc
    program_field = net.Address.from_program_id(program).to_group().to_x_coordinate()
    inputs = [program_field, net.Field.from_string(domain), vk.to_scalar().to_field(), net.U32(counter).to_field()]
    return str(net.Poseidon8().hash_to_scalar(inputs))


def derive_address_commitment(recipient: str, secret_nonce: str, network: str) -> str:
    """``BHP256.commit(bits(recipient), secret_nonce)`` as 64 lowercase hex characters, no prefix."""
    return encoding.private_mint_commitment(recipient, secret_nonce, network).hex()


def commitment_from_hook_data(hook_data: bytes) -> str:
    """The commitment inside a canonical private-mint hook (selector 2, 32 commitment bytes, 32 zero bytes)."""
    raw = bytes(hook_data)
    if len(raw) != encoding.HOOK_DATA_BYTES:
        raise ConfigurationError(f"Private mint hook data must contain {encoding.HOOK_DATA_BYTES} bytes")
    if raw[0] != 2 or raw[33:] != bytes(32):
        raise ConfigurationError("xReserve hook data is not a canonical private mint commitment")
    return raw[1:33].hex()


@dataclass(frozen=True)
class PrivateMintIdentity:
    """One reserved private-mint identity: the counter it was derived from, the recipient it commits,
    the secret scalar (never printed) and the public 32-byte commitment as lowercase hex."""
    counter: int
    recipient: str
    secret_nonce: str
    address_commitment: str

    def __repr__(self) -> str:
        return (f"PrivateMintIdentity(counter={self.counter}, recipient={self.recipient!r}, "
                f"address_commitment={self.address_commitment!r})")

    def to_dict(self) -> dict[str, Any]:
        return {"counter": self.counter, "recipient": self.recipient, "secretNonce": self.secret_nonce,
                "addressCommitment": self.address_commitment}

    @classmethod
    def from_dict(cls, data: Any) -> "PrivateMintIdentity":
        """Validate one stored record the way veil's ``validateIdentity`` does; never repair it."""
        if not isinstance(data, dict):
            raise ConfigurationError(_INVALID_RECORD)
        counter, recipient = data.get("counter"), data.get("recipient")
        nonce, commitment = data.get("secretNonce"), data.get("addressCommitment")
        if (isinstance(counter, bool) or not isinstance(counter, int) or not 0 <= counter <= MAX_COUNTER
                or not isinstance(recipient, str) or not recipient.startswith("aleo1")
                or not isinstance(nonce, str) or not nonce.endswith("scalar")
                or not isinstance(commitment, str) or not _COMMITMENT_RE.fullmatch(commitment)):
            raise ConfigurationError(_INVALID_RECORD)
        return cls(counter=counter, recipient=recipient, secret_nonce=nonce, address_commitment=commitment)


_COMMITMENT_RE = re.compile(r"[0-9a-f]{64}")
_INVALID_RECORD = ("Private mint identity store contains an invalid record; restore the identities file "
                   "from a backup before reserving or completing a private mint")
_EXHAUSTED = "Private mint identity counter space is exhausted"
DEFAULT_SCAN_LIMIT = 10_000
_FILE_VERSION = 1


# ── stores ────────────────────────────────────────────────────────────────────

class PrivateMintIdentityStore(Protocol):
    """Persistence that keeps a private-mint counter from being issued twice.

    ``allocate_counter(minimum)`` durably consumes a counter at or above *minimum* and returns it;
    it runs before derivation, so a failed derivation or save never recycles the counter. A store
    whose backing resource is shared between instances or processes should also expose
    ``exclusive()`` (a context manager) — :func:`reserve_identity` holds it around load → allocate
    → derive → save when present.
    """

    def load(self) -> list[PrivateMintIdentity]: ...
    def save(self, identities: list[PrivateMintIdentity]) -> None: ...
    def allocate_counter(self, minimum: int) -> int: ...


class MemoryPrivateMintIdentityStore:
    """A process-local store for tests and explicit opt-in. It restarts its counter at zero with
    the process, so it is never chosen by default: a reused counter links two deposits on chain."""

    def __init__(self, initial: Iterable[PrivateMintIdentity] = ()) -> None:
        self._identities = list(initial)
        self._next = 0

    def load(self) -> list[PrivateMintIdentity]:
        return list(self._identities)

    def save(self, identities: list[PrivateMintIdentity]) -> None:
        self._identities = list(identities)

    def allocate_counter(self, minimum: int) -> int:
        self._next = max(self._next, minimum)
        if self._next > MAX_COUNTER:
            raise ConfigurationError(_EXHAUSTED)
        value = self._next
        self._next += 1
        return value


class FilePrivateMintIdentityStore:
    """Identities in a JSON file, the next counter in ``<name>.counter``, and a ``<name>.lock``
    inode that serializes reservations across instances and processes sharing the path.

    Every write is an atomic 0600 replace. The file holds scalars that open the corresponding
    mints: treat it like a view key, and back it up together with its ``.counter``. A malformed
    file raises rather than silently restarting the sequence.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).expanduser()
        self.counter_path = self.path.with_name(self.path.name + ".counter")
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def __repr__(self) -> str:
        return f"FilePrivateMintIdentityStore({str(self.path)!r})"

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with counter_lock(self.lock_path):
            yield

    def load(self) -> list[PrivateMintIdentity]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise ConfigurationError(_INVALID_RECORD) from exc
        records = data.get("identities") if isinstance(data, dict) else None
        if not isinstance(records, list):
            raise ConfigurationError(_INVALID_RECORD)
        return [PrivateMintIdentity.from_dict(record) for record in records]

    def save(self, identities: list[PrivateMintIdentity]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps({"version": _FILE_VERSION, "identities": [i.to_dict() for i in identities]}, indent=2)
        atomic_write_text(self.path.parent, self.path, text)

    def _read_counter(self) -> int:
        if not self.counter_path.exists():
            return 0
        text = self.counter_path.read_text(encoding="utf-8").strip()
        if not text.isdigit():
            raise ConfigurationError(
                "Private mint identity counter file is unreadable; restore it (and the identities file) "
                "from a backup before reserving a private mint")
        return int(text)

    def allocate_counter(self, minimum: int) -> int:
        value = max(self._read_counter(), minimum)
        if value > MAX_COUNTER:
            raise ConfigurationError(_EXHAUSTED)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path.parent, self.counter_path, str(value + 1))
        return value


# ── reservation and recovery ─────────────────────────────────────────────────

_process_locks: "weakref.WeakKeyDictionary[Any, threading.Lock]" = weakref.WeakKeyDictionary()
_process_locks_by_id: dict[int, threading.Lock] = {}
_registry_lock = threading.Lock()


def _process_lock(store: Any) -> threading.Lock:
    with _registry_lock:
        try:
            return _process_locks.setdefault(store, threading.Lock())
        except TypeError:                                    # not weak-referenceable
            return _process_locks_by_id.setdefault(id(store), threading.Lock())


def _exclusive(store: Any) -> ContextManager[None]:
    exclusive: Any = getattr(store, "exclusive", None)
    return cast(ContextManager[None], exclusive()) if callable(exclusive) else nullcontext()


def _next_counter(identities: list[PrivateMintIdentity]) -> int:
    minimum = max((identity.counter + 1 for identity in identities), default=0)
    if minimum > MAX_COUNTER:
        raise ConfigurationError(_EXHAUSTED)
    return minimum


def reserve_identity(store: PrivateMintIdentityStore, *, view_key: Any, recipient: str, network: str,
                     program: str = DEFAULT_WRAPPER_PROGRAM) -> PrivateMintIdentity:
    """Reserve the next identity for one private deposit: consume a counter, derive, save, return.

    The counter is consumed before derivation, so a failed derivation or save leaves a gap rather
    than a reuse. One process-wide lock per store object plus the store's own ``exclusive()``
    (when it has one) close the races within and across processes.
    """
    with _process_lock(store), _exclusive(store):
        identities = store.load()
        counter = store.allocate_counter(_next_counter(identities))
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < _next_counter(identities):
            raise ConfigurationError("Private mint identity store returned an invalid counter allocation")
        if counter > MAX_COUNTER:
            raise ConfigurationError(_EXHAUSTED)
        secret_nonce = derive_secret_nonce(view_key, counter, network, program)
        identity = PrivateMintIdentity(counter=counter, recipient=recipient, secret_nonce=secret_nonce,
                                       address_commitment=derive_address_commitment(recipient, secret_nonce, network))
        store.save([*identities, identity])
        return identity


def peek_secret_nonce(store: PrivateMintIdentityStore, *, view_key: Any, recipient: str, network: str,
                      program: str = DEFAULT_WRAPPER_PROGRAM) -> str:
    """The scalar the next :func:`reserve_identity` would produce, without consuming anything.

    Indicative only (another reservation may land first); quotes use it so their hook data matches
    what execute will deposit in the common case.
    """
    del recipient                                            # the nonce does not depend on it
    with _process_lock(store), _exclusive(store):
        identities = store.load()
        minimum = _next_counter(identities)
        counter = max(minimum, store._read_counter()) if isinstance(store, FilePrivateMintIdentityStore) else minimum
    if counter > MAX_COUNTER:
        raise ConfigurationError(_EXHAUSTED)
    return derive_secret_nonce(view_key, counter, network, program)


def find_identity(store: PrivateMintIdentityStore, address_commitment: str) -> PrivateMintIdentity | None:
    """The stored identity behind a public commitment, or ``None`` when this store never issued it."""
    for identity in store.load():
        if identity.address_commitment == address_commitment:
            return identity
    return None


def recover_identity(store: PrivateMintIdentityStore, *, view_key: Any, recipient: str, address_commitment: str,
                     network: str, program: str = DEFAULT_WRAPPER_PROGRAM,
                     scan_limit: int = DEFAULT_SCAN_LIMIT) -> PrivateMintIdentity:
    """The identity behind *address_commitment*: from the store, or re-derived by counting up.

    A stored hit must name *recipient*. Otherwise counters ``0..scan_limit-1`` are derived with the
    account's view key and *recipient* until one commits to the same value; the hit is saved back
    so the next look-up is immediate. Losing the identities file therefore costs a scan, not the
    funds.
    """
    stored = find_identity(store, address_commitment)
    if stored is not None:
        if stored.recipient != recipient:
            raise ConfigurationError(
                "The stored private mint identity for this deposit names a different recipient than the "
                "transfer; refusing to mint under it")
        return stored
    for counter in range(min(scan_limit, MAX_COUNTER + 1)):
        secret_nonce = derive_secret_nonce(view_key, counter, network, program)
        if derive_address_commitment(recipient, secret_nonce, network) == address_commitment:
            identity = PrivateMintIdentity(counter=counter, recipient=recipient, secret_nonce=secret_nonce,
                                           address_commitment=address_commitment)
            with _process_lock(store), _exclusive(store):
                current = store.load()
                if find_identity(store, address_commitment) is None:
                    store.save([*current, identity])
            return identity
    raise ConfigurationError(
        f"No private mint identity for this deposit's commitment was found in the store or among the first "
        f"{scan_limit} counters derived from the account's view key. A deposit made with an explicit "
        "secret_nonce can only be completed with that same secret_nonce; otherwise raise scan_limit or "
        "restore the identities file that reserved it")


# ── the resolver every verb calls ────────────────────────────────────────────

def default_identity_store(checkpoints: Any) -> PrivateMintIdentityStore | None:
    """The store a ``Bridge`` binds when none is given: a hidden file inside a ``FileCheckpointStore``
    directory (journal metadata, skipped by its ``*.json`` listing), or none at all."""
    from .checkpoint import FileCheckpointStore
    if isinstance(checkpoints, FileCheckpointStore):
        return FilePrivateMintIdentityStore(checkpoints.directory / ".private-mint-identities.json")
    return None


def derivable_account(bridge: Any, recipient: str) -> Any | None:
    """The bridge's default account when it can derive for *recipient*: it exposes a view key and
    *recipient* is its own address. Derivation is never offered for another recipient — the
    identity would not be one that recipient's wallet could ever reproduce."""
    account = getattr(getattr(bridge, "aleo", None), "default_account", None)
    if not account or getattr(account, "view_key", None) is None:
        return None
    if str(getattr(account, "address", "")) != recipient:
        return None
    return account


def can_derive(bridge: Any, recipient: str) -> bool:
    """True when a private mint to *recipient* needs no ``secret_nonce`` on this bridge."""
    return getattr(bridge, "private_mint_identities", None) is not None and derivable_account(bridge, recipient) is not None


def _why_not(bridge: Any, recipient: str) -> str:
    account = getattr(getattr(bridge, "aleo", None), "default_account", None)
    if not account or getattr(account, "view_key", None) is None:
        return "the Aleo facade has no default account with a view key to derive from"
    if str(getattr(account, "address", "")) != recipient:
        return "the recipient is not the bridge's own Aleo account, and derived nonces apply only to its own mints"
    return "no identity store is bound to keep the derived nonce"


def resolve_secret_nonce(bridge: Any, *, recipient: str, program: str, address_commitment: str | None = None,
                         secret_nonce: str | None = None, placeholder: bool = False) -> str:
    """The scalar a private mint uses, from one of three sources in this order.

    1. An explicit *secret_nonce*: validated and returned; no store is read or written.
    2. Derivation for the bridge's own account with a store bound: a new identity is reserved when
       *address_commitment* is ``None`` (execute); otherwise the identity behind that commitment is
       looked up and, when the store lacks it, re-derived by counting up (resume / complete).
    3. Neither: ``"0scalar"`` when *placeholder* is set (a quote, where nothing is deposited), else a
       ``ConfigurationError`` naming both fixes.
    """
    if secret_nonce is not None:
        return encoding.validate_scalar(secret_nonce)
    store = getattr(bridge, "private_mint_identities", None)
    account = derivable_account(bridge, recipient)
    if store is None or account is None:
        if placeholder:
            return "0scalar"
        raise ConfigurationError(
            f"A private mint needs a secret_nonce, and this bridge cannot derive one: {_why_not(bridge, recipient)}. "
            "Either pass secret_nonce= (and keep it — complete() needs the same value), or let the bridge derive "
            "and keep it for its own account by binding a checkpoint directory (FileCheckpointStore) or "
            "private_mint_identities= to Bridge(...)")
    network = bridge.network
    if placeholder:
        return peek_secret_nonce(store, view_key=account.view_key, recipient=recipient, network=network, program=program)
    if address_commitment is None:
        return reserve_identity(store, view_key=account.view_key, recipient=recipient, network=network,
                                program=program).secret_nonce
    return recover_identity(store, view_key=account.view_key, recipient=recipient, address_commitment=address_commitment,
                            network=network, program=program).secret_nonce


__all__ = [
    "DEFAULT_SCAN_LIMIT", "DEFAULT_WRAPPER_PROGRAM", "MAX_COUNTER", "can_derive", "default_identity_store",
    "derivable_account", "resolve_secret_nonce",
    "SHIELDED_USDCX_DOMAIN_MAINNET", "SHIELDED_USDCX_DOMAIN_TESTNET",
    "FilePrivateMintIdentityStore", "MemoryPrivateMintIdentityStore", "PrivateMintIdentity",
    "PrivateMintIdentityStore", "commitment_from_hook_data", "derive_address_commitment",
    "derive_secret_nonce", "domain_field", "find_identity", "peek_secret_nonce", "recover_identity",
    "reserve_identity",
]
