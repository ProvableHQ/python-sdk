"""Aleo — Web3.py-style facade client (F1 skeleton).

This module provides :class:`Aleo`, the central client object.  It is
intentionally minimal in F1: it wires a provider to a network client, exposes
helpers and escape hatches, and defers account/programs/records/verb-ladder
to later phases.
"""
from __future__ import annotations

from typing import Any

from .._client_common import AleoNetworkError
from .._facade_common import credits_to_microcredits, microcredits_to_credits
from .provider import HTTPProvider

# AleoNetworkClient imported at runtime (avoid circular at module-level)
# to keep the TYPE_CHECKING guard unnecessary for the property return type.


class Aleo:
    """Web3.py-style client for the Aleo blockchain.

    Construct with an :class:`~aleo.facade.provider.HTTPProvider`::

        from aleo import Aleo
        aleo = Aleo(Aleo.HTTPProvider("https://api.provable.com/v2"))

    Parameters
    ----------
    provider:
        A :class:`~aleo.facade.provider.HTTPProvider` instance that
        configures the underlying network connection.
    """

    # Expose HTTPProvider as a nested class attribute so callers can write
    # ``Aleo.HTTPProvider(...)`` without a separate import.
    HTTPProvider = HTTPProvider

    def __init__(self, provider: object) -> None:
        if not isinstance(provider, HTTPProvider):
            raise TypeError(
                f"provider must be an HTTPProvider, got {type(provider).__name__}"
            )
        self._provider: HTTPProvider = provider
        # Access the build helper via name to avoid pyright's private-access error;
        # _build_client is intentionally package-private (single leading underscore).
        self._client: Any = provider._build_client()  # pyright: ignore[reportPrivateUsage]
        self._process: Any = None  # lazy — loaded on first access
        self._default_account: Any = None
        # Namespaced modules — constructed eagerly (they hold no state of their own)
        from .account import AccountModule
        self.account: AccountModule = AccountModule(self)

    # ── Escape hatches ─────────────────────────────────────────────────────

    @property
    def provider(self) -> HTTPProvider:
        """The :class:`~aleo.facade.provider.HTTPProvider` used to build this client."""
        return self._provider

    @property
    def network_client(self) -> Any:
        """The raw :class:`~aleo.network_client.AleoNetworkClient`."""
        return self._client

    @property
    def process(self) -> Any:
        """Lazily-loaded :class:`~aleo.mainnet.Process` (or testnet equivalent).

        The ``Process`` is not loaded until this property is first accessed so
        that constructing an :class:`Aleo` client is instantaneous even when
        the SRS keys are not pre-cached.
        """
        if self._process is None:
            net = self._provider.network
            if net == "testnet":
                from ..testnet import Process  # type: ignore[attr-defined]
            else:
                from ..mainnet import Process  # type: ignore[attr-defined]
            self._process = Process.load()
        return self._process

    # ── Default account ────────────────────────────────────────────────────

    @property
    def default_account(self) -> Any:
        """The default account used when a verb omits a signer."""
        return self._default_account

    @default_account.setter
    def default_account(self, account: Any) -> None:
        self._default_account = account

    # ── Network identity ───────────────────────────────────────────────────

    @property
    def network_id(self) -> int:
        """Numeric network identifier (0 = mainnet, 1 = testnet).

        Analog of ``web3.eth.chain_id``.
        """
        net = self._provider.network
        if net == "testnet":
            from ..testnet import Network  # type: ignore[attr-defined]
        else:
            from ..mainnet import Network  # type: ignore[attr-defined]
        return int(Network.id())

    @property
    def network_name(self) -> str:
        """Human-readable network name string.

        Returns ``"mainnet"`` or ``"testnet"`` (the normalised provider
        value, not the full snarkvm network display name).
        """
        return self._provider.network

    # ── Connectivity ───────────────────────────────────────────────────────

    def is_connected(self) -> bool:
        """Return ``True`` if the node is reachable.

        Performs a lightweight ``get_latest_height`` call and returns
        ``False`` on any error rather than propagating it.
        """
        try:
            self._client.get_latest_height()
            return True
        except Exception:
            # Reachability probe: any failure (network, socket, SSL, DNS, HTTP)
            # means "not connected" — matches web3.py's is_connected semantics.
            return False

    # ── Balance ────────────────────────────────────────────────────────────

    def get_balance(self, address: str) -> int:
        """Return the public credits balance for *address* in microcredits.

        Queries the ``credits.aleo`` ``account`` mapping.  Returns ``0`` when
        the address has no on-chain balance or when the mapping value is absent.

        Parameters
        ----------
        address:
            An Aleo address string (``aleo1…``).
        """
        try:
            raw: Any = self._client.get_program_mapping_value(
                "credits.aleo", "account", address
            )
            if raw is None:
                return 0
            # The mapping value may come back as a quoted string or plain int
            val = str(raw).strip().strip('"')
            if not val or val == "null":
                return 0
            # Strip suffix like "u64" if present
            if val.endswith("u64"):
                val = val[:-3]
            return int(val)
        except (AleoNetworkError, ValueError):
            return 0
        except Exception:
            return 0

    # ── Unit conversions ───────────────────────────────────────────────────

    def to_microcredits(self, credits: float | int) -> int:
        """Convert a credits amount to integer microcredits.

        ``1 credit == 1_000_000 microcredits``

        Parameters
        ----------
        credits:
            Credits amount as a float or integer (e.g. ``1.5``).
        """
        return credits_to_microcredits(credits)

    def from_microcredits(self, microcredits: int) -> float:
        """Convert an integer microcredits amount to credits.

        Parameters
        ----------
        microcredits:
            Integer microcredits (e.g. ``1_500_000``).
        """
        return microcredits_to_credits(microcredits)

    # ── Address validation ─────────────────────────────────────────────────

    def is_valid_address(self, s: str) -> bool:
        """Return ``True`` if *s* is a valid Aleo address.

        Wraps the network module's ``Address.is_valid`` class method.

        Parameters
        ----------
        s:
            The candidate address string.
        """
        try:
            net = self._provider.network
            if net == "testnet":
                from ..testnet import Address  # type: ignore[attr-defined]
            else:
                from ..mainnet import Address  # type: ignore[attr-defined]
            return bool(Address.is_valid(s))  # pyright: ignore[reportAttributeAccessIssue]
        except Exception:
            return False

    # ── Repr ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"Aleo(provider={self._provider!r})"


__all__ = ["Aleo"]
