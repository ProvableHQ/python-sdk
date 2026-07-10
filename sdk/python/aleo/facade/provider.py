"""HTTPProvider — configuration object that constructs an AleoNetworkClient.

Mirrors ``web3.HTTPProvider``: pass it to ``Aleo(provider)`` and the client
wires itself up from it.  A provider is a *config object*, not a connection;
it is safe to construct without a live network.
"""
from __future__ import annotations

from typing import Any

from ..network_client import AleoNetworkClient
from .._client_common import DEFAULT_HOST, DEFAULT_NETWORK

# AsyncAleoNetworkClient imported lazily to avoid pulling httpx at import time.

_VALID_NETWORKS = frozenset({"mainnet", "testnet"})


class HTTPProvider:
    """Configuration object for the Aleo client.

    Parameters
    ----------
    url:
        Versioned API root, e.g. ``"https://api.provable.com/v2"``.
    network:
        Network name — ``"mainnet"`` (default) or ``"testnet"``.
    api_key:
        Provable API key passed through to the underlying
        :class:`~aleo.network_client.AleoNetworkClient`.
    prover_uri:
        Base URI for the DPS prover (without network suffix).
    headers:
        Additional HTTP headers merged on top of the SDK defaults.
    transport:
        Optional callable ``(method, url, **kwargs) -> requests.Response``.
        Forwarded verbatim to :class:`~aleo.network_client.AleoNetworkClient`.
    """

    def __init__(
        self,
        url: str = DEFAULT_HOST,
        *,
        network: str = DEFAULT_NETWORK,
        api_key: str | None = None,
        prover_uri: str | None = None,
        headers: dict[str, str] | None = None,
        transport: Any = None,
    ) -> None:
        if network not in _VALID_NETWORKS:
            raise ValueError(
                f"Invalid network {network!r}. Must be one of: "
                + ", ".join(sorted(_VALID_NETWORKS))
            )
        self._url = url
        self._network = network
        self._api_key = api_key
        self._prover_uri = prover_uri
        self._headers = dict(headers) if headers else None
        self._transport = transport

    # ── Public read-only properties ────────────────────────────────────────

    @property
    def url(self) -> str:
        """The versioned API root URL."""
        return self._url

    @property
    def network(self) -> str:
        """Network name (``"mainnet"`` or ``"testnet"``)."""
        return self._network

    @property
    def api_key(self) -> str | None:
        """Provable API key, if set."""
        return self._api_key

    @property
    def prover_uri(self) -> str | None:
        """DPS prover URI, if set."""
        return self._prover_uri

    # ── Internal factory ───────────────────────────────────────────────────

    def _build_client(self) -> AleoNetworkClient:
        """Construct and return an :class:`~aleo.network_client.AleoNetworkClient`."""
        return AleoNetworkClient(
            self._url,
            network=self._network,
            api_key=self._api_key,
            prover_uri=self._prover_uri,
            headers=self._headers,
            transport=self._transport,
        )

    def _build_async_client(self) -> "Any":
        """Construct and return an :class:`~aleo.async_network_client.AsyncAleoNetworkClient`."""
        from ..async_network_client import AsyncAleoNetworkClient
        return AsyncAleoNetworkClient(
            self._url,
            network=self._network,
            api_key=self._api_key,
            prover_uri=self._prover_uri,
            headers=self._headers,
            transport=self._transport,
        )

    def __repr__(self) -> str:
        return (
            f"HTTPProvider(url={self._url!r}, network={self._network!r})"
        )
