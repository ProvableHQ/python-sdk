"""Circle xReserve attester — read-only HTTP (port of veil ``getAttestation``). Never signs or moves funds."""
from __future__ import annotations

from typing import Any

import requests

from . import encoding as enc
from ._keccak import keccak256
from .errors import AttestationError, ConfigurationError, InvalidRecipientError
from .types import Attestation


class CircleClient:
    """``GET {base_url}/{messageHash}``: 404 → ``None`` (pending); 200 → a verified :class:`Attestation`."""

    def __init__(self, base_url: str, session: Any = None, timeout: float = 30.0) -> None:
        if not isinstance(base_url, str) or not base_url.startswith("https://"):
            raise ConfigurationError(f"Circle attestation base URL must start with https://, got {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = session if session is not None else requests.Session()

    def get_attestation(self, message_hash_hex: str) -> Attestation | None:
        try:
            digest = enc.hex_to_bytes(message_hash_hex, 32)
        except (ValueError, InvalidRecipientError) as exc:
            raise AttestationError(f"Circle attestation lookup needs a 32-byte message hash, got {message_hash_hex!r}") from exc
        try:
            response = self._session.get(f"{self.base_url}/{enc.to_hex(digest)}", timeout=self.timeout)
        except requests.exceptions.RequestException as exc:
            raise AttestationError(f"Circle attester request failed: {exc}") from exc
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise AttestationError(f"Circle attester request failed with HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise AttestationError(f"Circle attester request failed: {exc}") from exc
        value = body.get("attestation") if isinstance(body, dict) else None
        if not isinstance(value, dict):
            raise AttestationError("Circle attester returned an invalid response (no attestation object)")
        try:
            payload = enc.hex_to_bytes(value["payload"], enc.XRESERVE_PAYLOAD_BYTES)
            signature = enc.hex_to_bytes(value["attestation"], enc.HOOK_DATA_BYTES)
            echoed = enc.hex_to_bytes(value["messageHash"], 32)
        except (KeyError, TypeError, ValueError, InvalidRecipientError) as exc:
            raise AttestationError("Circle attester returned an invalid response (payload/attestation/messageHash)") from exc
        if echoed != digest:
            raise AttestationError("Circle attester echoed a different message hash than requested")
        if keccak256(payload) != digest:
            raise AttestationError("Circle attestation payload does not match the requested message hash")
        return Attestation(payload=payload, message_hash=digest, attestation=signature, status="complete")


__all__ = ["CircleClient"]
