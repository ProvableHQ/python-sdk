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

from dataclasses import dataclass
from typing import Any

from . import encoding
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


__all__ = [
    "DEFAULT_WRAPPER_PROGRAM", "MAX_COUNTER", "SHIELDED_USDCX_DOMAIN_MAINNET", "SHIELDED_USDCX_DOMAIN_TESTNET",
    "PrivateMintIdentity", "commitment_from_hook_data", "derive_address_commitment", "derive_secret_nonce",
    "domain_field",
]
