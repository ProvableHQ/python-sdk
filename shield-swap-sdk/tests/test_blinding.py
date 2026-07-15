"""Blinded identity derivation vs golden vectors from the TS SDK
(test/utils/blinding/identity.test.ts), which pin the reference derivation in
amm-v3-tests.  The program's verify_blinded_address re-computes this hash and
rejects any deviation — the vectors must reproduce exactly."""
import pytest

from aleo_shield_swap.derivations import (
    BlindedIdentity,
    derive_blinded_address,
    derive_blinding_factor,
    next_blinded_identity,
)

VIEW_KEY_SCALAR = "334926304971763782347498121479281870911723639068413954564748091722770623877scalar"
SIGNER = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"
VECTORS = [
    (0, "4588552248780721950435785476596782217652350429588181106944985529417784595808field",
     "aleo1tucdl48jvu54emu9atq3vf0rslwtdpze83zcc2jrc8zxema0r5gq3zd76l"),
    (1, "6996211042158127437642182917952771252908546914090630418129936449807650494378field",
     "aleo17gc56avc2x3dwj3mjazag8szl5skm8y4u5h6ep37kvl34cynrqyqm0cuj8"),
    (7, "4426391170839722244039367865632426610408126795108463201618230895243256084792field",
     "aleo1jjq9qtr2uv86pans7f7v3tgcesg0autqhhu2cp2eecfxhtv4acgskyz80k"),
]


@pytest.mark.parametrize("counter,bf,ba", VECTORS)
def test_golden_vectors(counter, bf, ba):
    assert derive_blinding_factor(VIEW_KEY_SCALAR, counter) == bf
    assert derive_blinded_address(bf, SIGNER) == ba


def test_deterministic():
    a = derive_blinding_factor(VIEW_KEY_SCALAR, 0)
    b = derive_blinding_factor(VIEW_KEY_SCALAR, 0)
    assert a == b
    assert derive_blinded_address(a, SIGNER) == derive_blinded_address(b, SIGNER)


class _Mapping:
    def __init__(self, used):
        self.used = used

    def get(self, key):
        return "true" if key in self.used else None


class _StubAleo:
    network_name = "testnet"

    def __init__(self, used):
        mapping = _Mapping(used)

        class _Prog:
            def mapping(self, name):
                assert name == "used_blinded_addresses"
                return mapping

        class _Programs:
            def get(self, pid):
                return _Prog()

        self.programs = _Programs()


class _StubAccount:
    """Account stub carrying the vector view-key scalar + signer address."""

    class _VK:
        def to_scalar(self):
            return VIEW_KEY_SCALAR

    view_key = _VK()
    address = SIGNER


def test_next_blinded_identity_returns_first_free():
    ident = next_blinded_identity(_StubAleo(used=set()), _StubAccount())
    assert ident == BlindedIdentity(0, VECTORS[0][1], VECTORS[0][2])


def test_next_blinded_identity_skips_used():
    used = {VECTORS[0][2], VECTORS[1][2]}
    ident = next_blinded_identity(_StubAleo(used=used), _StubAccount())
    assert ident.counter == 2
    assert ident.blinded_address not in used


def test_next_blinded_identity_max_scan():
    # Counters 0 and 1 are both used and max_scan=2 stops the scan there.
    used = {VECTORS[0][2], VECTORS[1][2]}
    with pytest.raises(ValueError, match="No unused blinded address"):
        next_blinded_identity(_StubAleo(used=used), _StubAccount(), max_scan=2)


def test_blinded_identity_at_exact_counter_no_probe():
    from aleo_shield_swap.derivations import blinded_identity_at

    class _VK:
        def to_scalar(self):
            return VIEW_KEY_SCALAR

    class _Acct:
        view_key = _VK()
        address = SIGNER

    class _Aleo:
        network_name = "testnet"

        @property
        def programs(self):          # any chain probe is a bug
            raise AssertionError("blinded_identity_at must not touch the chain")

    for counter, bf, ba in VECTORS:
        ident = blinded_identity_at(_Aleo(), _Acct(), "shield_swap_v3.aleo",
                                    counter)
        assert (ident.counter, ident.blinding_factor,
                ident.blinded_address) == (counter, bf, ba)
