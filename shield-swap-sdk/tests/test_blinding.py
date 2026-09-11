"""Blinded identity derivation vs golden vectors.

The v3 vectors come from the TS SDK (test/utils/blinding/identity.test.ts),
which pins the reference derivation; they still verify the ALGORITHM (the
domain separators are unchanged in the new deployment) via an explicit
program argument.  The shield_swap.aleo vectors pin the DEFAULT program's
derivation — the program address feeds both hashes, so every address changed
with the cutover.  The program's verify_blinded_address re-computes this
hash and rejects any deviation — the vectors must reproduce exactly."""
import pytest

from aleo_shield_swap.derivations import (
    DEFAULT_PROGRAM,
    BlindedIdentity,
    derive_blinded_address,
    derive_blinding_factor,
    next_blinded_identity,
)

VIEW_KEY_SCALAR = "334926304971763782347498121479281870911723639068413954564748091722770623877scalar"
SIGNER = "aleo1rhgdu77hgyqd3xjj8ucu3jj9r2krwz6mnzyd80gncr5fxcwlh5rsvzp9px"

# Pinned for shield_swap_v3.aleo by the TS SDK reference suite.
V3_VECTORS = [
    (0, "4588552248780721950435785476596782217652350429588181106944985529417784595808field",
     "aleo1tucdl48jvu54emu9atq3vf0rslwtdpze83zcc2jrc8zxema0r5gq3zd76l"),
    (1, "6996211042158127437642182917952771252908546914090630418129936449807650494378field",
     "aleo17gc56avc2x3dwj3mjazag8szl5skm8y4u5h6ep37kvl34cynrqyqm0cuj8"),
    (7, "4426391170839722244039367865632426610408126795108463201618230895243256084792field",
     "aleo1jjq9qtr2uv86pans7f7v3tgcesg0autqhhu2cp2eecfxhtv4acgskyz80k"),
]

# Derived once for shield_swap.aleo (same algorithm, new program address).
VECTORS = [
    (0, "1084832000575072863530983109046262857691989153364570676666410266416291033880field",
     "aleo15mstsvdtzqf5nw8rfzx8mrllwxt907amfpt8nx3p8cskj4wd3uxq4uywn9"),
    (1, "5141395481140237504655245554781696675111779262144509416611105050866132602799field",
     "aleo1kafjl7kvfh8dwtqdwgje2maw5ugkm63pph5qt5d503yt8spg3u8s38haku"),
    (2, "4321510616277470162720724734723311517413314117144969624035468325948186568504field",
     "aleo13rmacw9jk943wwg4s6t5yxcycvl3ud356l3cyzgav8hw9fkcjsqsgt4vy3"),
    (7, "3586646586411194490118647465634943263277589324082503715221872233042073645843field",
     "aleo1xfe9cwtftdg5fhkcmtjh4xnuuqjlwqgzk5hz762rf7zef088tqzqgvpkj6"),
]


def test_default_program_is_new_core():
    assert DEFAULT_PROGRAM == "shield_swap.aleo"


@pytest.mark.parametrize("counter,bf,ba", V3_VECTORS)
def test_v3_algorithm_vectors(counter, bf, ba):
    assert derive_blinding_factor(VIEW_KEY_SCALAR, counter,
                                  "shield_swap_v3.aleo") == bf
    assert derive_blinded_address(bf, SIGNER, "shield_swap_v3.aleo") == ba


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
    # Counters 0 and 1 are both used and max_scan=2 stops the scan there —
    # when galloping past the window is disabled.
    used = {VECTORS[0][2], VECTORS[1][2]}
    with pytest.raises(ValueError, match="No unused blinded address"):
        next_blinded_identity(_StubAleo(used=used), _StubAccount(), max_scan=2,
                              gallop=False)


def _used_through(n):
    """Blinded addresses of counters 0..n-1 for the vector account."""
    out = set()
    for c in range(n):
        bf = derive_blinding_factor(VIEW_KEY_SCALAR, c)
        out.add(derive_blinded_address(bf, SIGNER))
    return out


def test_next_blinded_identity_gallops_past_a_long_used_run():
    """An account that has swapped more times than the linear window (the
    e2e account is past 64) must still find a free counter — and in O(log n)
    probes, not one per used counter.  Live failure 2026-09-03:
    'No unused blinded address in counters [0, 64)'."""
    used = _used_through(300)
    stub = _StubAleo(used=used)
    probes = []
    inner = stub.programs.get("x").mapping("used_blinded_addresses").get
    stub.programs.get("x").mapping("used_blinded_addresses").get = \
        lambda key: probes.append(key) or inner(key)
    ident = next_blinded_identity(stub, _StubAccount(), max_scan=8)
    assert ident.counter == 300
    assert ident.blinded_address not in used
    assert len(probes) < 8 + 2 * 12          # window + gallop + bisection, not 300


def test_next_blinded_identity_gallop_takes_a_gap_not_just_the_end():
    # Any unused counter is valid: a gap left by a failed swap is fine to reuse.
    used = _used_through(100) - {derive_blinded_address(
        derive_blinding_factor(VIEW_KEY_SCALAR, 40), SIGNER)}
    ident = next_blinded_identity(_StubAleo(used=used), _StubAccount(), max_scan=8)
    assert ident.blinded_address not in used


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

    for counter, bf, ba in V3_VECTORS:
        ident = blinded_identity_at(_Aleo(), _Acct(), "shield_swap_v3.aleo",
                                    counter)
        assert (ident.counter, ident.blinding_factor,
                ident.blinded_address) == (counter, bf, ba)
