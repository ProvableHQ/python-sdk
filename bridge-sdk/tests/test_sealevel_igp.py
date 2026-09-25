import pytest

from aleo_bridge import _sealevel as sl
from aleo_bridge.errors import BridgeError
from tests.fakes.sealevel_fixtures import (
    ALEO_MAINNET_DOMAIN,
    DESTINATION_GAS_AMOUNT,
    EXPECTED_IGP_PAYMENT_LAMPORTS,
    IGP,
    igp_account_data,
)

HEADER_NO_OWNER = 1 + 8 + 1 + 32 + 1 + 32 + 4      # initialized, disc, bump, salt, owner=None, beneficiary, count


def synthetic_igp(domain: int, *, tag: int = 0, rate: int = 1, price: int = 1, decimals: int = 9) -> bytes:
    data = bytearray(HEADER_NO_OWNER + sl.GAS_ORACLE_ENTRY_BYTES)
    data[0] = 1
    data[1:9] = sl.IGP_DISCRIMINATOR
    data[HEADER_NO_OWNER - 4:HEADER_NO_OWNER] = (1).to_bytes(4, "little")
    entry = HEADER_NO_OWNER
    data[entry:entry + 4] = domain.to_bytes(4, "little")
    data[entry + 4] = tag
    data[entry + 5:entry + 21] = rate.to_bytes(16, "little")
    data[entry + 21:entry + 37] = price.to_bytes(16, "little")
    data[entry + 37] = decimals
    return bytes(data)


def test_decode_recorded_inner_igp_account():
    account = sl.decode_igp_account(igp_account_data())
    assert account.bump == 255
    assert account.owner is not None and len(account.owner) in (43, 44)
    assert len(account.beneficiary) in (43, 44)
    assert account.gas_oracles[ALEO_MAINNET_DOMAIN] == sl.GasOracle(751_705_303_136, 83_169, 6)
    assert len(account.gas_oracles) + len(account.unsupported_oracles) == 42


def test_quote_reproduces_sealevel_notes_vector():
    assert sl.quote_igp_lamports(igp_account_data(), ALEO_MAINNET_DOMAIN, DESTINATION_GAS_AMOUNT) == EXPECTED_IGP_PAYMENT_LAMPORTS
    assert sl.igp_lamports(sl.GasOracle(751_705_303_136, 83_169, 6), 464_000) == 2_900_000


def test_igp_lamports_divides_when_token_decimals_exceed_nine():
    # dest_cost = 10^6 * 1 ; origin_cost = 10^6 * 10^19 / 10^19 = 10^6 ; decimals 12 → // 10^3
    assert sl.igp_lamports(sl.GasOracle(10 ** 19, 1, 12), 10 ** 6) == 1_000
    assert sl.igp_lamports(sl.GasOracle(10 ** 19, 1, 9), 10 ** 6) == 1_000_000


def test_quote_rejects_missing_domain_and_unknown_variant_tag():
    with pytest.raises(BridgeError, match="no gas-oracle entry for destination domain 999999999"):
        sl.quote_igp_lamports(igp_account_data(), 999_999_999, 1)
    with pytest.raises(BridgeError, match="unexpected GasOracle variant tag 7"):
        sl.quote_igp_lamports(synthetic_igp(42, tag=7), 42, 1)
    assert sl.quote_igp_lamports(synthetic_igp(42, rate=10 ** 19, price=5, decimals=9), 42, 3) == 15


def test_decode_rejects_malformed_layouts():
    with pytest.raises(BridgeError, match="not initialized"):
        sl.decode_igp_account(bytes(12))
    with pytest.raises(BridgeError, match="declared layout exceeds the supplied bytes"):
        sl.decode_igp_account(b"\x01" + sl.IGP_DISCRIMINATOR + bytes(3))
    with pytest.raises(BridgeError, match="discriminator"):
        sl.decode_igp_account(b"\x01" + b"WRONGDIS" + bytes(80))
    bad_owner = bytearray(80)
    bad_owner[0] = 1
    bad_owner[1:9] = sl.IGP_DISCRIMINATOR
    bad_owner[42] = 2
    with pytest.raises(BridgeError, match="owner option tag 2"):
        sl.decode_igp_account(bytes(bad_owner))
    assert IGP["address"] == "JAvHW21tYXE9dtdG83DReqU2b4LUexFuCbtJT5tF8X6M"
