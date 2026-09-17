import pytest

from aleo_bridge import encoding as enc
from aleo_bridge.errors import AttestationError, ConfigurationError, InvalidAmountError, InvalidRecipientError

RECIPIENT = "aleo1kypwp5m7qtk9mwazgcpg0tq8aal23mnrvwfvug65qgcg9xvsrqgspyjm6n"
RECIPIENT_BYTES32 = "b102e0d37e02ec5dbba2460287ac07ef7ea8ee636392ce235402308299901811"
EVM1 = "0x0000000000000000000000000000000000000001"
WRAPPER = "shielded_usdcx_wrapper.aleo"
MESSAGE_ID = "0xc7c2c763ef846ff1583d9222d8ecbfc56da2e0cdcc9a63bc4bde51467644794d"


# ── bech32m (veil test/utils/xreserve.test.ts) ────────────────────────────────

def test_aleo_address_round_trip_and_checksum():
    assert enc.aleo_address_to_bytes32(RECIPIENT).hex() == RECIPIENT_BYTES32
    assert enc.bytes32_to_aleo_address(bytes.fromhex(RECIPIENT_BYTES32)) == RECIPIENT
    with pytest.raises(InvalidRecipientError, match="Invalid Aleo recipient"):
        enc.aleo_address_to_bytes32(RECIPIENT[:-1] + "q")
    with pytest.raises(InvalidRecipientError):
        enc.aleo_address_to_bytes32("aleo1short")
    with pytest.raises(InvalidRecipientError):
        enc.aleo_address_to_bytes32("aleo1" + "b" * 58)  # 'b' is not in the alphabet
    with pytest.raises(InvalidRecipientError, match="32-byte Aleo recipient"):
        enc.bytes32_to_aleo_address(b"\x01")


def test_bech32_decoder_agrees_with_the_bindings():
    # Address.from_string(...).to_bits_le() is a 253-bit field-element encoding (no fixed byte
    # width) and Plaintext.from_string(...).to_bits_le() prefixes a literal-type discriminant
    # (279 bits total here) — neither packs cleanly into 32 bytes. to_bytes_le() is the API that
    # actually returns the 32-byte payload, so that's what the bech32m payload is checked against.
    from aleo import mainnet as net
    packed = bytes(net.Address.from_string(RECIPIENT).to_bytes_le())
    assert len(packed) == 32
    assert enc.aleo_address_to_bytes32(RECIPIENT) == packed


# ── EVM addresses ─────────────────────────────────────────────────────────────

def test_evm_address_to_bytes32_left_pads_and_checks_eip55():
    assert enc.evm_address_to_bytes32(EVM1).hex() == "00" * 31 + "01"
    assert enc.evm_address_to_bytes32("0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238").hex() == \
        "0000000000000000000000001c7d4b196cb0c7b01d743fbc6116a902379c7238"
    assert enc.is_evm_address("0x1c7d4b196cb0c7b01d743fbc6116a902379c7238")  # all-lowercase is fine
    assert enc.to_checksum_address("0x1c7d4b196cb0c7b01d743fbc6116a902379c7238") == \
        "0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238"
    with pytest.raises(InvalidRecipientError, match="Invalid Ethereum recipient"):
        enc.evm_address_to_bytes32("0x1234")
    with pytest.raises(InvalidRecipientError):  # mixed case with a wrong checksum
        enc.evm_address_to_bytes32("0x1C7D4B196Cb0C7B01d743Fbc6116a902379C7238")


# ── Hyperlane limbs (veil test/utils/hyperlane.test.ts) ───────────────────────

def test_evm_limbs():
    assert enc.evm_address_to_hyperlane_recipient("0x1e196d0a7d8189054c4db744ab3340c3f1c68b19") == (
        13858749752514421660238621190289096704, 33956464229475118999063216025592496509)
    assert enc.evm_address_to_hyperlane_recipient(EVM1) == (0, 1329227995784915872903807060280344576)
    for bad in ("0x1234", "0x" + "11" * 32):
        with pytest.raises(InvalidRecipientError, match="Invalid Ethereum Hyperlane recipient"):
            enc.evm_address_to_hyperlane_recipient(bad)


def test_solana_limbs():
    assert enc.solana_address_to_hyperlane_recipient("8YGT2pZwyZe94qBpGzWfY2TMEVcwaQ1bXAE7YAgpUaM7") == (
        127878782877948140186055645953777992816, 163261512394675613100746600600636171918)
    assert enc.solana_address_to_hyperlane_recipient("11111111111111111111111111111111") == (0, 0)
    for bad in ("not-base58!", "1111"):
        with pytest.raises(InvalidRecipientError, match="Invalid Solana Hyperlane recipient"):
            enc.solana_address_to_hyperlane_recipient(bad)


def test_limbs_and_literals():
    assert enc.bytes32_to_u128_limbs(bytes(31) + b"\x01") == (0, 1 << 120)
    with pytest.raises(InvalidRecipientError):
        enc.bytes32_to_u128_limbs(bytes(31))
    assert enc.u128_pair_literal((0, 1329227995784915872903807060280344576)) == \
        "[0u128, 1329227995784915872903807060280344576u128]"
    assert enc.u8_array_literal(bytes.fromhex("00ff")) == "[0u8,255u8]"
    assert enc.u8_array_literal(bytes(31) + b"\x01") == "[" + ",".join(["0u8"] * 31 + ["1u8"]) + "]"
    assert enc.hex_to_bytes("0x00ff", 2) == b"\x00\xff"
    assert enc.hex_to_bytes(b"\x00\xff") == b"\x00\xff"
    with pytest.raises(InvalidRecipientError, match="32 bytes"):
        enc.hex_to_bytes("0x00ff", 32)
    assert enc.to_hex(b"\x00\xff") == "0x00ff"


def test_hyperlane_delivery_key_vector():
    # brief §3.7 vector
    key = enc.hyperlane_delivery_key(enc.hex_to_bytes(MESSAGE_ID, 32))
    assert key == "{ id: [262854447642257427123071959211115528903u128, 102980212169860384794748804418278302317u128] }"
    with pytest.raises(InvalidRecipientError):
        enc.hyperlane_delivery_key(b"\x00")


# ── xReserve wire format (veil test/utils/xreserve.test.ts) ───────────────────

def _payload(hook: bytes) -> tuple[bytes, bytes]:
    nonce = enc.xreserve_deposit_nonce(0, bytes.fromhex("12" * 32), 4)
    payload = enc.xreserve_deposit_payload(
        amount=1_000_000, remote_domain=10002,
        remote_token=bytes.fromhex("b143ed52c774cd1d4a519d0e796f15916be5a9e1d45edcd9852dd23f68f53401"),
        remote_recipient=enc.aleo_address_to_bytes32(RECIPIENT),
        local_token="0x1c7D4B196Cb0C7B01d743Fbc6116a902379C7238", depositor=EVM1,
        max_fee=100_000, nonce=nonce, hook_data=hook)
    return nonce, payload


def test_deposit_nonce_payload_and_hash_layout():
    nonce, payload = _payload(enc.xreserve_hook_data("public", RECIPIENT, "testnet"))
    assert len(nonce) == 32 and len(payload) == 305
    assert payload[0:8].hex() == "5a2e0acd00000001"
    assert payload[8:40] == (1_000_000).to_bytes(32, "big")
    assert payload[40:44] == (10002).to_bytes(4, "big")
    assert payload[44:76].hex() == "b143ed52c774cd1d4a519d0e796f15916be5a9e1d45edcd9852dd23f68f53401"
    assert payload[76:108].hex() == RECIPIENT_BYTES32
    assert payload[108:140].hex() == "0000000000000000000000001c7d4b196cb0c7b01d743fbc6116a902379c7238"
    assert payload[140:172].hex() == "00" * 31 + "01"
    assert payload[172:204] == (100_000).to_bytes(32, "big")
    assert payload[204:236] == nonce
    assert payload[236:240].hex() == "00000041"
    assert payload[240:305] == bytes(65)
    assert len(enc.xreserve_message_hash(payload)) == 32
    assert enc.xreserve_nonce_from_payload(payload) == nonce


def test_deposit_nonce_matches_abi_encoding_via_web3():
    web3 = pytest.importorskip("web3")
    from eth_abi import encode
    tx_hash = bytes.fromhex("12" * 32)
    expected = bytes(web3.Web3.keccak(encode(["uint32"], [0]) + tx_hash + encode(["uint256"], [4])))
    assert enc.xreserve_deposit_nonce(0, tx_hash, 4) == expected
    assert enc.xreserve_deposit_nonce(0, "0x" + "12" * 32, 4) == expected  # hex accepted too


def test_deposit_payload_rejects_bad_widths():
    good = dict(amount=1, remote_domain=1, remote_token=bytes(32), remote_recipient=bytes(32),
                local_token=EVM1, depositor=EVM1, max_fee=0, nonce=bytes(32), hook_data=bytes(65))
    with pytest.raises(InvalidRecipientError, match="remote_token must contain 32 bytes"):
        enc.xreserve_deposit_payload(**{**good, "remote_token": bytes(31)})
    with pytest.raises(InvalidRecipientError, match="hook_data must contain 65 bytes"):
        enc.xreserve_deposit_payload(**{**good, "hook_data": bytes(64)})
    with pytest.raises(InvalidAmountError, match="does not fit"):
        enc.xreserve_deposit_payload(**{**good, "remote_domain": 1 << 32})
    with pytest.raises(InvalidRecipientError):
        enc.xreserve_deposit_payload(**{**good, "depositor": "0x1234"})


def test_deposit_nonce_bounds_source_domain_to_uint32():
    tx_hash = bytes.fromhex("12" * 32)
    assert enc.xreserve_deposit_nonce((1 << 32) - 1, tx_hash, 0)          # max uint32 is fine
    with pytest.raises(InvalidAmountError, match="does not fit"):
        enc.xreserve_deposit_nonce(1 << 32, tx_hash, 0)


def test_nonce_from_payload_rejects_bad_layout():
    with pytest.raises(AttestationError, match="invalid deposit layout"):
        enc.xreserve_nonce_from_payload(bytes(305))
    with pytest.raises(AttestationError):
        enc.xreserve_nonce_from_payload(bytes(304))


# ── hook data (BHP256 through the real bindings) ──────────────────────────────

def test_hook_data_public_and_record_are_pure():
    assert enc.xreserve_hook_data("public", RECIPIENT, "testnet") == bytes(65)
    assert enc.xreserve_hook_data("record", RECIPIENT, "testnet") == b"\x01" + bytes(64)


def test_hook_data_private_commits_recipient_with_selected_scalar():
    default = enc.xreserve_hook_data("private", RECIPIENT, "testnet")
    assert len(default) == 65 and default[0] == 2 and default[33:] == bytes(32)
    assert default == enc.xreserve_hook_data("private", RECIPIENT, "testnet", "0scalar")
    custom = enc.xreserve_hook_data("private", RECIPIENT, "testnet", "7scalar")
    assert custom[0] == 2 and custom != default
    # BHP256 is curve-level, not network-level: mainnet and testnet bindings agree.
    assert enc.xreserve_hook_data("private", RECIPIENT, "mainnet") == default
    # Independent recomputation through the bindings.
    from aleo import testnet as net
    commitment = net.BHP256().commit(net.Plaintext.from_string(RECIPIENT).to_bits_le(),
                                     net.Scalar.from_string("7scalar"))
    assert custom[1:33] == bytes(commitment.to_bytes_le())


def test_hook_data_private_pinned_vector():
    # Pinned from the bindings on 2026-09-03 (RECIPIENT, 0scalar). If this fails while the previous
    # test passes, the bindings' BHP256 output changed — investigate, do not re-pin blindly.
    assert enc.xreserve_hook_data("private", RECIPIENT, "mainnet")[1:33].hex() == \
        "dd46b467d619a9628e58a71ebf24873c777f93ae2b7ac5dee3db7282ecef8d10"


def test_hook_data_validation():
    with pytest.raises(ConfigurationError, match="mint mode"):
        enc.xreserve_hook_data("shielded", RECIPIENT, "mainnet")
    with pytest.raises(ConfigurationError, match="scalar"):
        enc.xreserve_hook_data("private", RECIPIENT, "mainnet", "not-a-scalar")
    with pytest.raises(ConfigurationError, match="network"):
        enc.xreserve_hook_data("private", RECIPIENT, "devnet")
    with pytest.raises(InvalidRecipientError):
        enc.xreserve_hook_data("private", "aleo1short", "mainnet")


# ── program address ───────────────────────────────────────────────────────────

def test_program_address_is_a_valid_address_and_matches_bindings():
    from aleo import mainnet as net
    addr = enc.aleo_program_address(WRAPPER, "mainnet")
    assert addr == str(net.Address.from_program_id(WRAPPER))
    assert len(enc.aleo_address_to_bytes32(addr)) == 32
    # Pinned 2026-09-03 from the bindings; see note on the hook-data pin above.
    assert addr == "aleo183r3zgsr57fwtgk5duzeq9kqdkpmmtfj4k5469ddvm3tcfhhls9szktw82"
    with pytest.raises(ConfigurationError, match="network"):
        enc.aleo_program_address(WRAPPER, "devnet")
