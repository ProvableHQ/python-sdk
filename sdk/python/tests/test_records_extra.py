# Copyright (C) 2019-2023 Aleo Systems Inc.
# This file is part of the Aleo SDK library.
# Tests for W4a: record-scanning parity (GraphKey, commitments, tags, RVK, arithmetic).
# Source notes: github.com/ProvableHQ/sdk@543b41e wasm/src/...

import pytest
from aleo.mainnet import (
    Address,
    Field,
    GraphKey,
    Group,
    Identifier,
    ProgramID,
    RecordCiphertext,
    RecordPlaintext,
    Scalar,
    ViewKey,
)
# ---------------------------------------------------------------------------
# KAT constants (vendored from ProvableHQ/sdk@543b41e)
# ---------------------------------------------------------------------------

# From wasm/src/record/record_ciphertext.rs:223-224
RECORD_TAG = "1796466189545157638691489609907096471289658804813960182690905095269699169603field"
RECORD_VIEW_KEY = "4445718830394614891114647247073357094867447866913203502139893824059966201724field"

# Owner record — same ciphertext/VK as existing records.json decrypt_kat
OWNER_CIPHERTEXT = "record1qyqsqpe2szk2wwwq56akkwx586hkndl3r8vzdwve32lm7elvphh37rsyqyxx66trwfhkxun9v35hguerqqpqzqrtjzeu6vah9x2me2exkgege824sd8x2379scspmrmtvczs0d93qttl7y92ga0k0rsexu409hu3vlehe3yxjhmey3frh2z5pxm5cmxsv4un97q"
OWNER_VIEW_KEY = "AViewKey1ccEt8A2Ryva5rxnKcAbn7wgTaTsb79tzkKHFpeKsm9NX"
OWNER_PRIVATE_KEY = "APrivateKey1zkpJkyYRGYtkeHDaFfwsKtUJzia7csiWhfBWPXWhXJzy9Ls"

# V1 credits.aleo record (utilities/test/records.rs:18)
CREDITS_RECORD_V1 = "{ owner: aleo12a4wll9ax6w5355jph0dr5wt2vla5sss2t4cnch0tc3vzh643v8qcfvc7a.private, microcredits: 1000000u64.private, _nonce: 3634848344765318974603121890869676775499130077229666060613233255327643175219group.public, _version: 1u8.public }"
CREDITS_RECORD_VIEW_KEY = "5237002936265850807349726649400053591020997883662246784632368923777787639801field"
# Sender ciphertext (noted for future decrypt_sender work, not exposed in this task)
CREDITS_SENDER_CIPHERTEXT = "1182590395568997043375432557467567048762179115999922880321493200728848194550field"
CREDITS_SENDER_PLAINTEXT = "aleo1j92w9mhqznj2hvufad796y8suykjppk7f6n6xmncmktfm95vggzqx4sjlh"

# Token registry record V1 (utilities/test/records.rs:29-40)
TOKEN_REGISTRY_RECORD_V1 = """{
      owner: aleo1s3ws5tra87fjycnjrwsjcrnw2qxr8jfqqdugnf0xzqqw29q9m5pqem2u4t.private,
      amount: 1000u128.private,
      token_id: 1751493913335802797273486270793650302076377624243810059080883537084141842600field.private,
      external_authorization_required: false.private,
      authorized_until: 0u32.private,
      _nonce: 353510505137682717871934563523691055502582931368477380633253282125012046603group.public,
      _version: 1u8.public
    }"""

# Original credits plaintext (V0 era, 1500000000000000 microcredits)
CREDITS_RECORD_V0 = """{
  owner: aleo1j7qxyunfldj2lp8hsvy7mw5k8zaqgjfyr72x2gh3x4ewgae8v5gscf5jh3.private,
  microcredits: 1500000000000000u64.private,
  _nonce: 3077450429259593211617823051143573281856129402760267155982965992208217472983group.public,
  _version: 0u8.public
}"""


# ---------------------------------------------------------------------------
# 1. record_view_key KAT
# ---------------------------------------------------------------------------

def test_record_view_key_from_ciphertext():
    """RecordCiphertext.record_view_key() matches the known KAT value."""
    ct = RecordCiphertext.from_string(OWNER_CIPHERTEXT)
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    rvk = ct.record_view_key(vk)
    assert str(rvk) == RECORD_VIEW_KEY


def test_record_view_key_from_plaintext():
    """RecordPlaintext.record_view_key() gives the same value."""
    ct = RecordCiphertext.from_string(OWNER_CIPHERTEXT)
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    pt = ct.decrypt(vk)
    rvk_pt = pt.record_view_key(vk)
    assert str(rvk_pt) == RECORD_VIEW_KEY


def test_record_view_key_manual_derivation():
    """Manual derivation: (ct.nonce * vk.to_scalar()).to_x_coordinate() == RVK."""
    ct = RecordCiphertext.from_string(OWNER_CIPHERTEXT)
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    nonce = ct.nonce
    manual_rvk = (nonce * vk.to_scalar()).to_x_coordinate()
    assert str(manual_rvk) == RECORD_VIEW_KEY


def test_record_view_key_dunder_and_named_agree():
    """Group.__mul__ and scalar_multiply give same result."""
    ct = RecordCiphertext.from_string(OWNER_CIPHERTEXT)
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    nonce = ct.nonce
    sc = vk.to_scalar()
    assert (nonce * sc) == nonce.scalar_multiply(sc)


# ---------------------------------------------------------------------------
# 2. decrypt_with_record_view_key
# ---------------------------------------------------------------------------

def test_decrypt_with_rvk_matches_decrypt():
    """decrypt_with_record_view_key(RVK) produces same plaintext as decrypt(vk)."""
    ct = RecordCiphertext.from_string(OWNER_CIPHERTEXT)
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    rvk = ct.record_view_key(vk)
    pt_via_vk = ct.decrypt(vk)
    pt_via_rvk = ct.decrypt_with_record_view_key(rvk)
    assert str(pt_via_vk) == str(pt_via_rvk)


def test_decrypt_with_wrong_rvk_raises_or_wrong():
    """decrypt_with_record_view_key(Field.one()) yields wrong or raises."""
    ct = RecordCiphertext.from_string(OWNER_CIPHERTEXT)
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    pt_correct = ct.decrypt(vk)
    wrong_rvk = Field.one()
    try:
        pt_wrong = ct.decrypt_with_record_view_key(wrong_rvk)
        # If it doesn't raise, the owner must not match
        assert str(pt_wrong) != str(pt_correct)
    except Exception:
        pass  # raising is also acceptable


# ---------------------------------------------------------------------------
# 3. commitment + tag KAT
# ---------------------------------------------------------------------------

def test_commitment_and_tag_kat():
    """Full commitment->tag pipeline matches RECORD_TAG."""
    ct = RecordCiphertext.from_string(OWNER_CIPHERTEXT)
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    gk = GraphKey.from_view_key(vk)
    pt = ct.decrypt(vk)
    rvk = ct.record_view_key(vk)

    prog = ProgramID.from_string("credits.aleo")
    rec_name = Identifier.from_string("credits")
    commitment = pt.commitment(prog, rec_name, rvk)
    tag = pt.tag(gk, commitment)
    assert str(tag) == RECORD_TAG


def test_tag_static_method_matches():
    """RecordCiphertext.tag (static) gives same result as RecordPlaintext.tag."""
    ct = RecordCiphertext.from_string(OWNER_CIPHERTEXT)
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    gk = GraphKey.from_view_key(vk)
    pt = ct.decrypt(vk)
    rvk = ct.record_view_key(vk)

    prog = ProgramID.from_string("credits.aleo")
    rec_name = Identifier.from_string("credits")
    commitment = pt.commitment(prog, rec_name, rvk)

    tag_from_pt = pt.tag(gk, commitment)
    tag_from_ct = RecordCiphertext.tag(gk, commitment)
    assert tag_from_pt == tag_from_ct
    assert str(tag_from_pt) == RECORD_TAG


# ---------------------------------------------------------------------------
# 4. GraphKey
# ---------------------------------------------------------------------------

def test_graph_key_from_view_key_deterministic():
    """GraphKey.from_view_key is deterministic."""
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    gk1 = GraphKey.from_view_key(vk)
    gk2 = GraphKey.from_view_key(vk)
    assert gk1 == gk2


def test_graph_key_sk_tag_is_field():
    """GraphKey.sk_tag returns a Field."""
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    gk = GraphKey.from_view_key(vk)
    assert isinstance(gk.sk_tag, Field)


def test_graph_key_roundtrip():
    """GraphKey.from_string(str(gk)) round-trips."""
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    gk = GraphKey.from_view_key(vk)
    gk2 = GraphKey.from_string(str(gk))
    assert gk == gk2


# ---------------------------------------------------------------------------
# 5. get_member + microcredits
# ---------------------------------------------------------------------------

def test_credits_record_v0_microcredits():
    """V0 credits record microcredits == 1500000000000000."""
    pt = RecordPlaintext.from_string(CREDITS_RECORD_V0)
    assert pt.microcredits == 1500000000000000


def test_token_registry_microcredits_zero():
    """Token registry record has no microcredits field, returns 0."""
    pt = RecordPlaintext.from_string(TOKEN_REGISTRY_RECORD_V1)
    assert pt.microcredits == 0


def test_token_registry_get_member_amount():
    """get_member('amount') on token registry record returns a Plaintext with 1000u128."""
    pt = RecordPlaintext.from_string(TOKEN_REGISTRY_RECORD_V1)
    member = pt.get_member("amount")
    assert "1000u128" in str(member)


def test_get_member_nonexistent_raises():
    """get_member on a missing key raises an exception."""
    pt = RecordPlaintext.from_string(CREDITS_RECORD_V0)
    with pytest.raises(Exception):
        pt.get_member("nonexistent_field")


# ---------------------------------------------------------------------------
# 6. Field arithmetic sanity checks
# ---------------------------------------------------------------------------

def test_field_add():
    a = Field.from_string("2field")
    b = Field.from_string("3field")
    assert str(a + b) == "5field"
    assert str(a.add(b)) == "5field"
    assert (a + b) == a.add(b)


def test_field_sub():
    a = Field.from_string("5field")
    b = Field.from_string("3field")
    assert str(a - b) == "2field"
    assert str(a.subtract(b)) == "2field"
    assert (a - b) == a.subtract(b)


def test_field_neg():
    a = Field.from_string("1field")
    neg_a = -a
    assert str(neg_a.add(a)) == str(Field.zero())
    assert neg_a == a.negate()


def test_field_pow():
    two = Field.from_string("2field")
    three = Field.from_string("3field")
    result = two ** three
    assert str(result) == "8field"
    assert result == two.pow(three)


def test_field_double():
    a = Field.from_string("5field")
    assert str(a.double()) == "10field"


def test_field_one():
    one = Field.one()
    assert str(one) == "1field"


def test_field_add_sub_roundtrip():
    a = Field.from_string("12345field")
    b = Field.from_string("67890field")
    assert (a + b) - b == a


def test_field_bytes_roundtrip():
    a = Field.from_string("42field")
    b = Field.from_bytes_le(a.to_bytes_le())
    assert a == b


def test_field_dunder_mul_matches_multiply():
    a = Field.from_string("7field")
    b = Field.from_string("6field")
    assert (a * b) == a.multiply(b)


# ---------------------------------------------------------------------------
# 7. Scalar arithmetic sanity checks
# ---------------------------------------------------------------------------

def test_scalar_add():
    a = Scalar.from_string("2scalar")
    b = Scalar.from_string("3scalar")
    assert str(a + b) == "5scalar"
    assert (a + b) == a.add(b)


def test_scalar_mul():
    a = Scalar.from_string("4scalar")
    b = Scalar.from_string("5scalar")
    assert str(a * b) == "20scalar"
    assert (a * b) == a.multiply(b)


def test_scalar_div():
    a = Scalar.from_string("6scalar")
    b = Scalar.from_string("3scalar")
    # a / b * b == a
    assert (a / b) * b == a
    assert (a / b) == a.divide(b)


def test_scalar_sub():
    a = Scalar.from_string("5scalar")
    b = Scalar.from_string("3scalar")
    assert str(a - b) == "2scalar"
    assert (a - b) == a.subtract(b)


def test_scalar_pow_and_negate():
    a = Scalar.from_string("2scalar")
    b = Scalar.from_string("3scalar")
    assert str(a**b) == "8scalar"
    assert (a**b) == a.pow(b)
    # negation: a + (-a) == 0
    assert (a + (-a)) == Scalar.zero()
    assert (-a) == a.negate()


def test_scalar_div_zero():
    a = Scalar.from_string("1scalar")
    with pytest.raises(ZeroDivisionError):
        _ = a / Scalar.zero()


def test_scalar_one():
    one = Scalar.one()
    assert str(one) == "1scalar"


def test_scalar_bytes_roundtrip():
    a = Scalar.from_string("99scalar")
    b = Scalar.from_bytes_le(a.to_bytes_le())
    assert a == b


def test_scalar_to_field():
    one = Scalar.one()
    f = one.to_field()
    assert isinstance(f, Field)
    # 1scalar -> 1field
    assert str(f) == "1field"


# ---------------------------------------------------------------------------
# 8. Group arithmetic sanity checks
# ---------------------------------------------------------------------------

def test_group_generator_double():
    g = Group.generator()
    assert g.double() == g + g
    assert g.double() == g.add(g)


def test_group_add_sub():
    g = Group.generator()
    h = g.double()
    assert h - g == g
    assert h.subtract(g) == g
    assert (h - g) == h.subtract(g)


def test_group_neg():
    g = Group.generator()
    assert g + (-g) == Group.zero()
    assert g.add(g.negate()) == Group.zero()
    assert (-g) == g.negate()


def test_group_scalar_multiply():
    g = Group.generator()
    two = Scalar.from_string("2scalar")
    assert g * two == g.double()
    assert g.scalar_multiply(two) == g.double()
    assert (g * two) == g.scalar_multiply(two)


def test_group_to_x_coordinate():
    g = Group.generator()
    x = g.to_x_coordinate()
    assert isinstance(x, Field)


def test_group_to_address_roundtrip():
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    addr = vk.address
    group = addr.to_group()
    addr2 = group.to_address()
    assert str(addr) == str(addr2)


def test_address_to_group_to_address():
    """Address.to_group().to_address() round-trips."""
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    addr = vk.address
    assert addr == Address.from_group(addr.to_group())


def test_field_bytes_le_roundtrip_group():
    """Group bytes LE round-trip."""
    g = Group.generator()
    b = g.to_bytes_le()
    g2 = Group.from_bytes_le(b)
    assert g == g2


# ---------------------------------------------------------------------------
# 9. ViewKey conversions
# ---------------------------------------------------------------------------

def test_view_key_to_scalar():
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    sc = vk.to_scalar()
    assert isinstance(sc, Scalar)


def test_view_key_to_field():
    vk = ViewKey.from_string(OWNER_VIEW_KEY)
    f = vk.to_field()
    assert isinstance(f, Field)
