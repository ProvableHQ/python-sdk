from aleo.mainnet import Field, Group, Scalar, Boolean

def test_field_roundtrip():
    assert str(Field.from_string("1field")) == "1field"

def test_group_roundtrip():
    assert str(Group.from_string("2group")) == "2group"

def test_field_arithmetic():
    a = Field.from_string("2field")
    b = Field.from_string("3field")
    assert str(a * b) == "6field"

def test_boolean():
    assert bool(Boolean(True)) is True
