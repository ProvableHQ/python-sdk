# -*- coding: utf-8 -*-
_leo_type_bits = [2**3, 2**4, 2**5, 2**6, 2**7]


def _get_leo_integer_type(signed, value_bits):
    for bits in _leo_type_bits:
        # Subtract 1 bit for signed integers
        max_bits = bits - 1 if signed else bits
        if value_bits <= max_bits:
            return f"{'i' if signed else 'u'}{bits.bit_length()}"

    raise ValueError(
        f"No leo type for {'signed' if signed else 'unsigned'} value with more than "
        f"{_leo_type_bits[-1]} bits. Try quantizing the model and/or the data."
    )


class LeoType:
    def __init__(self, value, type):
        self.value = value
        self.type = type
