# -*- coding: utf-8 -*-
from typing import Optional, Union

_leo_type_bits = [2**3, 2**4, 2**5, 2**6, 2**7]


def _get_leo_integer_bits(signed: bool, value_bits: int):
    for bits in _leo_type_bits:
        # Subtract 1 bit for signed integers
        max_bits = bits - 1 if signed else bits
        if value_bits <= max_bits:
            return bits

    raise ValueError(
        f"No leo type for {'signed' if signed else 'unsigned'} value with more than "
        f"{_leo_type_bits[-1]} bits. Try quantizing the model and/or the data."
    )


def _get_leo_integer_type(signed: bool, value_bits: int):
    for bits in _leo_type_bits:
        # Subtract 1 bit for signed integers
        max_bits = bits - 1 if signed else bits
        if value_bits <= max_bits:
            return f"{'i' if signed else 'u'}{bits}"

    raise ValueError(
        f"No leo type for {'signed' if signed else 'unsigned'} value with more than "
        f"{_leo_type_bits[-1]} bits. Try quantizing the model and/or the data."
    )


class LeoInteger:
    """Leo Integer."""

    def __init__(
        self,
        signed: bool,
        bits: int,
        original_value: Union[int, float],
        fixed_point_scaling_factor: Optional[int] = None,
    ):
        """Leo type.

        Parameters
        ----------
        value : int
            The value of the Leo type.
        type : str
            The Leo type.

        Returns
        -------
        LeoType
            The Leo type.
        """
        self.original_value = original_value
        self.type = type

        self.fixed_point_value = None  # todo
        self.decimal_value = None  # todo

        self.leo_value = self._get_leo_value(
            self.fixed_point_value, signed, bits, fixed_point_scaling_factor
        )
