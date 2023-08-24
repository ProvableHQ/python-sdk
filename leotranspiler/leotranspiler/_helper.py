# -*- coding: utf-8 -*-
def _get_rounding_decimal_places(value, max_places=10):
    if value == int(value):
        return 0
    for i in range(1, max_places + 1):
        rounded_val = round(value, i)
        if abs(value - rounded_val) < 10 ** (-i - 1):
            return i
    return max_places
