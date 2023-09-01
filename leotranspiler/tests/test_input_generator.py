# -*- coding: utf-8 -*-
import unittest

from leotranspiler._input_generator import _InputGenerator


class TestInputGenerator(unittest.TestCase):
    def test_hierarchies(self):
        ig = _InputGenerator()

        num_inputs = 1000
        for i in range(num_inputs):
            ig.add_input("uint256", "xi", True, i, f"input{i}")

        ig.get_struct_definitions_and_circuit_input_string()


if __name__ == "__main__":
    unittest.main()
