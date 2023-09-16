# -*- coding: utf-8 -*-
import logging
import sys
import unittest

from leotranspiler._input_generator import _InputGenerator


class TestInputGenerator(unittest.TestCase):
    def test_hierarchies_10_inputs(self):
        ig = _InputGenerator()

        num_inputs = 10
        for i in range(num_inputs):
            ig.add_input("uint256", "xi", True, i, f"input{i}")

        ig.get_struct_definitions_and_circuit_input_string()

    def test_hierarchies_100_inputs(self):
        ig = _InputGenerator()

        num_inputs = 100
        for i in range(num_inputs):
            ig.add_input("uint256", "xi", True, i, f"input{i}")

        ig.get_struct_definitions_and_circuit_input_string()

    def test_hierarchies_512_inputs(self):
        ig = _InputGenerator()

        num_inputs = 512
        for i in range(num_inputs):
            ig.add_input("uint256", "xi", True, i, f"input{i}")

        ig.get_struct_definitions_and_circuit_input_string()

    def test_hierarchies_513_inputs(self):
        ig = _InputGenerator()

        num_inputs = 513
        for i in range(num_inputs):
            ig.add_input("uint256", "xi", True, i, f"input{i}")

        ig.get_struct_definitions_and_circuit_input_string()

    def test_hierarchies_1000_inputs(self):
        ig = _InputGenerator()

        num_inputs = 1000
        for i in range(num_inputs):
            ig.add_input("uint256", "xi", True, i, f"input{i}")

        ig.get_struct_definitions_and_circuit_input_string()


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("testlogger").setLevel(logging.DEBUG)

    unittest.main()
