import unittest
from leotranspiler.leo_transpiler import LeoTranspiler

class TestLeoTranspiler(unittest.TestCase):
    
    def test_init(self):
        leo_transpiler = LeoTranspiler(None)
        self.assertEqual(leo_transpiler.model, None)
        self.assertEqual(leo_transpiler.validation_data, None)
        self.assertEqual(leo_transpiler.model_as_input, False)
        self.assertEqual(leo_transpiler.ouput_model_hash, None)
        self.assertEqual(leo_transpiler.transpilation_result, None)
        self.assertEqual(leo_transpiler.leo_program_stored, False)

if __name__ == '__main__':
    unittest.main()