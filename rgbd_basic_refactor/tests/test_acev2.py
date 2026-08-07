import unittest

from src.evaluation.acev2 import (
    addition_cost,
    mac_cost,
    model_acev2_profiles,
    multiply_cost,
    tsetlin_logic_cost,
)


class ACEv2Tests(unittest.TestCase):
    def test_primary_formulas(self):
        self.assertEqual(mac_cost(10, 4, 8), 320)
        self.assertEqual(multiply_cost(10, 4, 8), 240)
        self.assertEqual(addition_cost(10, 4, 8), 80)
        self.assertEqual(addition_cost(10, 32, 32, floating_point=True), 1920)

    def test_bnn_algorithm_and_python_profiles_are_separate(self):
        layers = [(16, 8), (8, 2)]
        result = model_acev2_profiles("bnn", layers)
        self.assertLess(result["algorithmic_acev2"]["acev2"], result["python_tensor_acev2"]["acev2"])
        self.assertEqual(result["algorithmic_acev2"]["weight_bits_by_layer"], [1, 1])

    def test_tsetlin_patch_count_and_nonstandard_status(self):
        result = tsetlin_logic_cost((32, 32), 3, (6, 6), 13, 200, 24)
        self.assertEqual(result["patches_per_image"], 729)
        self.assertEqual(result["features_per_patch"], 108)
        self.assertIsNone(result["acev2"])
        self.assertGreater(result["logic_operations_upper_bound"], 0)


if __name__ == "__main__":
    unittest.main()

