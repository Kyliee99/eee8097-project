import unittest

import numpy as np

from src.models.bnn import binary_dot_from_bits


class XNOREquivalenceTests(unittest.TestCase):
    def test_xnor_popcount_equals_signed_binary_dot_product(self):
        rng = np.random.default_rng(42)
        input_bits = rng.integers(0, 2, size=(7, 33), dtype=np.int8).astype(bool)
        weight_bits = rng.integers(0, 2, size=(5, 33), dtype=np.int8).astype(bool)
        xnor = binary_dot_from_bits(input_bits, weight_bits)
        signed_inputs = np.where(input_bits, 1, -1)
        signed_weights = np.where(weight_bits, 1, -1)
        expected = signed_inputs @ signed_weights.T
        np.testing.assert_array_equal(xnor, expected)


if __name__ == "__main__":
    unittest.main()

