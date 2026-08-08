import unittest

from src.evaluation.acev2 import (
    addition_cost,
    calculate_tsetlin_extended_acev2,
    extract_trained_clause_profile,
    mac_cost,
    model_acev2_profiles,
    multiply_cost,
    patch_traversal_report,
    tsetlin_logic_cost,
)


class ClauseBank:
    def __init__(self, lengths):
        self.number_of_clauses = len(lengths)
        self.number_of_patches = 729
        self.incremental = True
        self._lengths = lengths

    def number_of_include_actions(self, clause):
        return self._lengths[clause]


class ClauseBanks(dict):
    def classes(self):
        return self.keys()


class FakeTsetlinModel:
    platform = "CPU"
    incremental = True

    def __init__(self):
        self.clause_banks = ClauseBanks({
            0: ClauseBank([0, 2]),
            1: ClauseBank([1, 4]),
        })


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

    def test_extended_tsetlin_cost_uses_real_clause_lengths(self):
        result = calculate_tsetlin_extended_acev2(
            image_shape=(4, 4),
            boolean_channels=1,
            patch_dim=(2, 2),
            class_count=2,
            clauses_per_class=2,
            max_included_literals=4,
            weighted_clauses=True,
            trained_clause_lengths=[0, 1, 2, 4],
        )
        self.assertAlmostEqual(result["extended_acev2_tm_upper_bound"], 120.8)
        self.assertAlmostEqual(result["extended_acev2_tm_trained_model"], 104.8)
        self.assertEqual(result["tm_extended_acev2_basis"], "trained Clause include actions with all configured patches")
        self.assertLess(result["tm_extended_acev2"], result["extended_acev2_tm_upper_bound"])

    def test_clause_profile_and_patch_traversal_are_taken_from_model(self):
        model = FakeTsetlinModel()
        profile = extract_trained_clause_profile(model)
        self.assertEqual(profile["clause_count"], 4)
        self.assertEqual(profile["included_literals_mean"], 1.75)
        self.assertEqual(profile["model_number_of_patches"], 729)
        public_profile = {key: value for key, value in profile.items() if key != "_flat_lengths"}
        traversal = patch_traversal_report(model, public_profile)
        self.assertTrue(traversal["all_patches_visited"])
        self.assertEqual(traversal["clause_patch_checks_per_image"], 2 * 2 * 729)


if __name__ == "__main__":
    unittest.main()

