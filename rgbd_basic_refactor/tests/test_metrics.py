import unittest

from src.evaluation.metrics import classification_metrics


class MetricsTests(unittest.TestCase):
    def test_macro_and_weighted_f1_are_distinct_and_named(self):
        labels = [0] * 9 + [1]
        predictions = [0] * 10
        result = classification_metrics(labels, predictions, ["majority", "minority"])
        self.assertIn("macro_f1", result)
        self.assertIn("weighted_f1", result)
        self.assertNotIn("f1", result)
        self.assertNotEqual(result["macro_f1"], result["weighted_f1"])
        self.assertAlmostEqual(result["accuracy"], 0.9)


if __name__ == "__main__":
    unittest.main()

