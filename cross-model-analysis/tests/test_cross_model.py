"""Small standard-library regression tests for cross-model summaries."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_model import describe_group, pearson, ranks, sign

class ComparisonTests(unittest.TestCase):
    def test_signs(self):
        self.assertEqual([sign(-2),sign(0),sign(3) ], [-1,0,1])

    def test_replication_requires_four(self):
        rows = {m:{"mean":"-0.05","q_value":"0.01"} for m in ("gemma","llama31","mistral")}
        self.assertFalse(describe_group(rows)["same_sign_and_fdr_all_four"])
        rows["qwen3"]={"mean":"-0.02","q_value":"0.04"}
        self.assertTrue(describe_group(rows)["same_sign_and_fdr_all_four"])

    def test_direction_and_fdr(self):
        rows = {m:{"mean":"-0.05","q_value":"0.01"} for m in ("gemma","llama31","mistral","qwen3")}
        rows["qwen3"]["mean"]="0.01"
        self.assertFalse(describe_group(rows)["same_nonzero_sign"])
        self.assertEqual(describe_group(rows)["fdr_significant_families"],4)

    def test_tied_ranks(self):
        self.assertEqual(ranks([2,1,2,4]),[2.5,1,2.5,4])

    def test_correlation(self):
        self.assertTrue(math.isclose(pearson([1,2,3],[2,4,6]),1))
        self.assertIsNone(pearson([1,1,1],[2,3,4]))

if __name__ == "__main__":
    unittest.main()
