import unittest
from src.pattern_mining_toolkit import PatternMiningToolkit

class TestPatternMiningToolkit(unittest.TestCase):

    def setUp(self):
        self.toolkit = PatternMiningToolkit()

    def test_load_text_data(self):
        texts = ["apple banana", "banana orange", "apple"]
        self.toolkit.load_text_data(texts)
        self.assertEqual(len(self.toolkit.transactions), 3)
        self.assertIn("apple", self.toolkit.vocab)
        self.assertIn("banana", self.toolkit.vocab)
        self.assertIn("orange", self.toolkit.vocab)

    def test_apriori_algorithm(self):
        texts = ["apple banana", "banana orange", "apple"]
        self.toolkit.load_text_data(texts)
        result = self.toolkit.apriori_algorithm(min_support=0.5)
        self.assertIn(frozenset(["banana"]), result)
        self.assertIn(frozenset(["apple"]), result)

    def test_fp_growth(self):
        texts = ["apple banana", "banana orange", "apple"]
        self.toolkit.load_text_data(texts)
        result = self.toolkit.fp_growth(min_support=0.5)
        self.assertIn(frozenset(["banana"]), result)
        self.assertIn(frozenset(["apple"]), result)

    def test_maximal_pattern_mining(self):
        texts = ["apple banana", "banana orange", "apple"]
        self.toolkit.load_text_data(texts)
        result = self.toolkit.maximal_pattern_mining(min_support=0.5)
        self.assertIn(frozenset(["banana"]), result)

    def test_sequential_pattern_mining(self):
        texts = ["apple banana", "banana orange", "apple"]
        result = self.toolkit.sequential_pattern_mining(texts, min_support=0.5)
        self.assertIn(("banana",), result)

    def test_graph_based_pattern_mining(self):
        texts = ["apple banana", "banana orange", "apple"]
        result = self.toolkit.graph_based_pattern_mining(texts, min_support=0.5)
        self.assertGreater(len(result), 0)

    def test_closed_pattern_mining(self):
        texts = ["apple banana", "banana orange", "apple"]
        self.toolkit.load_text_data(texts)
        result = self.toolkit.closed_pattern_mining(min_support=0.5)
        self.assertIn(frozenset(["banana"]), result)

    def test_eclat_algorithm(self):
        texts = ["apple banana", "banana orange", "apple"]
        self.toolkit.load_text_data(texts)
        result = self.toolkit.eclat_algorithm(min_support=0.5)
        self.assertIn(frozenset(["banana"]), result)

    def test_charm_algorithm(self):
        texts = ["apple banana", "banana orange", "apple"]
        self.toolkit.load_text_data(texts)
        result = self.toolkit.charm_algorithm(min_support=0.5)
        self.assertIn(frozenset(["banana"]), result)

    def test_spade_algorithm(self):
        texts = ["apple banana", "banana orange", "apple"]
        result = self.toolkit.spade_algorithm(texts, min_support=0.5)
        self.assertIn(("banana",), result)

    def test_prefixspan_algorithm(self):
        texts = ["apple banana", "banana orange", "apple"]
        result = self.toolkit.prefixspan_algorithm(texts, min_support=0.5)
        self.assertIn(("banana",), result)

if __name__ == '__main__':
    unittest.main()