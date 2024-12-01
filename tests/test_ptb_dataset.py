
import unittest
import sys
import os

# 添加專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.ptb_dataset import PTBDataset
from nltk.corpus import treebank


class TestPTBDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = PTBDataset()

    def test_get_corpus(self):
        corpus = self.dataset.get_corpus()
        self.assertIsInstance(corpus, list)
        self.assertGreater(len(corpus), 0)

    def test_get_text(self):
        text = self.dataset.get_text()
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

if __name__ == '__main__':
    unittest.main()