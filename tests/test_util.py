import unittest
import numpy as np
import sys
import os

# 添加專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from util import preprocess

class TestPreprocess(unittest.TestCase):
    def test_preprocess(self):
        text = "Hello world. Hello."
        corpus, word_to_id, id_to_word = preprocess(text)
        
        expected_corpus = np.array([0, 1, 2, 0, 2])
        expected_word_to_id = {'hello': 0, 'world': 1, '.': 2}
        expected_id_to_word = {0: 'hello', 1: 'world', 2: '.'}
        
        np.testing.assert_array_equal(corpus, expected_corpus)
        self.assertEqual(word_to_id, expected_word_to_id)
        self.assertEqual(id_to_word, expected_id_to_word)

if __name__ == '__main__':
    unittest.main()