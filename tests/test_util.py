import unittest
import numpy as np
import sys
import os

# 添加專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi

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

    def test_create_co_matrix(self):
        text = "Hello world. Hello."
        corpus, word_to_id, id_to_word = preprocess(text)
        vocab_size = len(word_to_id)
        co_matrix = create_co_matrix(corpus, vocab_size, window_size=1)
        
        expected_co_matrix = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ])
        
        np.testing.assert_array_equal(co_matrix, expected_co_matrix)

    def test_cos_similarity(self):
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        similarity = cos_similarity(x, y)
        
        expected_similarity = 0.9746318461970762
        self.assertAlmostEqual(similarity, expected_similarity, places=7)

    def test_most_similar(self):
        text = "You say goodbye and I say hello."
        corpus, word_to_id, id_to_word = preprocess(text)
        vocab_size = len(word_to_id)
        co_matrix = create_co_matrix(corpus, vocab_size, window_size=1)
        
        # 測試 most_similar 函數
        print("Most similar words to 'you':")
        most_similar('you', word_to_id, id_to_word, co_matrix, top=3)

    def test_ppmi(self):
        text = "You say goodbye and I say hello."
        corpus, word_to_id, id_to_word = preprocess(text)
        vocab_size = len(word_to_id)
        co_matrix = create_co_matrix(corpus, vocab_size, window_size=1)
        
        ppmi_matrix = ppmi(co_matrix)
        
        expected_ppmi_matrix = np.array([
            [0.        , 1.8073549, 0.        , 0.        , 0.        , 0.        , 0.       ],
            [1.8073549 , 0.        , 0.8073549 , 0.        , 0.8073549 , 0.8073549 , 0.       ],
            [0.        , 0.8073549 , 0.        , 1.8073549 , 0.        , 0.        , 0.       ],
            [0.        , 0.        , 1.8073549 , 0.        , 1.8073549 , 0.        , 0.       ],
            [0.        , 0.8073549 , 0.        , 1.8073549 , 0.        , 0.        , 0.       ],
            [0.        , 0.8073549 , 0.        , 0.        , 0.        , 0.        , 2.807355 ],
            [0.        , 0.        , 0.        , 0.        , 0.        , 2.807355  , 0.       ]
        ], dtype=np.float32)
        
        np.testing.assert_array_almost_equal(ppmi_matrix, expected_ppmi_matrix, decimal=5)

if __name__ == '__main__':
    unittest.main()