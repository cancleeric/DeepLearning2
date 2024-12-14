import unittest
import sys
import os

import numpy as np

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

    def test_load_data(self):
        # 測試訓練資料載入
        corpus, word_to_id, id_to_word = self.dataset.load_data('train')
        self.assertIsInstance(corpus, np.ndarray)
        self.assertIsInstance(word_to_id, dict)
        self.assertIsInstance(id_to_word, dict)
        
        # 檢查資料切分比例
        train_size = len(corpus)
        total_size = len(self.dataset.get_corpus())
        self.assertAlmostEqual(train_size / total_size, 0.8, places=1)

    def test_vocab_mappings(self):
        # 測試詞彙映射的一致性
        word_to_id = self.dataset.word_to_id
        id_to_word = self.dataset.id_to_word
        
        # 檢查映射的完整性
        for word, idx in word_to_id.items():
            self.assertEqual(word, id_to_word[idx])
            
        # 檢查索引的連續性
        self.assertEqual(len(word_to_id), len(id_to_word))
        self.assertEqual(set(word_to_id.values()), set(range(len(word_to_id))))

    def test_invalid_data_type(self):
        # 測試無效的資料類型
        with self.assertRaises(ValueError):
            self.dataset.load_data('invalid_type')

    def test_get_vocab_size(self):
        # 測試詞彙表大小計算
        vocab_size = self.dataset.get_vocab_size()
        self.assertEqual(vocab_size, len(self.dataset.word_to_id))
        self.assertGreater(vocab_size, 0)

if __name__ == '__main__':
    unittest.main()