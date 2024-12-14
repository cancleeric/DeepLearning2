import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import shutil
import numpy as np
from dataset.ptb_dataset_web import WebPTBDataset

class TestWebPTBDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 使用測試用的臨時目錄
        cls.test_dir = 'test_ptb_data'
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
        
    def setUp(self):
        self.dataset = WebPTBDataset(data_dir=self.test_dir)
    
    def tearDown(self):
        # 清理測試檔案
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """測試初始化和目錄建立"""
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertIsNotNone(self.dataset.word_to_id)
        self.assertIsNotNone(self.dataset.id_to_word)
    
    def test_vocab(self):
        """測試詞彙表功能"""
        # 檢查詞彙表是否包含基本符號
        self.assertIn('<eos>', self.dataset.word_to_id)
        # 檢查word_to_id和id_to_word是否相互對應
        for word, idx in self.dataset.word_to_id.items():
            self.assertEqual(word, self.dataset.id_to_word[idx])
    
    def test_load_data(self):
        """測試資料載入功能"""
        for data_type in ['train', 'test', 'valid']:
            corpus, word_to_id, id_to_word = self.dataset.load_data(data_type)
            # 檢查回傳值
            self.assertIsInstance(corpus, np.ndarray)
            self.assertGreater(len(corpus), 0)
            self.assertEqual(word_to_id, self.dataset.word_to_id)
            self.assertEqual(id_to_word, self.dataset.id_to_word)
    
    def test_invalid_data_type(self):
        """測試無效的資料類型"""
        with self.assertRaises(ValueError):
            self.dataset.load_data('invalid_type')

if __name__ == '__main__':
    unittest.main()