import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from common.negative_sampling_layer import UnigramSampler

class TestUnigramSampler(unittest.TestCase):
    def setUp(self):
        # 準備測試資料
        self.corpus = np.array([0, 1, 2, 3, 1, 2, 1, 2, 3])  # 詞頻：0(1次), 1(3次), 2(3次), 3(2次)
        self.power = 0.75
        self.sample_size = 2
        self.sampler = UnigramSampler(self.corpus, self.power, self.sample_size)
        
    def test_prepare(self):
        # 測試詞頻統計和機率計算
        expected_freq = np.array([1, 3, 3, 2])
        expected_prob = np.power(expected_freq, self.power)
        expected_prob = expected_prob / np.sum(expected_prob)
        
        np.testing.assert_array_almost_equal(
            self.sampler.word_p, 
            expected_prob
        )
        
    def test_get_negative_sample_shape(self):
        # 測試負採樣的輸出形狀
        target = np.array([1, 2, 3])
        neg_sample = self.sampler.get_negative_sample(target)
        
        self.assertEqual(
            neg_sample.shape, 
            (len(target), self.sample_size)
        )
        
    def test_get_negative_sample_range(self):
        # 測試採樣的值域是否正確
        target = np.array([1, 2, 3])
        neg_sample = self.sampler.get_negative_sample(target)
        
        self.assertTrue(np.all(neg_sample >= 0))
        self.assertTrue(np.all(neg_sample < self.sampler.vocab_size))
        
    def test_negative_sample_excludes_target(self):
        # 測試負採樣不包含目標詞
        target = np.array([1])
        for _ in range(5):  # 多次測試以減少隨機性影響
            neg_sample = self.sampler.get_negative_sample(target)
            # 確保負採樣的結果不包含目標詞
            self.assertTrue(np.all(neg_sample != target))

    def test_sample_size_larger_than_vocab(self):
        # 測試當 sample_size 大於詞彙表大小時的行為
        large_sample_size = 10
        sampler = UnigramSampler(self.corpus, self.power, large_sample_size)
        target = np.array([1])
        neg_sample = sampler.get_negative_sample(target)
        
        # 應確保輸出形狀正確且處理正確的重複抽樣邏輯
        self.assertEqual(neg_sample.shape, (len(target), large_sample_size))
        self.assertTrue(np.all(neg_sample >= 0))
        self.assertTrue(np.all(neg_sample < sampler.vocab_size))
        
    def test_empty_corpus(self):
        # 測試空的語料庫行為
        empty_corpus = np.array([])
        with self.assertRaises(ValueError):
            UnigramSampler(empty_corpus, self.power, self.sample_size)
            
    def test_all_targets_same(self):
        # 測試當所有目標詞都是相同的情況
        target = np.array([2, 2, 2])
        neg_sample = self.sampler.get_negative_sample(target)
        
        # 確保沒有負樣本等於目標詞
        for t, samples in zip(target, neg_sample):
            self.assertTrue(np.all(samples != t))
    
    def test_target_out_of_vocab(self):
        # 測試當目標詞超出詞彙表範圍時
        invalid_target = np.array([10])  # 詞彙表大小為 4，無效目標詞
        with self.assertRaises(IndexError):
            self.sampler.get_negative_sample(invalid_target)

    def test_negative_sampling_different_targets(self):
        # 測試不同目標詞是否生成正確的負樣本
        target = np.array([0, 1, 2, 3])
        neg_sample = self.sampler.get_negative_sample(target)
        
        # 確保對每個目標詞的負樣本均合法
        for t, samples in zip(target, neg_sample):
            self.assertTrue(np.all(samples >= 0))
            self.assertTrue(np.all(samples < self.sampler.vocab_size))
            self.assertTrue(np.all(samples != t))

if __name__ == '__main__':
    unittest.main()
