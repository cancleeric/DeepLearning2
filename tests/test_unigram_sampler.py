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

if __name__ == '__main__':
    unittest.main()
