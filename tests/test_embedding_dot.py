import unittest
import numpy as np
import sys
import os

# 添加專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.negative_sampling_layer import EmbeddingDot

class TestEmbeddingDot(unittest.TestCase):
    def setUp(self):
        # 準備測試資料
        self.vocab_size = 5
        self.hidden_size = 3
        self.W = np.random.randn(self.vocab_size, self.hidden_size)
        self.layer = EmbeddingDot(self.W)
        
    def test_forward(self):
        # 準備輸入資料
        batch_size = 4
        h = np.random.randn(batch_size, self.hidden_size)
        idx = np.array([1, 3, 0, 2])  # 隨機索引值
        
        # 執行前向傳播
        out = self.layer.forward(h, idx)
        
        # 驗證輸出形狀
        self.assertEqual(out.shape, (batch_size,))
        
        # 驗證計算結果
        expected = np.array([
            np.sum(self.W[idx[i]] * h[i])
            for i in range(batch_size)
        ])
        np.testing.assert_almost_equal(out, expected)
        
    def test_forward_edge_cases(self):
        # 測試邊界情況，例如 batch_size = 1
        batch_size = 1
        h = np.random.randn(batch_size, self.hidden_size)
        idx = np.array([2])  # 單一索引值
        
        out = self.layer.forward(h, idx)
        self.assertEqual(out.shape, (batch_size,))
        
        expected = np.array([
            np.sum(self.W[idx[i]] * h[i])
            for i in range(batch_size)
        ])
        np.testing.assert_almost_equal(out, expected)

    def test_backward(self):
        # 先執行前向傳播
        batch_size = 4
        h = np.random.randn(batch_size, self.hidden_size)
        idx = np.array([1, 3, 0, 2])
        self.layer.forward(h, idx)
        
        # 執行反向傳播
        dout = np.random.randn(batch_size)
        dh = self.layer.backward(dout)
        
        # 驗證梯度形狀
        self.assertEqual(dh.shape, h.shape)
        
    def test_backward_edge_cases(self):
        # 測試邊界情況，例如 batch_size = 1
        batch_size = 1
        h = np.random.randn(batch_size, self.hidden_size)
        idx = np.array([2])
        self.layer.forward(h, idx)
        
        dout = np.random.randn(batch_size)
        dh = self.layer.backward(dout)
        
        self.assertEqual(dh.shape, h.shape)
    
    def test_cache_error(self):
        # 測試在沒有執行 forward 之前呼叫 backward 時的錯誤處理
        layer = EmbeddingDot(self.W)
        dout = np.random.randn(4)
        
        with self.assertRaises(ValueError):
            layer.backward(dout)

    def test_invalid_idx(self):
        # 測試無效的索引值，應該拋出 IndexError
        batch_size = 4
        h = np.random.randn(batch_size, self.hidden_size)
        idx = np.array([1, 3, 6, 2])  # 6 超出詞彙表範圍
        
        with self.assertRaises(IndexError):
            self.layer.forward(h, idx)

    def test_zero_vector_input(self):
        # 測試零向量作為輸入
        batch_size = 4
        h = np.zeros((batch_size, self.hidden_size))  # 全零向量
        idx = np.array([1, 3, 0, 2])
        
        out = self.layer.forward(h, idx)
        np.testing.assert_almost_equal(out, np.zeros(batch_size))
        
        dout = np.random.randn(batch_size)
        dh = self.layer.backward(dout)
        np.testing.assert_almost_equal(dh, np.zeros_like(h))

if __name__ == '__main__':
    unittest.main()
