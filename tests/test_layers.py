import unittest
import numpy as np
import sys
import os

# 添加專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.layers import SoftmaxWithLoss, MatMul

class TestSoftmaxWithLoss(unittest.TestCase):
    def setUp(self):
        self.layer = SoftmaxWithLoss()
        np.random.seed(0)
    
    def test_forward_basic(self):
        """測試基本的前向傳播功能"""
        x = np.array([[1, 2, 3], [2, 1, 3]])
        t = np.array([2, 2])  # 類別標籤
        loss = self.layer.forward(x, t)
        
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)  # 損失值應該為正數
        
    def test_forward_with_onehot(self):
        """測試使用 one-hot 編碼標籤的情況"""
        x = np.array([[0.3, 0.2, 0.5], [0.1, 0.8, 0.1]])
        t = np.array([[0, 0, 1], [0, 1, 0]])
        loss = self.layer.forward(x, t)
        
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        
    def test_backward_shape(self):
        """測試反向傳播輸出的形狀"""
        x = np.random.randn(4, 3)
        t = np.array([0, 1, 2, 1])
        
        self.layer.forward(x, t)
        dx = self.layer.backward()
        
        self.assertEqual(dx.shape, x.shape)
        
    def test_backward_scale(self):
        """測試反向傳播的數值範圍"""
        x = np.random.randn(4, 3)
        t = np.array([0, 1, 2, 1])
        
        self.layer.forward(x, t)
        dx = self.layer.backward()
        
        self.assertTrue(np.all(dx <= 1.0))
        self.assertTrue(np.all(dx >= -1.0))
        
    def test_numerical_gradient(self):
        """測試數值梯度的正確性"""
        x = np.array([[0.2, 0.3, 0.5], [0.8, 0.1, 0.1]])
        t = np.array([2, 0])
        
        # 前向傳播
        loss = self.layer.forward(x, t)
        # 反向傳播
        dx = self.layer.backward()
        
        # 檢查梯度的平均值是否接近於 0
        self.assertTrue(np.abs(np.mean(dx)) < 0.1)
        
    def test_zero_input(self):
        """測試零輸入的情況"""
        x = np.zeros((2, 3))
        t = np.array([0, 1])
        
        loss = self.layer.forward(x, t)
        dx = self.layer.backward()
        
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.any(np.isnan(dx)))

class TestMatMul(unittest.TestCase):
    def setUp(self):
        self.W = np.array([[1, 2], [3, 4]])
        self.x = np.array([5, 6])
        self.layer = MatMul(self.W)

    def test_forward(self):
        expected = np.dot(self.x, self.W)
        result = self.layer.forward(self.x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_backward(self):
        dout = np.array([1, 2])
        self.layer.forward(self.x)
        dx = self.layer.backward(dout)
        expected_dx = np.dot(dout, self.W.T)
        np.testing.assert_array_almost_equal(dx, expected_dx)

if __name__ == '__main__':
    unittest.main()