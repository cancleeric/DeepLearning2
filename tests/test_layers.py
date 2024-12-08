import unittest
import numpy as np
import sys
import os

# 添加專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.layers import SoftmaxWithLoss, MatMul

class TestSoftmaxWithLoss(unittest.TestCase):
    def test_forward(self):
        """
        測試 SoftmaxWithLoss 類的前向傳播。
        """
        np.random.seed(0)
        x = np.random.randn(10, 3)  # 假設有 10 個樣本，每個樣本有 3 個類別
        t = np.random.randint(0, 3, size=(10,))  # 目標標籤

        layer = SoftmaxWithLoss()
        loss = layer.forward(x, t)
        
        # 檢查損失值是否為浮點數
        self.assertIsInstance(loss, float, "損失值應該是浮點數")

    def test_backward(self):
        """
        測試 SoftmaxWithLoss 類的反向傳播。
        """
        np.random.seed(0)
        x = np.random.randn(10, 3)  # 假設有 10 個樣本，每個樣本有 3 個類別
        t = np.random.randint(0, 3, size=(10,))  # 目標標籤

        layer = SoftmaxWithLoss()
        layer.forward(x, t)
        dout = layer.backward()

        # 檢查反向傳播的輸出形狀是否正確
        self.assertEqual(dout.shape, x.shape, "反向傳播的輸出形狀應該與輸入形狀相同")

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