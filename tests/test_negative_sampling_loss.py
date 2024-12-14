
import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.layers import SigmoidWithLoss
from common.negative_sampling_layer import NegativeSamplingLoss

class TestNegativeSamplingLoss(unittest.TestCase):
    def setUp(self):
        # 準備測試資料
        self.vocab_size = 5
        self.hidden_size = 3
        self.sample_size = 2
        self.batch_size = 4
        
        # 初始化權重矩陣
        self.W = np.random.randn(self.vocab_size, self.hidden_size)
        
        # 準備簡單的語料庫
        self.corpus = np.array([0, 1, 2, 3, 1, 2, 1, 2, 3])
        
        # 初始化 NegativeSamplingLoss
        self.loss_layer = NegativeSamplingLoss(self.W, self.corpus, 
                                             power=0.75, sample_size=self.sample_size)
        
    def test_init(self):
        # 測試初始化後的參數
        self.assertEqual(len(self.loss_layer.loss_layers), self.sample_size + 1)
        self.assertEqual(len(self.loss_layer.embed_dot_layers), self.sample_size + 1)
        
    def test_forward(self):
        # 準備輸入資料
        h = np.random.randn(self.batch_size, self.hidden_size)
        target = np.array([1, 3, 0, 2])
        
        # 執行前向傳播
        loss = self.loss_layer.forward(h, target)
        
        # 驗證損失值
        self.assertTrue(isinstance(loss, float))
        self.assertTrue(loss >= 0)
        
    def test_backward(self):
        # 準備資料並執行前向傳播
        h = np.random.randn(self.batch_size, self.hidden_size)
        target = np.array([1, 3, 0, 2])
        self.loss_layer.forward(h, target)
        
        # 執行反向傳播
        dh = self.loss_layer.backward(1)
        
        # 驗證梯度形狀
        self.assertEqual(dh.shape, h.shape)
        
    def test_params_and_grads(self):
        # 驗證參數和梯度列表的長度相同
        self.assertEqual(len(self.loss_layer.params), len(self.loss_layer.grads))
        
        # 驗證每個嵌入點積層的參數都被正確加入
        expected_params_count = len(self.loss_layer.embed_dot_layers) * len(self.loss_layer.embed_dot_layers[0].params)
        self.assertEqual(len(self.loss_layer.params), expected_params_count)

if __name__ == '__main__':
    unittest.main()