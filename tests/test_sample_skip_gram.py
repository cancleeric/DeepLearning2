import unittest
import numpy as np
import sys
import os

# 添加專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.sample_skip_gram import SimpleSkipGram
from common.layers import MatMul, SoftmaxWithLoss

class TestSimpleSkipGram(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 10
        self.hidden_size = 5
        self.model = SimpleSkipGram(self.vocab_size, self.hidden_size)
        
    def _to_onehot(self, indices):
        """Convert indices to one-hot vectors"""
        N = len(indices)
        one_hot = np.zeros((N, self.vocab_size), dtype=np.float32)
        for i, idx in enumerate(indices):
            one_hot[i, idx] = 1
        return one_hot
        
    def test_init(self):
        self.assertEqual(len(self.model.params), 2)  # W_in, W_out
        self.assertEqual(len(self.model.grads), 2)
        self.assertEqual(self.model.word_vecs.shape, (self.vocab_size, self.hidden_size))
        
    def test_forward(self):
        batch_size = 3
        target = np.array([1, 3, 4])  # 假設的目標詞索引
        target_onehot = self._to_onehot(target)  # 轉換為 one-hot
        contexts = np.array([[2, 3],   # 每個目標詞的上下文
                            [4, 5],
                            [1, 2]])
        
        loss = self.model.forward(contexts, target_onehot)
        print(f"Loss type: {type(loss)}")  # 添加這行來打印 loss 的類型
        self.assertTrue(isinstance(loss, (float, np.floating)))  # 修改這一行，支持 numpy 和 Python float 類型
        self.assertTrue(loss > 0)
        
    def test_backward(self):
        batch_size = 3
        target = np.array([1, 3, 4])
        target_onehot = self._to_onehot(target)  # 轉換為 one-hot
        contexts = np.array([[2, 3],
                           [4, 5],
                           [1, 2]])
        
        self.model.forward(contexts, target_onehot)
        self.model.backward()
        
        # 檢查梯度是否已經計算
        for grad in self.model.grads:
            self.assertTrue(np.any(grad != 0))
            
    def test_params_shapes(self):
        # 檢查參數維度
        self.assertEqual(self.model.params[0].shape, (self.vocab_size, self.hidden_size))  # W_in
        self.assertEqual(self.model.params[1].shape, (self.hidden_size, self.vocab_size))  # W_out

if __name__ == '__main__':
    unittest.main()
