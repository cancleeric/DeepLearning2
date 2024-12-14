import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.cbow import CBOW
from common.layers import Embedding
from common.negative_sampling_layer import NegativeSamplingLoss

class TestCBOW(unittest.TestCase):
    def setUp(self):
        # 使用較小的詞彙表大小以避免索引越界
        self.vocab_size = 5
        self.hidden_size = 3
        self.window_size = 1
        self.corpus = np.array([0, 1, 2, 3, 1, 2, 1, 2, 3])
        self.model = CBOW(self.vocab_size, self.hidden_size, self.window_size, self.corpus)

    def test_init(self):
        # 測試 CBOW 模型初始化
        self.assertEqual(len(self.model.in_layers), 2 * self.window_size)  # 輸入層數量
        self.assertEqual(self.model.word_vecs.shape, (self.vocab_size, self.hidden_size))  # 詞向量維度

    def test_forward(self):
        # 使用確定的、有效範圍內的索引
        contexts = np.array([[1, 3]])  # Context words
        target = np.array([2])        # Target word
        
        loss = self.model.forward(contexts, target)
        self.assertTrue(isinstance(loss, float))
        self.assertTrue(loss >= 0)

    def test_backward(self):
        # 使用確定的、有效範圍內的索引
        contexts = np.array([[1, 3]])  # Context words
        target = np.array([2])        # Target word
        
        self.model.forward(contexts, target)
        self.model.backward()
        
        for grad in self.model.grads:
            self.assertFalse(np.all(grad == 0))  # 梯度應該被更新

    def test_input_shape_error(self):
        # 測試輸入形狀錯誤情況
        batch_size = 3
        invalid_contexts = np.random.randint(0, self.vocab_size, (batch_size, self.window_size))  # 少了一半的上下文
        target = np.random.randint(0, self.vocab_size, batch_size)

        # 應該拋出 ValueError
        with self.assertRaises(ValueError):
            self.model.forward(invalid_contexts, target)

    def test_batch_size_mismatch(self):
        # 測試 batch_size 不匹配
        contexts = np.random.randint(0, self.vocab_size, (3, 2 * self.window_size))
        target = np.random.randint(0, self.vocab_size, 2)  # 不同的 batch_size

        # 應該拋出 ValueError
        with self.assertRaises(ValueError):
            self.model.forward(contexts, target)

    def test_single_sample(self):
        # 使用確定的、有效範圍內的索引
        contexts = np.array([[1, 3]])
        target = np.array([2])
        
        loss = self.model.forward(contexts, target)
        self.assertTrue(isinstance(loss, float))
        self.assertTrue(loss >= 0)

if __name__ == '__main__':
    unittest.main()
