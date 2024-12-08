import numpy as np
import sys
import os
import unittest

# 添加專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.simple_cbow import SimpleCBOW
from common.optimizer import Adam
from common.util import preprocess, create_contexts_target

class TestSimpleCBOW(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 100
        self.embedding_size = 50
        self.model = SimpleCBOW(self.vocab_size, self.embedding_size)
        self.contexts = np.array([[1, 2], [3, 4], [5, 6]])
        self.target = np.array([0, 1, 2])

    def test_forward(self):
        loss = self.model.forward(self.contexts, self.target)
        self.assertIsInstance(loss, (float, np.float32))

    def test_backward(self):
        self.model.forward(self.contexts, self.target)
        self.model.backward()
        for grad in self.model.grads:
            self.assertIsNotNone(grad)

    def test_get_word_vecs(self):
        word_vecs = self.model.get_word_vecs()
        self.assertEqual(word_vecs.shape, (self.vocab_size, self.embedding_size))

if __name__ == '__main__':
    unittest.main()
