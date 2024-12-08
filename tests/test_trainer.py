import unittest
import numpy as np
import sys
import os
# 添加專案路徑到模塊搜索路徑
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# 導入自定義模塊
from common.functions import relu
from common.trainer import Trainer
from common.layers import MatMul, SoftmaxWithLoss,  Sigmoid
from common.optimizer import SGD
from common.util import convert_one_hot

class DummyModel:
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(self, x, t):
        return np.sum(x)  # 假設損失是輸入數據的和
    
    def backward(self):
        pass
    
    def predict(self, x):
        return np.zeros((x.shape[0], 10))  # 假設有10個類別

class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel()
        self.optimizer = SGD()
        self.trainer = Trainer(self.model, self.optimizer)
        self.x = np.random.rand(100, 784)  # 假設輸入數據有100個樣本，每個樣本784維
        self.t = np.random.randint(0, 10, size=(100,))  # 假設輸出是10個類別的標籤

    def test_fit(self):
        loss_list, accuracy_list = self.trainer.fit(self.x, self.t, max_epoch=1, batch_size=10, eval_interval=1, verbose=False)
        self.assertTrue(len(loss_list) > 0)
        self.assertTrue(len(accuracy_list) > 0)

    def test_evaluate(self):
        accuracy = self.trainer.evaluate(self.x, self.t)
        self.assertTrue(0 <= accuracy <= 1)

if __name__ == '__main__':
    unittest.main()
