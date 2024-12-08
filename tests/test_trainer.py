import unittest
import numpy as np
import sys
import os
import matplotlib.pyplot as plt  # 添加這行
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

    def test_batch_size_validation(self):
        # 測試不同的批次大小
        batch_sizes = [1, 10, 50, 100]
        for batch_size in batch_sizes:
            loss_list, accuracy_list = self.trainer.fit(
                self.x, self.t, 
                max_epoch=1, 
                batch_size=batch_size, 
                eval_interval=1,
                verbose=False
            )
            self.assertTrue(len(loss_list) > 0)

    def test_max_grad_clipping(self):
        # 測試梯度裁剪
        max_grads = [1.0, 5.0, 10.0]
        for max_grad in max_grads:
            loss_list, accuracy_list = self.trainer.fit(
                self.x, self.t,
                max_epoch=1,
                batch_size=10,
                max_grad=max_grad,
                eval_interval=1,
                verbose=False
            )
            self.assertTrue(len(loss_list) > 0)

    def test_plot_loss_accuracy(self):
        # 測試繪圖功能
        self.trainer.fit(
            self.x, self.t,
            max_epoch=2,
            batch_size=10,
            eval_interval=1,
            verbose=False
        )
        try:
            self.trainer.plot_loss_accuracy()
            plt.close('all')  # 修改為 close('all')
        except Exception as e:
            self.fail(f"plot_loss_accuracy 失敗: {str(e)}")

    def test_edge_cases(self):
        # 測試邊界情況
        with self.assertRaises(ValueError):
            # 測試批次大小為0的情況
            self.trainer.fit(self.x, self.t, batch_size=0)
        
        with self.assertRaises(ValueError):
            # 測試週期數為負數的情況
            self.trainer.fit(self.x, self.t, max_epoch=-1)

if __name__ == '__main__':
    unittest.main()
