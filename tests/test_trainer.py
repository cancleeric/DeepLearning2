import unittest
import numpy as np
import sys
import os
# 添加專案路徑到模塊搜索路徑
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# 導入自定義模塊
from common.trainer import Trainer
from common.layers import MatMul, SoftmaxWithLoss, ReLU
from common.optimizer import SGD
from common.util import convert_one_hot
# 簡單模型定義
class SimpleModel:
    def __init__(self, input_size, output_size):
        """
        初始化簡單模型。
        """
        W1 = 0.01 * np.random.randn(input_size, 50).astype('f')  # 隱藏層 50 個單元
        b1 = np.zeros(50).astype('f')
        W2 = 0.01 * np.random.randn(50, output_size).astype('f')  # 輸出層
        b2 = np.zeros(output_size).astype('f')
        
        self.layers = [
            MatMul(W1),  # 第一層
            ReLU(),      # 激活函數
            MatMul(W2)   # 第二層
        ]
        self.loss_layer = SoftmaxWithLoss()
        self.params, self.grads = [], []
        for layer in self.layers:
            if isinstance(layer, MatMul):
                self.params += layer.params
                self.grads += layer.grads

    def forward(self, x, t):
        """
        前向傳播。
        :param x: 輸入數據。
        :param t: 目標標籤。
        :return: 損失值。
        """
        for layer in self.layers:
            x = layer.forward(x)
        loss = self.loss_layer.forward(x, t)
        return loss

    def backward(self, dout=1):
        """
        反向傳播。
        :param dout: 上一層梯度。
        :return: 梯度。
        """
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def predict(self, x):
        """
        預測數據。
        :param x: 輸入數據。
        :return: 預測結果。
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x


# 測試 Trainer 類
class TestTrainer(unittest.TestCase):
    def test_trainer(self):
        """
        測試 Trainer 類的訓練過程，並將目標標籤轉換為 one-hot 編碼。
        """
        np.random.seed(0)
        input_size = 10
        output_size = 3
        model = SimpleModel(input_size, output_size)
        optimizer = SGD(lr=0.01)
        trainer = Trainer(model, optimizer)

        # 生成假數據
        x = np.random.randn(100, input_size).astype('f')
        t = np.random.randint(0, output_size, size=(100,)).astype(np.int32)

        # 將目標標籤轉換為 one-hot 編碼
        t_one_hot = convert_one_hot(t, output_size)

                # 打印調試信息
        print("Shape of x:", x.shape)
        print("Shape of t:", t.shape)
        print("Shape of t_one_hot:", t_one_hot.shape)

                # 確保 one-hot 編碼正確
        self.assertEqual(t_one_hot.shape, (100, output_size), "t_one_hot 的形狀不正確")

        
        # 訓練模型
        loss_list, accuracy_list = trainer.fit(x, t_one_hot, max_epoch=10, batch_size=10, eval_interval=1, verbose=False)
        # 打印調試信息
        print("Loss List:", loss_list)
        print("Accuracy List:", accuracy_list)        
        # 確保訓練過程中有數據返回
        self.assertGreater(len(loss_list), 0, "loss_list 為空，訓練過程未執行")
        self.assertGreater(len(accuracy_list), 0, "accuracy_list 為空，訓練過程未執行")

        # 檢查損失是否減少
        self.assertTrue(loss_list[-1] < loss_list[0], "損失未減少")

        # 檢查準確率是否增加
        self.assertTrue(accuracy_list[-1] > accuracy_list[0], "準確率未增加")


if __name__ == '__main__':
    unittest.main()