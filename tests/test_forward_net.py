import numpy as np
import sys
import os

# 添加專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from forward_net import TwoLayerNet

def test_two_layer_net():
    # 測試網路初始化
    input_size = 2
    hidden_size = 3
    output_size = 2
    net = TwoLayerNet(input_size, hidden_size, output_size)
    
    # 檢查網路結構
    assert len(net.layers) == 3, "網路應該包含3層"
    assert len(net.params) == 4, "應該有4個參數（W1, b1, W2, b2）"
    
    # 檢查參數形狀
    W1, b1 = net.layers[0].params
    W2, b2 = net.layers[2].params
    
    assert W1.shape == (input_size, hidden_size), f"W1形狀應為({input_size}, {hidden_size})"
    assert b1.shape == (hidden_size,), f"b1形狀應為({hidden_size},)"
    assert W2.shape == (hidden_size, output_size), f"W2形狀應為({hidden_size}, {output_size})"
    assert b2.shape == (output_size,), f"b2形狀應為({output_size},)"
    
    # 測試前向傳播
    x = np.array([[0.5, -0.2]])  # 一個樣本，兩個特徵
    y = net.predict(x)
    
    assert y.shape == (1, output_size), f"輸出形狀應為(1, {output_size})"
    
    print("所有測試通過！")

if __name__ == "__main__":
    test_two_layer_net() 