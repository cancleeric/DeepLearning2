
import numpy as np
import sys
import os

# 添加專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.optimizer import SGD, Adam, AdaGrad, Momentum

def test_optimization_function(optimizer, params, grads, n_iterations=100):
    """
    測試優化器在給定迭代次數內是否能減少損失
    """
    initial_params = [p.copy() for p in params]
    initial_loss = compute_loss(params)
    
    # 執行多次更新
    for _ in range(n_iterations):
        optimizer.update(params, grads)
        # 更新梯度（在實際應用中這會由模型計算）
        grads = [compute_gradient(p) for p in params]
    
    final_loss = compute_loss(params)
    
    print(f"Initial loss: {initial_loss:.6f}")
    print(f"Final loss: {final_loss:.6f}")
    print(f"Loss reduction: {initial_loss - final_loss:.6f}")
    
    return final_loss < initial_loss

def compute_loss(params):
    """
    計算簡單的損失函數 (例如: x^2 + y^2)
    """
    return sum(np.sum(p**2) for p in params)

def compute_gradient(param):
    """
    計算簡單損失函數的梯度 (例如: 2x, 2y)
    """
    return 2 * param

def test_all_optimizers():
    # 設置隨機種子以確保結果可重現
    np.random.seed(42)
    
    # 初始化測試參數和梯度
    params = [np.random.randn(3, 2), np.random.randn(2, 3)]
    grads = [compute_gradient(p) for p in params]
    
    # 測試各種優化器
    optimizers = {
        'SGD': SGD(lr=0.01),
        'Adam': Adam(lr=0.01),
        'AdaGrad': AdaGrad(lr=0.01),
        'Momentum': Momentum(lr=0.01)
    }
    
    results = {}
    for name, optimizer in optimizers.items():
        print(f"\nTesting {name} optimizer:")
        # 為每個優化器創建參數的深複本
        test_params = [p.copy() for p in params]
        test_grads = [g.copy() for g in grads]
        
        success = test_optimization_function(optimizer, test_params, test_grads)
        results[name] = success
        print(f"{name} optimization {'succeeded' if success else 'failed'}")
    
    return results

def verify_optimizer_behavior():
    """
    驗證優化器的特定行為
    """
    # 測試 SGD
    sgd = SGD(lr=0.1)
    param = np.array([1.0])
    grad = np.array([1.0])
    sgd.update([param], [grad])
    assert param == 0.9, "SGD update failed"
    
    # 測試 Momentum
    momentum = Momentum(lr=0.1, momentum=0.9)
    param = np.array([1.0])
    grad = np.array([1.0])
    momentum.update([param], [grad])
    # 第一次更新：v = 0.9 * 0 - 0.1 * 1 = -0.1
    assert abs(param - 0.9) < 1e-7, "Momentum update failed"
    
    # 測試 AdaGrad
    adagrad = AdaGrad(lr=0.1)
    param = np.array([1.0])
    grad = np.array([1.0])
    adagrad.update([param], [grad])
    expected = 1.0 - 0.1 / (np.sqrt(1.0) + 1e-7)
    assert abs(param - expected) < 1e-7, "AdaGrad update failed"
    
    print("All behavior tests passed!")

if __name__ == "__main__":
    # 執行所有優化器的測試
    results = test_all_optimizers()
    
    # 驗證優化器的具體行為
    verify_optimizer_behavior()