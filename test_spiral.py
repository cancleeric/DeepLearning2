import numpy as np
import matplotlib.pyplot as plt
from spiral import load_data, visualize

def test_spiral_data():
    # 測試資料生成
    x, t = load_data()
    
    # 檢查資料維度
    assert x.shape == (300, 2), "特徵矩陣形狀應為 (300, 2)"
    assert t.shape == (300, 3), "標籤矩陣形狀應為 (300, 3)"
    
    # 檢查 one-hot 編碼是否正確
    assert np.all(np.sum(t, axis=1) == 1), "每個樣本應該只屬於一個類別"
    
    # 檢查資料範圍
    assert np.all(x >= -1.5) and np.all(x <= 1.5), "資料應該在合理範圍內"
    
    print("資料測試通過！")
    
    # 繪製不同視角的圖
    plt.figure(figsize=(15, 5))
    
    # 原始散點圖
    plt.subplot(131)
    visualize(x, t)
    plt.title('原始螺旋資料')
    
    # 添加網格的散點圖
    plt.subplot(132)
    for i in range(3):
        plt.scatter(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], s=40, marker=['o', 'x', '^'][i])
    plt.grid(True)
    plt.title('帶網格的螺旋資料')
    
    # 熱力圖
    plt.subplot(133)
    h = plt.hist2d(x[:, 0], x[:, 1], bins=30, cmap='viridis')
    plt.colorbar(h[3])
    plt.title('資料密度分布')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_spiral_data() 