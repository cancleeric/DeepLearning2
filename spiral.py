import numpy as np

def load_data(seed=1984):
    np.random.seed(seed)
    N = 100  # 每個類別的樣本數
    DIM = 2  # 數據的維度
    CLS_NUM = 3  # 類別數

    x = np.zeros((N*CLS_NUM, DIM))
    t = np.zeros((N*CLS_NUM, CLS_NUM), dtype=np.int32)

    for j in range(CLS_NUM):
        for i in range(N):
            rate = i / N
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2

            ix = N*j + i
            x[ix] = np.array([radius*np.sin(theta),
                            radius*np.cos(theta)]).flatten()
            t[ix, j] = 1

    # 打亂數據
    indices = np.random.permutation(N*CLS_NUM)
    x = x[indices]
    t = t[indices]

    return x, t

def visualize(x, t):
    """
    繪製螺旋資料集的散點圖
    """
    import matplotlib.pyplot as plt
    
    # 設定中文字型
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
    plt.rcParams['axes.unicode_minus'] = False
    
    N = 100
    CLS_NUM = 3
    markers = ['o', 'x', '^']
    
    for i in range(CLS_NUM):
        plt.scatter(x[t[:, i] == 1, 0], x[t[:, i] == 1, 1], s=40, marker=markers[i])
    
    plt.title('螺旋資料集')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

if __name__ == '__main__':
    x, t = load_data()
    visualize(x, t) 