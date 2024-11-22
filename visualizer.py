import numpy as np
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        # 設定中文字型
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
        # plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows
        plt.rcParams['axes.unicode_minus'] = False
        
    def plot_decision_boundary(self, model, x, t):
        """繪製決策邊界"""
        h = 0.01
        x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
        y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        X = np.c_[xx.ravel(), yy.ravel()]
        scores = model.predict(X)
        predictions = np.argmax(scores, axis=1)
        predictions = predictions.reshape(xx.shape)
        
        plt.figure(figsize=(10, 10))
        plt.contourf(xx, yy, predictions, alpha=0.3)
        plt.colorbar()
        
        markers = ['o', 'x', '^']
        for i in range(3):
            plt.scatter(x[t[:, i] == 1, 0], 
                       x[t[:, i] == 1, 1],
                       s=50, 
                       marker=markers[i],
                       label=f'類別 {i}')
        
        plt.title('決策邊界視覺化')
        plt.xlabel('特徵 1')
        plt.ylabel('特徵 2')
        plt.legend()
        plt.savefig('decision_boundary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, loss_list, accuracy_list):
        """繪製學習曲線"""
        plt.figure(figsize=(10, 4))
        
        plt.subplot(121)
        plt.plot(loss_list)
        plt.title('損失函數變化')
        plt.xlabel('迭代次數')
        plt.ylabel('損失')
        
        plt.subplot(122)
        plt.plot(accuracy_list)
        plt.title('準確率變化')
        plt.xlabel('週期')
        plt.ylabel('準確率')
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show() 