import numpy as np

class SGD:
    """
    隨機梯度下降 (Stochastic Gradient Descent) 優化器
    """
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        """
        更新參數

        參數:
            params (list): 參數列表
            grads (list): 梯度列表
        """
        for param, grad in zip(params, grads):
            param -= self.lr * grad 


class Adam:
    """
    Adam 優化器
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None   

    def update(self, params, grads):
        """
        更新參數

        參數:
            params (list): 參數列表
            grads (list): 梯度列表
        """
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)   

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


class AdaGrad:
    """
    AdaGrad 優化器
    """
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        """
        更新參數

        參數:
            params (list): 參數列表
            grads (list): 梯度列表
        """
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] += grads[i] ** 2
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)
    
class Momentum:
    """
    動量 (Momentum) 優化器
    """
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        """
        更新參數

        參數:
            params (list): 參數列表
            grads (list): 梯度列表
        """
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]

