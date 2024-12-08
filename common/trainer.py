import numpy as np
from common.optimizer import SGD

# 移除 clip_grads 引入，改為直接實作
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads.values():
            grad *= rate

class Trainer:
    def __init__(self, model, optimizer=SGD()):
        self.model = model
        self.optimizer = optimizer
        
        # 訓練紀錄
        self.loss_list = []
        self.accuracy_list = []
        
    def fit(self, x, t, max_epoch=300, batch_size=30, 
            eval_interval=20, max_grad=None, verbose=True):  # 新增 max_grad 參數
        # 輸入驗證
        if batch_size <= 0:
            raise ValueError("batch_size 必須大於 0")
        if max_epoch <= 0:
            raise ValueError("max_epoch 必須大於 0")
        if eval_interval <= 0:
            raise ValueError("eval_interval 必須大於 0")

        """訓練模型"""
        data_size = len(x)
        max_iters = data_size // batch_size
        total_loss = 0
        loss_count = 0
        
        for epoch in range(max_epoch):
            # 打亂資料
            idx = np.random.permutation(data_size)
            x = x[idx]
            t = t[idx]
            
            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]
                
                # 計算梯度，更新參數
                loss = self.model.forward(batch_x, batch_t)
                self.model.backward()
                params, grads = remove_duplicate(self.model.params, self.model.grads)  # 刪除重複的參數並合併梯度
                if max_grad is not None:
                    clip_grads(grads, max_grad)  # 裁剪梯度
                self.optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1
                
                # 損失值記錄到 loss_list
                avg_loss = total_loss / loss_count
                self.loss_list.append(avg_loss)

                if (iters + 1) % eval_interval == 0:
                    avg_loss = total_loss / loss_count
                    if verbose:
                        print(f'| 週期: {epoch + 1} | 迭代: {iters + 1}/{max_iters} | 損失: {avg_loss:.2f}')
                    self.loss_list.append(avg_loss)
                    total_loss, loss_count = 0, 0  # 在這裡重置損失統計
                        
            # 計算準確率
            accuracy = self.evaluate(x, t)
            self.accuracy_list.append(accuracy)
            if verbose:
                print(f'準確率: {accuracy:.3f}')
                
        return self.loss_list, self.accuracy_list
    
    def evaluate(self, x, t, batch_size=100):
        """評估模型準確率"""
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
            
        correct_count = 0
        for i in range(0, len(x), batch_size):
            batch_x = x[i:i + batch_size]
            batch_t = t[i:i + batch_size]
            score = self.model.predict(batch_x)
            predict = np.argmax(score, axis=1)
            correct_count += np.sum(predict == batch_t)
            
        accuracy = correct_count / len(x)
        return accuracy
    
    def plot_loss_accuracy(self):
        """繪製損失和準確率曲線"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))

        # 繪製損失曲線
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_list)
        plt.title('Loss over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        # 繪製準確率曲線
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy_list)
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')

        plt.tight_layout()
        plt.show()
        
def remove_duplicate(params, grads):
    """刪除重複的參數並合併梯度"""
    unique_params = {}
    unique_grads = {}
    
    for key in params.keys():
        if key not in unique_params:
            unique_params[key] = params[key]
            unique_grads[key] = grads[key]
        else:
            unique_grads[key] += grads[key]  # 合併梯度
    
    return unique_params, unique_grads