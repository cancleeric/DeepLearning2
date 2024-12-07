import numpy as np
from common.optimizer import SGD

class Trainer:
    def __init__(self, model, optimizer=SGD()):
        self.model = model
        self.optimizer = optimizer
        
        # 訓練紀錄
        self.loss_list = []
        self.accuracy_list = []
        
    def fit(self, x, t, max_epoch=300, batch_size=30, 
            eval_interval=20, verbose=True):
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
                self.optimizer.update(self.model.params, self.model.grads)
                
                total_loss += loss
                loss_count += 1
                
                                # 損失值記錄到 loss_list
                avg_loss = total_loss / loss_count
                self.loss_list.append(avg_loss)

                if verbose:
                    print(f'Epoch {epoch + 1}/{max_epoch}, Iter {iters + 1}/{max_iters}, Loss: {avg_loss:.4f}')
                
                total_loss, loss_count = 0, 0  # 重置損失統計

                # # 定期輸出學習狀況
                # if (iters+1) % eval_interval == 0 and verbose:
                #     avg_loss = total_loss / loss_count
                #     print(f'| 週期: {epoch+1} | 迭代: {iters+1}/{max_iters} | 損失: {avg_loss:.2f}')
                #     self.loss_list.append(avg_loss)
                #     total_loss, loss_count = 0, 0
                    
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
    
    @staticmethod
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
        