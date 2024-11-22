import numpy as np
from forward_net import TwoLayerNet
from optimizer import SGD
from spiral import load_data
import matplotlib.pyplot as plt

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # Mac
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # Windows
plt.rcParams['axes.unicode_minus'] = False

def train_model():
    # 超參數設定
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    # 載入資料
    x, t = load_data()
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    optimizer = SGD(lr=learning_rate)

    # 用於繪製學習曲線
    loss_list = []
    accuracy_list = []
    
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
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            
            total_loss += loss
            loss_count += 1
            
            # 定期輸出學習狀況
            if (iters+1) % 10 == 0:
                avg_loss = total_loss / loss_count
                print(f'| 週期: {epoch+1} | 迭代: {iters+1}/{max_iters} | 損失: {avg_loss:.2f}')
                loss_list.append(avg_loss)
                total_loss, loss_count = 0, 0
                
        # 計算準確率
        correct_count = 0
        for i in range(data_size):
            score = model.predict(x[i:i+1])
            predict = np.argmax(score, axis=1)
            if predict == np.argmax(t[i:i+1], axis=1):
                correct_count += 1
                
        accuracy = correct_count / data_size
        accuracy_list.append(accuracy)
        print(f'準確率: {accuracy:.3f}')

    # 繪製學習曲線
    plt.figure(figsize=(10, 4))
    
    plt.subplot(121)
    plt.plot(loss_list)
    plt.title('損失函數變化', fontsize=12)
    plt.xlabel('迭代次數', fontsize=10)
    plt.ylabel('損失', fontsize=10)
    
    plt.subplot(122)
    plt.plot(accuracy_list)
    plt.title('準確率變化', fontsize=12)
    plt.xlabel('週期', fontsize=10)
    plt.ylabel('準確率', fontsize=10)
    
    plt.tight_layout()
    
    # 嘗試保存圖片，以防顯示時出現問題
    try:
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("圖片已保存為 'training_curves.png'")
    except Exception as e:
        print(f"保存圖片時出錯：{e}")
    
    plt.show()

if __name__ == '__main__':
    train_model() 