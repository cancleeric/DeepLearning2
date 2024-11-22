from forward_net import TwoLayerNet
from optimizer import SGD
from spiral import load_data
from trainer import Trainer
from visualizer import Visualizer

def main():
    # 超參數設定
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    learning_rate = 1.0

    # 載入資料
    x, t = load_data()
    
    # 初始化模型和優化器
    model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
    optimizer = SGD(lr=learning_rate)
    
    # 創建訓練器和視覺化器
    trainer = Trainer(model, optimizer)
    visualizer = Visualizer()
    
    # 訓練模型
    print("開始訓練...")
    loss_list, accuracy_list = trainer.fit(x, t, 
                                         max_epoch=max_epoch,
                                         batch_size=batch_size,
                                         eval_interval=10)
    print("訓練完成！")
    
    # 視覺化結果
    print("\n繪製學習曲線...")
    visualizer.plot_learning_curves(loss_list, accuracy_list)
    
    print("\n繪製決策邊界...")
    visualizer.plot_decision_boundary(model, x, t)

if __name__ == '__main__':
    main() 