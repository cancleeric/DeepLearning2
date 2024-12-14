import sys
import numpy as np
import sys
import os

# 添加專案路徑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.optimizer import Adam
from common.trainer import Trainer
from common.simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

def train_simple_cbow():
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    vocab_size = len(word_to_id)
    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    contexts, target = create_contexts_target(corpus, window_size)
    print("Contexts shape:", contexts.shape)  # 應該是 (6, 2)
    print("Target shape:", target.shape)      # 應該是 (6,)

    # 直接使用索引進行訓練，不需要轉換為 one-hot
    model = SimpleCBOW(vocab_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    # 使用 Trainer 進行訓練
    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot_loss_accuracy()

    # 測試詞向量
    word_vecs = model.get_word_vecs()
    # print("Word vectors:", word_vecs)
    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])

if __name__ == "__main__":
    train_simple_cbow()