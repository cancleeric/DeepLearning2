
import sys
import os
import numpy as np
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import preprocess, create_contexts_target
from common.cbow import CBOW

def train_cbow():
    # 準備訓練資料
    text = "You say goodbye and I say hello."
    corpus, word_to_id, id_to_word = preprocess(text)
    
    # 設定超參數
    hidden_size = 100
    window_size = 1
    batch_size = 100
    max_epoch = 10

    vocab_size = len(word_to_id)

    # 建立訓練資料
    contexts, target = create_contexts_target(corpus, window_size)
    print(f'語料庫大小: {len(corpus)}')
    print(f'詞彙表大小: {vocab_size}')
    print(f'Contexts 形狀: {contexts.shape}')
    print(f'Target 形狀: {target.shape}')

    # 建立模型、最佳化器和訓練器
    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    # 開始訓練
    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot_loss_accuracy()

    # 顯示每個單詞的詞向量
    word_vecs = model.word_vecs
    for word_id, word in id_to_word.items():
        print(f'{word}: {word_vecs[word_id]}')

    # 儲存模型（可選）
    # np.save('cbow_weights.npy', word_vecs)

if __name__ == '__main__':
    train_cbow()