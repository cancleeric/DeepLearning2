import sys
import numpy as np
from common.optimizer import Adam
from common.trainer import Trainer
from common.simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

def test_simple_cbow():
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

    # 將上下文轉換為 one-hot 格式
    contexts_one_hot = convert_one_hot(contexts, vocab_size)  # (6, 2, vocab_size)
    # 將目標轉換為 one-hot 格式
    target_one_hot = convert_one_hot(target, vocab_size)  # (6, vocab_size)

    # 創建模型
    model = SimpleCBOW(vocab_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    # 使用 Trainer 進行訓練
    trainer.fit(contexts_one_hot, target_one_hot, max_epoch, batch_size)
    trainer.plot_loss_accuracy()

    # 測試詞向量
    word_vecs = model.get_word_vecs()
    print("Word vectors:", word_vecs)

if __name__ == "__main__":
    test_simple_cbow()