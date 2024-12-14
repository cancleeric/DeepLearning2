
import os
import numpy as np
from dataset.ptb_dataset import PTBDataset

def load_data(data_type='train'):
    """
    載入 PTB 資料集
    :param data_type: 'train', 'test', 或 'valid' (validation)
    :return: 語料庫單字ID列表，單字到ID的字典，ID到單字的字典
    """
    if data_type not in ['train', 'test', 'valid']:
        raise ValueError("data_type 必須是 'train', 'test' 或 'valid'")

    # 使用現有的 PTBDataset 類別
    dataset = PTBDataset()
    words = dataset.get_corpus()

    # 根據資料類型選擇不同的切分
    train_size = int(len(words) * 0.8)
    valid_size = int(len(words) * 0.1)
    
    if data_type == 'train':
        target_words = words[:train_size]
    elif data_type == 'valid':
        target_words = words[train_size:train_size + valid_size]
    else:  # test
        target_words = words[train_size + valid_size:]

    # 建立詞彙表
    word_to_id = {}
    id_to_word = {}
    
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    # 轉換文字為 ID
    corpus = np.array([word_to_id[w] for w in target_words])

    return corpus, word_to_id, id_to_word

def get_vocab_size():
    """
    獲取詞彙表大小
    """
    dataset = PTBDataset()
    words = dataset.get_corpus()
    return len(set(words))