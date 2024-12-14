import nltk
import numpy as np
from nltk.corpus import treebank


class PTBDataset:
    def __init__(self):
        self._download_treebank()
        self.corpus = self._load_corpus()
        self._prepare_data()

    def _download_treebank(self):
        try:
            treebank.ensure_loaded()
        except LookupError:
            nltk.download('treebank')

    def _load_corpus(self):
        return list(treebank.words())

    def get_corpus(self):
        return self.corpus

    def get_text(self):
        return ' '.join(self.corpus)

    def _prepare_data(self):
        """準備詞彙表和映射"""
        self.word_to_id = {}
        self.id_to_word = {}
        
        for word in self.corpus:
            if word not in self.word_to_id:
                new_id = len(self.word_to_id)
                self.word_to_id[word] = new_id
                self.id_to_word[new_id] = word
    
    def load_data(self, data_type='train'):
        """
        載入指定類型的資料
        :param data_type: 'train', 'test', 或 'valid'
        :return: (corpus, word_to_id, id_to_word)
        """
        if data_type not in ['train', 'test', 'valid']:
            raise ValueError("data_type 必須是 'train', 'test' 或 'valid'")

        # 切分資料
        train_size = int(len(self.corpus) * 0.8)
        valid_size = int(len(self.corpus) * 0.1)
        
        if data_type == 'train':
            target_words = self.corpus[:train_size]
        elif data_type == 'valid':
            target_words = self.corpus[train_size:train_size + valid_size]
        else:  # test
            target_words = self.corpus[train_size + valid_size:]

        # 轉換文字為 ID
        corpus = np.array([self.word_to_id[w] for w in target_words])
        return corpus, self.word_to_id, self.id_to_word

    def get_vocab_size(self):
        """獲取詞彙表大小"""
        return len(self.word_to_id)