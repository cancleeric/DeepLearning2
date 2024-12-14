
from dataset.ptb_dataset import PTBDataset
import numpy as np
import pickle
import os
import urllib.request

url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
key_file = {
    'train':'ptb.train.txt',
    'test':'ptb.test.txt',
    'valid':'ptb.valid.txt'
}
save_file = {
    'train':'ptb.train.npy',
    'test':'ptb.test.npy',
    'valid':'ptb.valid.npy'
}
vocab_file = 'ptb.vocab.pkl'

class WebPTBDataset(PTBDataset):
    def __init__(self, data_dir='dataset'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._download_ptb()
        self.load_vocab()
        self.corpus = self._load_corpus()

    def _download(self, file_name):
        """下載單個檔案"""
        file_path = os.path.join(self.data_dir, file_name)
        if not os.path.exists(file_path):
            try:
                print(f"Downloading {file_name}...")
                urllib.request.urlretrieve(url_base + file_name, file_path)
                print(f"Downloaded {file_name}")
            except Exception as e:
                print(f"Error downloading {file_name}: {e}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise

    def _download_ptb(self):
        """下載所有需要的PTB檔案"""
        for filename in key_file.values():
            self._download(filename)

    def load_vocab(self):
        """載入或建立詞彙表"""
        vocab_path = os.path.join(self.data_dir, vocab_file)
        
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                self.word_to_id, self.id_to_word = pickle.load(f)
        else:
            self.word_to_id = {}
            self.id_to_word = {}
            for key in key_file.keys():
                file_path = os.path.join(self.data_dir, key_file[key])
                with open(file_path, 'r') as f:
                    words = f.read().replace('\n', '<eos>').strip().split()
                    
                for word in words:
                    if word not in self.word_to_id:
                        idx = len(self.word_to_id)
                        self.word_to_id[word] = idx
                        self.id_to_word[idx] = word
            
            with open(vocab_path, 'wb') as f:
                pickle.dump((self.word_to_id, self.id_to_word), f)

    def _load_corpus(self):
        """載入訓練資料集作為預設語料庫"""
        file_path = os.path.join(self.data_dir, key_file['train'])
        with open(file_path, 'r') as f:
            words = f.read().replace('\n', '<eos>').strip().split()
        return words

    def load_data(self, data_type='train'):
        """載入指定類型的資料"""
        if data_type not in key_file:
            raise ValueError(f"data_type must be one of {list(key_file.keys())}")
            
        save_path = os.path.join(self.data_dir, save_file[data_type])
        if os.path.exists(save_path):
            corpus = np.load(save_path)
        else:
            file_path = os.path.join(self.data_dir, key_file[data_type])
            with open(file_path, 'r') as f:
                words = f.read().replace('\n', '<eos>').strip().split()
            corpus = np.array([self.word_to_id[w] for w in words])
            np.save(save_path, corpus)
            
        return corpus, self.word_to_id, self.id_to_word