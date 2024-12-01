
import nltk
from nltk.corpus import ptb

class Dataset:
    def __init__(self):
        nltk.download('ptb')
        self.corpus = ptb.words()

    def get_corpus(self):
        return self.corpus

    def get_text(self):
        return ' '.join(self.corpus)