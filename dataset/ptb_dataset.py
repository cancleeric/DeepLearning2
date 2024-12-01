import nltk
from nltk.corpus import treebank


class PTBDataset:
    def __init__(self):
        self._download_treebank()
        self.corpus = self._load_corpus()

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