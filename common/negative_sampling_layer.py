import numpy as np
from common.layers import Embedding, SigmoidWithLoss
from collections import Counter


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        if self.cache is None:
            raise ValueError("Cache is not set. Ensure forward method is called before backward.")
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh

GPU = False  # Set to True if using GPU

class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None
        self.prepare(corpus, power)

    def prepare(self, corpus, power):
        word_freq = Counter(corpus)
        vocab_size = len(word_freq)
        self.vocab_size = vocab_size
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = word_freq[i]
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size), replace=True, p=self.word_p)
        return negative_sample

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = []
        self.embed_dot_layers = []
        for i in range(sample_size + 1):
            layer = SigmoidWithLoss()
            self.loss_layers.append(layer)  # SigmoidWithLoss   
            layer = EmbeddingDot(W)
            self.embed_dot_layers.append(layer)  # EmbeddingDot 
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None
    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)
        # 正例的前向传播
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        # 负例的前向传播
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)
        return loss
    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            ds = l0.backward(dout)
            dh += l1.backward(ds)
        return dh

    
