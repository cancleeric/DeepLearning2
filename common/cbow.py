import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.layers import Embedding
from common.negative_sampling_layer import NegativeSamplingLoss


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        self.vocab_size = vocab_size
        V, H = vocab_size, hidden_size
        W_in = 0.01 * np.random.randn(vocab_size, hidden_size).astype('f')  # (V, H)
        W_out = 0.01 * np.random.randn(vocab_size, hidden_size).astype('f')  # (V, H)

        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        self.word_vecs = W_in
        
    def forward(self, contexts, target):
        if not np.all(contexts < self.vocab_size) or not np.all(target < self.vocab_size):
            raise ValueError("上下文或目標詞的索引超出詞彙表範圍")

        if contexts.shape[1] != len(self.in_layers):
            raise ValueError(f"contexts 的第二維大小應該是 {len(self.in_layers)}，但得到了 {contexts.shape[1]}")
        
        if contexts.shape[0] != target.shape[0]:
            raise ValueError("contexts 和 target 的 batch_size 不一致")

        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        # 確保返回標量值
        return float(loss)
    
    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None
