import numpy as np

from common.layers import MatMul, SoftmaxWithLoss


class SimpleCBOW:
     def __init__(self, vocab_size, embedding_size):
            V, H = vocab_size, embedding_size
            
            # 初始化權重
            W_in = 0.01 * np.random.randn(V, H).astype('f')
            W_out = 0.01 * np.random.randn(H, V).astype('f')
            
            # 建立層
            self.in_layer0 = MatMul(W_in)
            self.in_layer1 = MatMul(W_in)
            self.out_layer = MatMul(W_out)
            self.loss_layer = SoftmaxWithLoss()
            
            # 將所有的權重與梯度整合到字典中
            layers = [self.in_layer0, self.in_layer1, self.out_layer]
            self.params, self.grads = [], []
            for layer in layers:
                self.params += layer.params
                self.grads += layer.grads
            
            # 將詞向量與詞集合到字典中
            self.word_vecs = W_in

     def forward(self, contexts, target):
            h0 = self.in_layer0.forward(contexts[:, 0])
            h1 = self.in_layer1.forward(contexts[:, 1])
            h = (h0 + h1) * 0.5
            score = self.out_layer.forward(h)
            loss = self.loss_layer.forward(score, target)
            return loss
     
     def backward(self, dout=1):
            ds = self.loss_layer.backward(dout)
            da = self.out_layer.backward(ds)
            da *= 0.5
            self.in_layer1.backward(da)
            self.in_layer0.backward(da)
            return None
     
     def get_word_vecs(self):
            return self.word_vecs
