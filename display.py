from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.extmath import randomized_svd

def display_word_vectors(text):
    corpus, word_to_id, id_to_word = preprocess(text)
    
    print("Corpus:", corpus)
    print("Word to ID:", word_to_id)
    print("ID to Word:", id_to_word)
    
    # 產生共生矩陣
    vocab_size = len(word_to_id)
    co_matrix = create_co_matrix(corpus, vocab_size, window_size=1)
    print("Co-Matrix:")
    print(co_matrix)
    
    # 計算 PPMI 矩陣
    ppmi_matrix = ppmi(co_matrix)
    print("PPMI Matrix:")
    print(ppmi_matrix)
    
    # 使用 sklearn 的 randomized_svd 壓縮 PPMI 矩陣
    U, S, V = randomized_svd(ppmi_matrix, n_components=2)
    word_vecs = U[:, :2]  # 取前兩個主成分
    print("Word Vectors after SVD:")
    print(word_vecs)
    
    # 繪製圖表
    for word, word_id in word_to_id.items():
        plt.annotate(word, (word_vecs[word_id, 0], word_vecs[word_id, 1]))
    plt.scatter(word_vecs[:, 0], word_vecs[:, 1], alpha=0.5)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Word Vectors after SVD')
    plt.show()
    
    # 計算餘弦相似度
    c0 = co_matrix[word_to_id['you']]
    c1 = co_matrix[word_to_id['i']]
    similarity = cos_similarity(c0, c1)
    print(f"Cosine similarity between 'you' and 'i': {similarity}")
    
    # 查找與 "you" 最相似的詞
    print("Most similar words to 'you':")
    most_similar('you', word_to_id, id_to_word, co_matrix, top=3)