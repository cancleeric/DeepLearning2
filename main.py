from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi

def main():
    text = 'You say goodbye and I say hello.'
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
    
    # 計算餘弦相似度
    c0 = co_matrix[word_to_id['you']]
    c1 = co_matrix[word_to_id['i']]
    similarity = cos_similarity(c0, c1)
    print(f"Cosine similarity between 'you' and 'i': {similarity}")
    
    # 查找與 "you" 最相似的詞
    print("Most similar words to 'you':")
    most_similar('you', word_to_id, id_to_word, co_matrix, top=3)

if __name__ == '__main__':
    main()