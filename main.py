
from util import preprocess , create_co_matrix

def main():
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    
    print("Corpus:", corpus)
    print("Word to ID:", word_to_id)
    print("ID to Word:", id_to_word)
    
    #產生共生矩陣
    print("Create Co-Matrix")
    print(create_co_matrix(corpus, len(word_to_id)))


if __name__ == '__main__':
    main()