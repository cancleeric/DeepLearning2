
from util import preprocess

def main():
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)
    
    print("Corpus:", corpus)
    print("Word to ID:", word_to_id)
    print("ID to Word:", id_to_word)

if __name__ == '__main__':
    main()