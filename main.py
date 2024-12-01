from common.util import preprocess, create_co_matrix, cos_similarity, most_similar, ppmi
import numpy as np
import matplotlib.pyplot as plt
from display import display_word_vectors

def main():
    text = 'You say goodbye and I say hello.'
    display_word_vectors(text)

if __name__ == '__main__':
    main()