
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding
import numpy as np
import os


loaded_embeddings_index = {}

def get_embeddings_index(name, output_size, file_path = None):
    if name not in loaded_embeddings_index:
        loaded_embeddings_index[name] = {}
    
    if output_size not in loaded_embeddings_index[name]:
        print(f'Loading pre-trained embedding of size {output_size}...')
        
        embeddings_index = dict()
        
        if not file_path:
            file_path = f'{os.path.dirname(__file__)}/glove.6B/glove.6B.{output_size}d.txt'
        
        with open(file_path, encoding = 'utf8') as f:
            number_of_words, embedding_size = f.readline().rstrip().split(' ')
            print(f'Number of words: {number_of_words}, Embedding size: {embedding_size}')
            
            for line in f:
                values = line.rstrip().split(' ')
                word = values[0]
                coefs = np.asarray(values[1:], dtype = 'float32')
                embeddings_index[word] = coefs
        
        loaded_embeddings_index[name][output_size] = embeddings_index
        
        print(f'Loaded {len(embeddings_index)} word vectors.')
    
    return loaded_embeddings_index[name][output_size]

# output_size: 50, 100, 200, 300
# word_index: word_index from Keras Tokenizer
def GloveEmbedding(output_size, word_index, glove_file_path = None):
    embeddings_index = get_embeddings_index('Glove', output_size, glove_file_path)
    
    # +1 for the padding
    maximum_vocabulary_size = len(word_index) + 1
    
    embedding_matrix = np.zeros((maximum_vocabulary_size, output_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return Embedding(maximum_vocabulary_size, output_size, weights = [embedding_matrix], trainable = False)

# output_size: 50, 100, 200, 300
# word_index: word_index from Keras Tokenizer
# name: used in conjunction with output_size only for memoization when loading the embedding
def PreTrainedEmbedding(name, output_size, word_index, file_path = None):
    embeddings_index = get_embeddings_index(name, output_size, file_path)
    
    # +1 for the padding
    maximum_vocabulary_size = len(word_index) + 1
    
    embedding_matrix = np.zeros((maximum_vocabulary_size, output_size))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return Embedding(maximum_vocabulary_size, output_size, weights = [embedding_matrix], trainable = False)