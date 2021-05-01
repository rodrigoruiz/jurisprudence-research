
import json
import numpy as np
from functools import reduce
from itertools import chain
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from os.path import dirname, abspath


def save_json(data, file_path = 'data.json'):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent = 4)

def load_json(file_path = 'data.json'):
    with open(file_path) as file:
        return json.load(file)

def flat_map(f, items):
    return chain.from_iterable(map(f, items))

def flatten(items):
    return chain.from_iterable(items)

def remove(element, l):
    new_list = list(l)
    new_list.remove(element)
    return new_list

def get(key, default = None, obj = None):
    def f(obj):
        if obj == None:
            return default
        
        if isinstance(obj, list) and key < len(obj):
            return obj[key]
        
        if isinstance(obj, dict):
            return obj.get(key, default)
        
        return default
    
    if obj == None:
        return f
    else:
        return f(obj)

def SequentialModel(layers):
    y = reduce(lambda output, layer: layer(output), layers)
    return Model(layers[0], y)

def balance_data(data, y_key, random_state = None):
    sample_size = data.groupby(y_key).size().max()
    
    return data.groupby(y_key) \
        .apply(lambda x: x.sample(sample_size, replace = True, random_state = random_state)) \
        .sample(frac = 1, random_state = random_state) \
        .reset_index(drop = True)

def get_tokenizer(texts, vocabulary_size = None, show_prints = True):
    tokenizer = Tokenizer(num_words = vocabulary_size)
    tokenizer.fit_on_texts(texts)
    
    if show_prints:
        maximum_vocabulary_size = len(tokenizer.word_index) + 1
        print(f'Maximum vocabulary size: {maximum_vocabulary_size}')
        
        words_length = list(map(len, tokenizer.texts_to_sequences(texts)))
        print(f'Minimum sentence length: {np.min(words_length)}')
        print(f'Maximum sentence length: {np.max(words_length)}')
    
    # data['question1_words_length'] = list(map(len, tokenizer.texts_to_sequences(data['question1'])))
    # data['question2_words_length'] = list(map(len, tokenizer.texts_to_sequences(data['question2'])))
    
    # minimum_sentence_length = 2
    # data = data[data['question1_words_length'] >= minimum_sentence_length]
    # data = data[data['question2_words_length'] >= minimum_sentence_length]
    
    # training_question_1 = sequence.pad_sequences(
    #     tokenizer.texts_to_sequences(training['question1']),
    #     maxlen = maximum_sequence_length,
    #     truncating = 'post'
    # ).tolist()
    
    return tokenizer

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def correlation_2_columns(data, col_1, col_2):
    res = data.groupby([col_1, col_2]).size().unstack()
    res['percentage'] = (res[res.columns[1]] / (res[res.columns[0]] + res[res.columns[1]]))
    return res

def fill_with_zeros(array, shape, dtype = None):
    z = np.zeros(shape = shape, dtype = dtype)
    z[:array.shape[0], :array.shape[1]] = array
    return z

def chunks(array, size):
    for i in range(0, array.shape[0], size):
        yield array[i:i+size]
