
import csv
from datetime import datetime, timedelta
from functools import reduce
import json
import nltk
import numpy as np
import os
import pandas as pd
from PIL import Image
from Plot import plot_history, show_confusion_matrix_stats
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
import sys
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Bidirectional, concatenate, Dense, Dropout, Embedding, GlobalAveragePooling1D, Input, Lambda, LSTM
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence

# from keras_tqdm import TQDMNotebookCallback
# from tqdm.keras import TqdmCallback
from timeit import default_timer as timer
from transformers import TFAutoModel, AutoTokenizer

from Metrics import metrics_from_confusion_matrix
from PreTrainedEmbeddings import PreTrainedEmbedding
from Transformer import Encoder, PaddingMask
from Utilities import chunks, flat_map, get_tokenizer, load_json, save_json


bert_path = 'neuralmind/bert-base-portuguese-cased'
number_of_bert_hidden_units = 768

precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

stop_words = set(nltk.corpus.stopwords.words('portuguese'))

# text_processing_type: 'clean_report', 'clean_report_without_nltk_stopwords', 'None'
def preprocess_text(texts, text_processing_type):
    if text_processing_type == 'clean_report':
        return texts.apply(lambda text: ' '.join(text_to_word_sequence(text)))
    
    if text_processing_type == 'clean_report_without_nltk_stopwords':
        def is_alpha_or_space(s):
            return s.isalpha() or s.isspace()
        
        def clean_text(text):
            text = ''.join(filter(is_alpha_or_space, text))
            text = text_to_word_sequence(text)
            text = filter(lambda word: word not in stop_words, text)
            text = ' '.join(text)
            return text
        
        return texts.apply(clean_text)
    
    return texts

def create_text_processing_function(training_texts, parameters):
    training_texts = preprocess_text(training_texts, parameters.get('text_processing_type'))
    
    # only used for pre-trained embeddings
    tokenizer = None
    
    if parameters['representation_type'] == 'BoW':
        count_vectorizer = CountVectorizer(max_features = parameters['vocabulary_size']).fit(training_texts)
        
        def text_representation(texts):
            return count_vectorizer.transform(texts).toarray().tolist()
    
    if parameters['representation_type'] == 'TF-IDF':
        tfidf_vectorizer = TfidfVectorizer(max_features = parameters['vocabulary_size']).fit(training_texts)
        
        def text_representation(texts):
            return tfidf_vectorizer.transform(texts).toarray().tolist()
    
    if parameters['representation_type'] == 'Tokenized':
        tokenizer = get_tokenizer(
            texts = training_texts,
            vocabulary_size = parameters['vocabulary_size'],
            show_prints = False
        )
        
        def text_representation(texts):
            return pad_sequences(
                tokenizer.texts_to_sequences(texts),
                maxlen = parameters['maximum_sequence_length'],
                truncating = 'post'
            ).tolist()
    
    if parameters['representation_type'] == 'bert_tokenizer':
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        model = TFAutoModel.from_pretrained(bert_path, from_pt = True)
        
        def text_representation(texts):
            input_ids = texts.apply(lambda text: tokenizer.encode(text, add_special_tokens = True))
            input_ids = pad_sequences(
                input_ids,
                maxlen = parameters['maximum_sequence_length'],
                truncating = 'post'
            )
            
            # embeddings = np.array(list(map(lambda x: model(np.array([x]))[0][:, 0, :].numpy()[0], input_ids)))
            batch_size = 64
            
            def f(x):
                return model(x)[0][:, 0, :].numpy()
            embeddings = np.array(list(flat_map(f, chunks(input_ids, batch_size))))
            
            return embeddings
    
    def text_processing_function(texts):
        return text_representation(preprocess_text(texts, parameters.get('text_processing_type')))
    
    return text_processing_function, tokenizer

# lookup_data can be used for validation data to look for references in all data (training + validation)
def create_generator(
    text_processing_function,
    data,
    lookup_data = None,
    batch_size = 32,
    random_state = None
):
    if lookup_data is None:
        lookup_data = data
    
    half_batch_size = int(batch_size / 2)
    
    data_with_references =  data[data['known_references'].apply(lambda r: len(r) > 0)]
    
    def get_report_with_id(identifier):
        return lookup_data[lookup_data['id'] == identifier]['report'].values[0]
    
    if random_state:
        np.random.seed(random_state)
    
    while True:
        batch_with_references = data_with_references.sample(
            n = half_batch_size,
            replace = True,
            random_state = random_state + 1 if random_state else None
        )
        
        input_1 = text_processing_function(pd.Series(np.concatenate((
            batch_with_references['report'].values,
            data.sample(
                n = half_batch_size,
                replace = True,
                random_state = random_state + 2 if random_state else None
            )['report'].values
        ))))
        
        input_2 = text_processing_function(pd.Series(np.concatenate((
            batch_with_references['known_references'].apply(lambda r: get_report_with_id(np.random.choice(r))).values,
            lookup_data.sample(
                n = half_batch_size,
                replace = True,
                random_state = random_state + 3 if random_state else None
            )['report'].values
        ))))
        
        output = np.concatenate((np.ones(half_batch_size), np.zeros(half_batch_size)))
        output = output.reshape((output.shape[0], 1))
        
        yield [np.array(list(input_1)), np.array(list(input_2))], output


def get_pre_trained_embedding(embedding_type, tokenizer):
    if embedding_type == 'Glove50':
        return PreTrainedEmbedding(
            name = 'Glove',
            output_size = 50,
            word_index = tokenizer.word_index,
            file_path = 'C:/Users/Gamer/Desktop/AI/GlovePt/glove_s50.txt'
        )
    if embedding_type == 'Glove300':
        return PreTrainedEmbedding(
            name = 'Glove',
            output_size = 300,
            word_index = tokenizer.word_index,
            file_path = 'C:/Users/Gamer/Desktop/AI/GlovePt/glove_s300.txt'
        )
    
    if embedding_type == 'Word2Vec50':
        return PreTrainedEmbedding(
            name = 'Word2Vec',
            output_size = 50,
            word_index = tokenizer.word_index,
            file_path = 'C:/Users/Gamer/Desktop/AI/Word2VecPt/skip_s50.txt'
        )
    if embedding_type == 'Word2Vec300':
        return PreTrainedEmbedding(
            name = 'Word2Vec',
            output_size = 300,
            word_index = tokenizer.word_index,
            file_path = 'C:/Users/Gamer/Desktop/AI/Word2VecPt/skip_s300.txt'
        )
    
    if embedding_type == 'FastText50':
        return PreTrainedEmbedding(
            name = 'FastText',
            output_size = 50,
            word_index = tokenizer.word_index,
            file_path = 'C:/Users/Gamer/Desktop/AI/FastTextPt/skip_s50.txt'
        )
    if embedding_type == 'FastText300':
        return PreTrainedEmbedding(
            name = 'FastText',
            output_size = 300,
            word_index = tokenizer.word_index,
            file_path = 'C:/Users/Gamer/Desktop/AI/FastTextPt/skip_s300.txt'
        )
    
    raise Exception(f'Couldn\'t find embedding type: {embedding_type}')


def snn_output(output_type, features_1, features_2, dropout_rate = 0, regularizer_l1_rate = 0, regularizer_l2_rate = 0):
    if output_type == 'MLP':
        return reduce(lambda output, layer: layer(output), [
            concatenate([features_1, features_2]),
            Dropout(dropout_rate),
            Dense(
                units = 256,
                activation = 'relu',
                kernel_regularizer = regularizers.l1_l2(l1 = regularizer_l1_rate, l2 = regularizer_l2_rate)
            ),
            Dropout(dropout_rate),
            Dense(
                units = 256,
                activation = 'relu',
                kernel_regularizer = regularizers.l1_l2(l1 = regularizer_l1_rate, l2 = regularizer_l2_rate)
            ),
            Dropout(dropout_rate),
            Dense(
                units = 1,
                activation = 'sigmoid',
                kernel_regularizer = regularizers.l1_l2(l1 = regularizer_l1_rate, l2 = regularizer_l2_rate)
            )
        ])
    
    if output_type == 'Manhattan':
        # def exponent_neg_manhattan_distance(left, right):
        #     return K.exp(-K.sum(K.abs(left - right), axis = 1, keepdims = True))
        return Lambda(
            # function = lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
            function = lambda x: K.exp(-K.sum(K.abs(x[0] - x[1]), axis = 1, keepdims = True)),
            output_shape = lambda x: (x[0][0], 1)
        )([features_1, features_2])


def create_lr_model(vocabulary_size):
    input_1 = Input(shape = (vocabulary_size,))
    input_2 = Input(shape = (vocabulary_size,))
    
    output = reduce(lambda output, layer: layer(output), [
        concatenate([input_1, input_2]),
        Dense(
            units = 1,
            activation = 'sigmoid'
        )
    ])
    
    model = Model(inputs = [input_1, input_2], outputs = [output])
    
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy', precision, recall]
    )
    
    return model

def create_mlp_model(vocabulary_size):
    input_1 = Input(shape = (vocabulary_size,))
    input_2 = Input(shape = (vocabulary_size,))
    
    output = reduce(lambda output, layer: layer(output), [
        concatenate([input_1, input_2]),
        Dense(
            units = 256,
            activation = 'relu'
        ),
        Dense(
            units = 256,
            activation = 'relu'
        ),
        Dense(
            units = 1,
            activation = 'sigmoid'
        )
    ])
    
    model = Model(inputs = [input_1, input_2], outputs = [output])
    
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy', precision, recall]
    )
    
    return model

def create_snn_mlp_model(vocabulary_size, output_type, dropout_rate = 0, regularizer_l1_rate = 0, regularizer_l2_rate = 0):
    input_1 = Input(shape = (vocabulary_size,))
    input_2 = Input(shape = (vocabulary_size,))
    
    feature_model = Sequential([
        Dropout(dropout_rate),
        Dense(
            units = 256,
            activation = 'relu',
            kernel_regularizer = regularizers.l1_l2(l1 = regularizer_l1_rate, l2 = regularizer_l2_rate),
        ),
        Dropout(dropout_rate),
        Dense(
            units = 256,
            activation = 'relu',
            kernel_regularizer = regularizers.l1_l2(l1 = regularizer_l1_rate, l2 = regularizer_l2_rate),
        ),
        Dropout(dropout_rate),
        Dense(
            units = 256,
            activation = 'sigmoid',
            kernel_regularizer = regularizers.l1_l2(l1 = regularizer_l1_rate, l2 = regularizer_l2_rate),
        ),
        Dropout(dropout_rate),
    ])
    
    features_1 = feature_model(input_1)
    features_2 = feature_model(input_2)
    
    output = snn_output(output_type, features_1, features_2, dropout_rate, regularizer_l1_rate, regularizer_l2_rate)
    
    model = Model(inputs = [input_1, input_2], outputs = [output])
    
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy', precision, recall]
    )
    
    return model

def create_snn_lstm_model(
    pre_trained_embedding,
    tokenizer,
    vocabulary_size,
    maximum_sequence_length,
    output_type,
    batch_size
):
    input_1 = Input(shape = (maximum_sequence_length,))
    input_2 = Input(shape = (maximum_sequence_length,))
    
    if pre_trained_embedding == None:
        embedding = Embedding(vocabulary_size, 256)
    else:
        embedding = get_pre_trained_embedding(pre_trained_embedding, tokenizer)
    
    feature_model = Sequential([
        embedding,
        Bidirectional(LSTM(
            units = 256,
            return_sequences = True
        )),
        Bidirectional(LSTM(
            units = 256,
            return_sequences = True
        )),
        Bidirectional(LSTM(
            units = 256
        ))
    ])
    
    features_1 = feature_model(input_1)
    features_2 = feature_model(input_2)
    
    output = snn_output(output_type, features_1, features_2)
    
    model = Model(inputs = [input_1, input_2], outputs = [output])
    
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy', precision, recall]
    )
    
    return model

# transformer_parameters = {
#     num_layers = 6,
#     d_model = 512,
#     num_heads = 8,
#     d_ff = 2048,
#     dropout_rate = 0.1
# }
def create_snn_transformer_model(
    pre_trained_embedding,
    tokenizer,
    vocabulary_size,
    maximum_sequence_length,
    output_type,
    transformer_parameters
):
    input_1 = Input(shape = (maximum_sequence_length,))
    input_2 = Input(shape = (maximum_sequence_length,))
    
    if pre_trained_embedding == None:
        embedding = None
    else:
        embedding = get_pre_trained_embedding(pre_trained_embedding, tokenizer)
    
    # Paper "Attention Is All You Need" parameters (https://arxiv.org/abs/1706.03762)
    # encoder = Encoder(
    #     num_layers = 6,
    #     d_model = 512,
    #     num_heads = 8,
    #     d_ff = 2048,
    #     vocab_size = vocabulary_size,
    #     dropout_rate = 0.1
    # )
    encoder = Encoder(
        pre_trained_embedding = embedding,
        num_layers = transformer_parameters['num_layers'],
        d_model = transformer_parameters['d_model'], # Embedding size
        num_heads = transformer_parameters['num_heads'],
        d_ff = transformer_parameters['d_ff'],
        vocab_size = vocabulary_size,
        dropout_rate = transformer_parameters['dropout_rate']
    )
    def feature_model(input_n):
        padding_mask = PaddingMask()(input_n)
        encoder_output, encoder_attention = encoder(input_n, padding_mask)
        return K.sum(encoder_output, axis = 1) / K.sum(K.cast_to_floatx(padding_mask[0] == 0))
    
    features_1 = feature_model(input_1)
    features_2 = feature_model(input_2)
    
    output = snn_output(output_type, features_1, features_2)
    
    model = Model(inputs = [input_1, input_2], outputs = [output])
    
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy', precision, recall]
    )
    
    return model

def create_bert_model(output_type):
    input_1 = Input(shape = (number_of_bert_hidden_units,))
    input_2 = Input(shape = (number_of_bert_hidden_units,))
    
    output = snn_output(output_type, input_1, input_2)
    
    model = Model(inputs = [input_1, input_2], outputs = [output])
    
    model.compile(
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics = ['accuracy', precision, recall]
    )
    
    return model

def create_model(parameters, tokenizer):
    if parameters['type'] == 'LR':
        return create_lr_model(
            vocabulary_size = parameters['vocabulary_size']
        )
    
    if parameters['type'] == 'MLP':
        return create_mlp_model(
            vocabulary_size = parameters['vocabulary_size']
        )
    
    if parameters['type'] == 'SNN + MLP':
        return create_snn_mlp_model(
            vocabulary_size = parameters['vocabulary_size'],
            output_type = parameters['snn_output_type'],
            dropout_rate = parameters.get('regularization', {}).get('dropout_rate', 0),
            regularizer_l1_rate = parameters.get('regularization', {}).get('regularizer_l1_rate', 0),
            regularizer_l2_rate = parameters.get('regularization', {}).get('regularizer_l2_rate', 0)
        )
    
    if parameters['type'] == 'SNN + LSTM':
        return create_snn_lstm_model(
            pre_trained_embedding = parameters.get('pre_trained_embedding'),
            tokenizer = tokenizer,
            vocabulary_size = parameters['vocabulary_size'],
            maximum_sequence_length = parameters['maximum_sequence_length'],
            output_type = parameters['snn_output_type'],
            batch_size = parameters['training_batch_size']
        )
    
    if parameters['type'] == 'SNN + Transformer':
        return create_snn_transformer_model(
            pre_trained_embedding = parameters.get('pre_trained_embedding'),
            tokenizer = tokenizer,
            vocabulary_size = parameters['vocabulary_size'],
            maximum_sequence_length = parameters['maximum_sequence_length'],
            output_type = parameters['snn_output_type'],
            transformer_parameters = parameters['transformer_parameters']
        )
    
    if parameters['type'] == 'BERT':
        return create_bert_model(
            output_type = parameters['snn_output_type']
        )


def load_experiment_data(data_path, parameters):
    training_cases = pd.read_csv(data_path + '/training_cases.csv', usecols = ['id', 'known_references', 'report'])
    training_cases['known_references'] = training_cases['known_references'].str.replace('\'', '"').apply(json.loads)
    
    validation_cases = pd.read_csv(data_path + '/validation_cases.csv', usecols = ['id', 'known_references', 'report'])
    validation_cases['known_references'] = validation_cases['known_references'].str.replace('\'', '"').apply(json.loads)
    
    text_processing_function, tokenizer = create_text_processing_function(training_cases['report'], parameters)
    
    training_generator = create_generator(
        text_processing_function = text_processing_function,
        data = training_cases,
        batch_size = parameters['training_batch_size']
    ) if 'training_batch_size' in parameters else None
    
    validation_data = next(create_generator(
        text_processing_function = text_processing_function,
        data = validation_cases,
        lookup_data = pd.concat([validation_cases, training_cases]),
        batch_size = parameters['validation_batch_size'],
        random_state = 7
    ))
    
    return training_generator, validation_data, tokenizer


# parameters = {
#     'text_processing_type': 'clean_report_without_nltk_stopwords',
#     'representation_type': 'BoW',
#     'type': 'NB',
#     'vocabulary_size': 10000,
#     'alpha': 1.0,
#     'training_batch_size': 16,
#     'validation_batch_size': 30000,
#     'steps_per_epoch': 2000,
#     'epochs': 10
# }
def nb_experiment(data_path, parameters, experiment_folder, initial_epoch):
    if should_skip_training(parameters, experiment_folder, initial_epoch):
        print('Skipping training')
        read_experiment(
            data_path = data_path,
            folder_name = experiment_folder.split('/')[-1],
            show_min_val_loss = False,
            show_training_time = True,
            show_parameters = False,
            show_history = False,
            plot_metrics = True,
            show_confusion_matrix = True,
            resave_images = True
        )
        return
    
    model = MultinomialNB(alpha = parameters.get('alpha', 1.0))
    
    previous_model_folder = parameters.get('previous_model_folder')
    if previous_model_folder:
        model.set_params(load_json(f'{data_path}/Results/{previous_model_folder}/weights.best.json'))
    
    training_generator, validation_data, tokenizer = load_experiment_data(data_path, parameters)
    val_x = validation_data[0]
    val_x = np.concatenate(val_x, axis = 1)
    val_y = validation_data[1]
    val_y = val_y.reshape(val_y.shape[0],)
    
    history = {
        'epoch': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': []
    }
    with open(experiment_folder + '/history.csv', 'w', newline = '') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter = ',', quotechar = '"', quoting = csv.QUOTE_MINIMAL)
        if not previous_model_folder:
            csv_writer.writerow(['epoch', 'val_accuracy', 'val_precision', 'val_recall'])
        
        epochs = parameters['epochs']
        
        for epoch in range(epochs):
            for step in range(parameters['steps_per_epoch']):
                x, y = next(training_generator)
                x = np.concatenate(x, axis = 1)
                y = y.reshape(y.shape[0],)
                model.partial_fit(x, y, classes = [0, 1])
            
            p = model.predict(val_x)
            
            cm = metrics.confusion_matrix(val_y, p)
            m = metrics_from_confusion_matrix(cm)
            val_accuracy = m['accuracy']
            val_precision = m['precision']
            val_recall = m['recall']
            
            history['epoch'].append(epoch)
            history['val_accuracy'].append(val_accuracy)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            
            csv_writer.writerow([epoch, val_accuracy, val_precision, val_recall])
            
            save_json(data = model.get_params(), file_path = f'{experiment_folder}/weights.best.json')
            
            steps_per_epoch = parameters['steps_per_epoch']
            print(f'Epoch {epoch+1}/{epochs}')
            print(f'Steps: {steps_per_epoch} - val_accuracy: {val_accuracy} - val_precision: {val_precision} - val_recall: {val_recall}')
    
    history.pop('epoch', None)
    
    plot_history(
        history = history,
        save_figure_path = experiment_folder + '/history.png'
    )
    
    best_model = model
    
    print('Validation confusion matrix')
    show_confusion_matrix_stats(
        model = best_model,
        x_data = val_x,
        y_data = val_y,
        save_figure_path = experiment_folder + '/validation_confusion_matrix.png'
    )

# parameters = {
#     'text_processing_type': 'clean_report_without_nltk_stopwords',
#     'representation_type': 'BoW',
#     'type': 'CS',
#     'vocabulary_size': 10000,
#     'validation_batch_size': 30000
# }
def cs_experiment(data_path, parameters, experiment_folder):
    training_generator, validation_data, tokenizer = load_experiment_data(data_path, parameters)
    
    val_x = validation_data[0]
    val_y = validation_data[1]
    val_y = val_y.reshape(val_y.shape[0],)
    
    class CSModel:
        
        def __init__(self, threshold = 0.3):
            self.threshold = threshold
        
        def predict(self, x):
            return (np.diag(cosine_similarity(x[0], x[1])) > self.threshold).astype(int)
    
    print('Validation confusion matrix')
    print()
    
    for threshold in np.arange(0.1, 1, 0.1):
        model = CSModel(threshold)
        
        print('Threshold: %.1f' % threshold)
        show_confusion_matrix_stats(
            model = model,
            x_data = val_x,
            y_data = val_y,
            save_figure_path = experiment_folder + '/validation_confusion_matrix_%.1f.png' % threshold
        )


def should_skip_training(parameters, experiment_folder, initial_epoch):
    previous_model_folder = parameters.get('previous_model_folder')
    if not previous_model_folder:
        return False
    
    epochs = parameters['epochs']
    
    return initial_epoch >= epochs

def get_initial_epoch(parameters, experiment_folder):
    previous_model_folder = parameters.get('previous_model_folder')
    if previous_model_folder:
        return pd.read_csv(f'{experiment_folder}/history.csv').shape[0]
    
    return 0

def print_epoch_with_min_val_loss(experiment_folder):
    history = pd.read_csv(f'{experiment_folder}/history.csv')
    row = history.iloc[history['val_loss'].argmin()]
    val_loss = row['val_loss']
    epoch = int(row['epoch']) + 1
    
    print(f'Minimum val_loss: {val_loss} - Epoch: {epoch}')

# parameters = {
#     'text_processing_type': 'clean_report',
#     'representation_type': 'BoW',
#     'type': 'MLP',
#     'snn_output_type': 'MLP',
#     'vocabulary_size': 10000,
#     'training_batch_size': 16,
#     'validation_batch_size': 30000,
#     'steps_per_epoch': 500,
#     'epochs': 20,
#     'previous_model_folder': None
# }
# parameters = {
#     'text_processing_type': 'clean_report',
#     'representation_type': 'Tokenized',
#     'type': 'SNN + LSTM',
#     'snn_output_type': 'MLP',
#     'pre_trained_embedding': None, # None, 'Word2Vec', 'Glove', 'FastText'
#     'vocabulary_size': 10000,
#     'maximum_sequence_length': 1000,
#     'training_batch_size': 16,
#     'validation_batch_size': 30000,
#     'steps_per_epoch': 500,
#     'epochs': 20,
#     'previous_model_folder': None
# }
# parameters = {
#     'text_processing_type': None,
#     'representation_type': 'bert_tokenizer',
#     'type': 'BERT',
#     'snn_output_type': 'MLP',
#     'maximum_sequence_length': 1000,
#     'training_batch_size': 16,
#     'validation_batch_size': 30000,
#     'steps_per_epoch': 500,
#     'epochs': 20,
#     'previous_model_folder': None
# }
data_cache = None
def keras_experiment(data_path, parameters, experiment_folder, initial_epoch, use_data_cache):
    if should_skip_training(parameters, experiment_folder, initial_epoch):
        print('Skipping training')
        read_experiment(
            data_path = data_path,
            folder_name = experiment_folder.split('/')[-1],
            show_min_val_loss = True,
            show_training_time = True,
            show_parameters = False,
            show_history = False,
            plot_metrics = True,
            show_confusion_matrix = True,
            resave_images = True
        )
        return
    
    global data_cache
    if (not data_cache) or (not use_data_cache):
        data_cache = load_experiment_data(data_path, parameters)
    
    training_generator, validation_data, tokenizer = data_cache
    
    model = create_model(parameters, tokenizer)
    
    best_model_filepath = f'{experiment_folder}/weights.best.hdf5'
    last_model_filepath = f'{experiment_folder}/weights.last.hdf5'
    
    best_callback = keras.callbacks.ModelCheckpoint(filepath = best_model_filepath, save_best_only = True)
    internal_filepath = f'{experiment_folder}/internal.json'
    
    previous_model_folder = parameters.get('previous_model_folder')
    if previous_model_folder:
        model.load_weights(best_model_filepath)
        val_loss = model.evaluate(
            x = validation_data[0],
            y = validation_data[1],
            batch_size = 16,
            return_dict = True
        )['loss']
        best_callback.best = val_loss
        print(f'val_loss: {val_loss}')
        
        model.load_weights(last_model_filepath)
    # ----------------------------------------------------------------------------------------
    
    history = model.fit(
        x = training_generator, 
        steps_per_epoch = parameters['steps_per_epoch'],
        epochs = parameters['epochs'],
        verbose = 1,
        validation_data = validation_data,
        callbacks = [
            # TQDMNotebookCallback(),
            # TqdmCallback(verbose = 2),
            best_callback,
            keras.callbacks.ModelCheckpoint(filepath = last_model_filepath, save_best_only = False),
            keras.callbacks.CSVLogger(filename = experiment_folder + '/history.csv', append = True)
        ],
        initial_epoch = initial_epoch
    ).history
    
    print_epoch_with_min_val_loss(experiment_folder)
    
    history = pd.read_csv(f'{experiment_folder}/history.csv').drop(columns = ['epoch']).to_dict(orient = 'list')
    
    plot_history(
        history = history,
        save_figure_path = experiment_folder + '/history.png'
    )
    
    model.load_weights(best_model_filepath)
    
    print('Validation confusion matrix')
    show_confusion_matrix_stats(
        model = model,
        x_data = validation_data[0],
        y_data = validation_data[1],
        batch_size = 16,
        save_figure_path = experiment_folder + '/validation_confusion_matrix.png'
    )


# text_processing_type: 'clean_report', 'clean_report_without_nltk_stopwords', None
# representation_type: 'BoW', 'TF-IDF', 'Tokenized', 'bert_tokenizer'
# type: 'NB', 'CS', 'LR' 'MLP', 'SNN + MLP', 'SNN + LSTM', 'SNN + Transformer', 'BERT'
# snn_output_type: 'MLP', 'Manhattan'
def run_experiment(
    data_path,
    parameters,
    use_data_cache = False
):
    start_time = timer()
    
    previous_model_folder = parameters.get('previous_model_folder')
    if previous_model_folder:
        experiment_folder = os.path.expanduser(f'{data_path}/Results/{previous_model_folder}')
    else:
        date_string = datetime.now().isoformat().replace(':', '-')
        experiment_folder = os.path.expanduser(f'{data_path}/Results/{date_string}')
        os.mkdir(experiment_folder)
    
    print(experiment_folder)
    
    initial_epoch = get_initial_epoch(parameters, experiment_folder)
    
    save_json(data = parameters, file_path = f'{experiment_folder}/parameters.json')
    
    if parameters['type'] == 'NB':
        nb_experiment(data_path, parameters, experiment_folder, initial_epoch)
    if parameters['type'] == 'CS':
        cs_experiment(data_path, parameters, experiment_folder)
    if parameters['type'] in ['LR', 'MLP', 'SNN + MLP', 'SNN + LSTM', 'SNN + Transformer', 'BERT']:
        keras_experiment(data_path, parameters, experiment_folder, initial_epoch, use_data_cache)
    
    end_time = timer()
    
    if should_skip_training(parameters, experiment_folder, initial_epoch):
        return
    
    notes_filepath = f'{experiment_folder}/notes.json'
    
    if previous_model_folder:
        notes = load_json(notes_filepath)
    else:
        notes = {}
    
    training_time_key = 'training_time'
    epochs = parameters.get('epochs')
    if epochs:
        training_time_key = f'{training_time_key} ({initial_epoch} => {epochs})'
    
    notes[training_time_key] = str(timedelta(seconds = end_time - start_time))
    save_json(data = notes, file_path = notes_filepath)
    
    training_time = notes[training_time_key]
    print(f'Training time: {training_time}')

def read_experiment(
    data_path,
    folder_name,
    show_min_val_loss = False,
    show_training_time = False,
    show_parameters = False,
    show_history = False,
    plot_metrics = None,
    show_confusion_matrix = False,
    use_full_history = False,
    resave_images = False
):
    experiment_path = f'{data_path}/Results/{folder_name}'
    
    if show_min_val_loss:
        print_epoch_with_min_val_loss(experiment_path)
    
    if show_training_time:
        notes = load_json(f'{experiment_path}/notes.json')
        print(f'Training times')
        for key, value in notes.items():
            print(f'{key}: {value}')
    
    parameters = load_json(f'{experiment_path}/parameters.json')
    
    if show_parameters:
        print(json.dumps(parameters, indent = 4))
    
    history = pd.read_csv(f'{experiment_path}/history.csv')
    
    if use_full_history:
        previous_model_folder = parameters.get('previous_model_folder')
        
        while previous_model_folder:
            previous_history = pd.read_csv(f'{data_path}/Results/{previous_model_folder}/history.csv')
            history = pd.concat([history, previous_history])
            previous_model_folder = parameters.get('previous_model_folder')
    
    history = history.drop(columns = ['epoch'])
    
    if show_history:
        display(history)
    
    if plot_metrics:
        if plot_metrics == True:
            plot_history(
                history = history.to_dict(orient = 'list'),
                save_figure_path = f'{experiment_path}/history.png' if resave_images else None
            )
        else:
            plot_history(
                history = history[plot_metrics].to_dict(orient = 'list'),
                save_figure_path = f'{experiment_path}/history.png' if resave_images else None
            )
    
    if show_confusion_matrix:
        print('Validation confusion matrix')
        display(Image.open(f'{experiment_path}/validation_confusion_matrix.png'))

def run_test(data_path, folder_name, test_batch_size):
    experiment_folder = os.path.expanduser(f'{data_path}/Results/{folder_name}')
    parameters = load_json(f'{experiment_folder}/parameters.json')
    
    training_cases = pd.read_csv(data_path + '/training_cases.csv', usecols = ['id', 'known_references', 'report'])
    training_cases['known_references'] = training_cases['known_references'].str.replace('\'', '"').apply(json.loads)
    
    validation_cases = pd.read_csv(data_path + '/validation_cases.csv', usecols = ['id', 'known_references', 'report'])
    validation_cases['known_references'] = validation_cases['known_references'].str.replace('\'', '"').apply(json.loads)
    
    test_cases = pd.read_csv(data_path + '/test_cases.csv', usecols = ['id', 'known_references', 'report'])
    test_cases['known_references'] = test_cases['known_references'].str.replace('\'', '"').apply(json.loads)
    
    text_processing_function, tokenizer = create_text_processing_function(training_cases['report'], parameters)
    
    test_data = next(create_generator(
        text_processing_function = text_processing_function,
        data = test_cases,
        lookup_data = pd.concat([test_cases, validation_cases, training_cases]),
        batch_size = test_batch_size,
        random_state = 7
    ))
    
    best_model_filepath = f'{experiment_folder}/weights.best.hdf5'
    
    model = create_model(parameters, tokenizer)
    model.load_weights(best_model_filepath)
    
    show_confusion_matrix_stats(
        model = model,
        x_data = test_data[0],
        y_data = test_data[1],
        batch_size = 16,
        save_figure_path = experiment_folder + '/test_confusion_matrix.png'
    )
