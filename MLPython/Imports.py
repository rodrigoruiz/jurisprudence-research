
import pandas as pd
import numpy as np
pd.options.display.max_columns = None

# full screen width
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

# display all output, instead of just last line
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
sns.set(color_codes = True)

import os

# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras import layers, optimizers, regularizers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Activation, Input, Dense, Dropout, Embedding, concatenate, Lambda, Bidirectional, Layer, LeakyReLU, SpatialDropout1D, TimeDistributed
# from keras.layers import merge
# from keras.layers import LSTM
# from keras.layers import CuDNNLSTM as LSTM
# from tensorflow.keras.models import load_model, Model
# from functools import reduce
# from nltk.corpus import stopwords
# import nltk
# nltk.download('stopwords')
# from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
# from tensorflow.keras.preprocessing import sequence
# from sklearn.metrics import confusion_matrix
# from sklearn.model_selection import train_test_split
# from sklearn.utils.class_weight import compute_class_weight

import sys
# sys.path.insert(0, 'C:/Users/Gamer/Desktop/AI')
sys.path.insert(0, 'C:/Users/Gamer/Dropbox/Projetos/MLPython')

# from plot_confusion_matrix import plot_confusion_matrix
# from PreTrainedEmbeddings import GloveEmbedding
# from TrainingValidationTestSplit import training_validation_test_split
