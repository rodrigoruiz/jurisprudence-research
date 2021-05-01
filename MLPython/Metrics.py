
from tensorflow.keras import backend as K
from Utilities import fill_with_zeros
import sys
import numpy as np
import tensorflow as tf


def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
        y_pred = y_pred[:, 1:2]
        y_true = y_true[:, 1:2]
    
    return y_true, y_pred

def precision(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred): # sensitivity
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def f1(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def metrics_from_confusion_matrix(cm):
    cm = fill_with_zeros(np.array(cm), (2, 2))
    
    return {
        'accuracy': cm.diagonal().sum() / (cm.sum() + sys.float_info.epsilon),
        'precision': cm[1, 1] / (cm[1, 1] + cm[0, 1] + sys.float_info.epsilon),
        'recall': cm[1, 1] / (cm[1, 1] + cm[1, 0] + sys.float_info.epsilon)
    }

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
