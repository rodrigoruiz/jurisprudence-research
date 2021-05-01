
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import math
from Utilities import fill_with_zeros


# def plot_history(history):
#     print('Train accuracy (last): ' + str(history.history['acc'][-1]))
#     print('Validation accuracy (last): ' + str(history.history['val_acc'][-1]))
#     print()
#     print('Train accuracy (max): ' + str(max(history.history['acc'])))
#     print('Validation accuracy (max): ' + str(max(history.history['val_acc'])))
# 
#     # "Accuracy"
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc = 'upper left')
#     plt.show()
# 
#     # "Loss"
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'validation'], loc = 'upper left')
#     plt.show()

def plot_history(history, save_figure_path = None):
    metrics = list(filter(lambda s: 'val' not in s, history.keys()))
    
    has_training_metrics = len(metrics) != 0
    if not has_training_metrics:
        metrics = list(filter(lambda s: 'val_' in s, history.keys()))
        metrics = list(map(lambda s: s[4:], metrics))
    
    if has_training_metrics:
        epochs = range(1, len(history[metrics[0]]) + 1)
    else:
        epochs = range(1, len(history['val_' + metrics[0]]) + 1)
    
    def plot(data, label, max_or_min, ax, color = None):
        last = format(data[-1], '.5f')
        if max_or_min == 'max':
            best = format(max(data), '.5f')
        else:
            best = format(min(data), '.5f')
        
        ax.plot(range(1, len(data) + 1), data, label = f'{label} ( last: {last} | {max_or_min}: {best} )', color = color)
    
    subplots_per_row = 2
    number_of_rows = math.ceil(len(metrics) / subplots_per_row)
    
    fig = plt.figure(figsize = (12, 5 * number_of_rows))
    
    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(number_of_rows, subplots_per_row, i + 1)
        
        max_or_min = 'min' if 'loss' in metric else 'max'
        
        if has_training_metrics:
            plot(data = history[metric], label = 'Training', max_or_min = max_or_min, ax = ax, color = 'tab:blue')
        
        if 'val_' + metric in history:
            plot(data = history['val_' + metric], label = 'Validation', max_or_min = max_or_min, ax = ax, color = 'tab:orange')
        
        ax.set_title(metric)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric)
        ax.legend()
        
        fig.tight_layout()
    
    if save_figure_path:
        plt.savefig(save_figure_path)
    
    plt.show()

def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names,
    figsize = (10, 8),
    fontsize = 14,
    save_figure_path = None
) -> Figure:
    cm = fill_with_zeros(array = confusion_matrix, shape = (len(class_names), len(class_names)), dtype = np.int64)
    
    df_cm = pd.DataFrame(
        cm,
        index = class_names,
        columns = class_names
    )
    
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111)
    
    heatmap = sns.heatmap(df_cm, annot = True, ax = ax, fmt = 'd', cmap = 'Blues')
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation = 0, ha = 'right', fontsize = fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation = 0, ha = 'center', fontsize = fontsize)
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if cm.shape[0] == 2:
        precision_recall_text = (
            f'Precision: {cm[1, 1] / (cm[1, 1] + cm[0, 1])}\n' +
            f'Recall: {cm[1, 1] / (cm[1, 1] + cm[1, 0])}\n'
        )
    else:
        precision_recall_text = ''
    
    plt.text(
        0, 0,
        f'Accuracy: {cm.diagonal().sum() / cm.sum()}\n' + precision_recall_text,
        fontsize = fontsize
    )
    
    if save_figure_path:
        plt.savefig(save_figure_path)
    
    plt.show()
    
    return fig

def show_confusion_matrix_stats(
    model,
    x_data,
    y_data,
    threshold = 0.5,
    class_names = None,
    batch_size = None,
    save_figure_path = None
):
    predictions = model.predict(x_data, batch_size = batch_size) if batch_size else model.predict(x_data)
    predictions = (predictions > threshold).astype(int)
    
    cm = metrics.confusion_matrix(y_data, predictions)
    
    # print(f'Accuracy: {(cm[0, 0] + cm[1, 1]) / cm.sum()}')
    # print(f'Precision: {cm[1, 1] / (cm[1, 1] + cm[0, 1])}')
    # print(f'Recall: {cm[1, 1] / (cm[1, 1] + cm[1, 0])}')
    
    if not class_names:
        class_names = list(map(lambda n: str(n), range(2)))
    _ = plot_confusion_matrix(
        confusion_matrix = cm,
        class_names = class_names,
        save_figure_path = save_figure_path
    )
