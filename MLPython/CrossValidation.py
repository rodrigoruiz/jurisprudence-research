
import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from Metrics import metrics_from_confusion_matrix
from Utilities import flatten


class EnsembleModel:
    
    def __init__(self, models):
        self.models = models
    
    def predict(self, x):
        predictions = np.array(list(map(lambda model: model.predict(x), self.models)))
        predictions = np.mean(predictions, axis = 0)
        return predictions

def cross_validation_ensemble_summary(results):
    summary = 'Cross Validation\n'
    
    for data_type in ['training', 'validation', 'test']:
        data_metrics = list(map(lambda fold: metrics_from_confusion_matrix(fold[f'{data_type}_confusion_matrix']), results['cross_validation']))
        
        summary += f'  {data_type.capitalize()}\n'
        
        for metric_key in data_metrics[0].keys():
            cv_metric = np.array(list(map(lambda m: m[metric_key], data_metrics)))
            summary += f'  - {metric_key.capitalize()}: {cv_metric.mean()} ± {cv_metric.std()}\n'
    
    m = metrics_from_confusion_matrix(results['ensemble']['test_confusion_matrix'])
    test_accuracy = m['accuracy']
    test_precision = m['precision']
    test_recall = m['recall']
    
    summary += 'Ensemble - Test\n'
    summary += f'- Accuracy: {test_accuracy}\n'
    summary += f'- Precision: {test_precision}\n'
    summary += f'- Recall: {test_recall}\n'
    
    return summary

# Model must have the method "predict"
def cross_validation_ensemble(
    number_of_folds,
    x_training_validation_data,
    y_training_validation_data,
    x_test_data,
    y_test_data,
    create_and_train_model,
    has_multiple_inputs = False, # useful for siamese network
    show_progress = False,
    random_state = None
):
    if show_progress:
        print('Starting cross validation...')
        print(f'Cross validation fold: 0/{number_of_folds}')
    
    x_training_validation_data = np.array(x_training_validation_data)
    y_training_validation_data = np.array(y_training_validation_data)
    
    if has_multiple_inputs:
        transpose_axes = [1, 0] + list(range(2, len(x_training_validation_data.shape)))
        x_training_validation_data = x_training_validation_data.transpose(transpose_axes)
    
    kfold = StratifiedKFold(n_splits = number_of_folds, shuffle = True, random_state = random_state)
    
    def process_fold(splits):
        index, value = splits
        kfold_training_indexes, kfold_validation_indexes = value
        
        x_training_kfold = x_training_validation_data[kfold_training_indexes]
        y_training_kfold = y_training_validation_data[kfold_training_indexes]
        x_validation_kfold = x_training_validation_data[kfold_validation_indexes]
        y_validation_kfold = y_training_validation_data[kfold_validation_indexes]
        
        if has_multiple_inputs:
            x_training_kfold = x_training_kfold.transpose(transpose_axes).tolist()
            x_validation_kfold = x_validation_kfold.transpose(transpose_axes).tolist()
        
        model = create_and_train_model(
            x_training_kfold,
            y_training_kfold,
            x_validation_kfold,
            y_validation_kfold
        )
        
        training_predictions = (model.predict(x_training_kfold) > 0.5).astype(int)
        training_confusion_matrix = metrics.confusion_matrix(y_training_kfold, training_predictions)
        
        validation_predictions = (model.predict(x_validation_kfold) > 0.5).astype(int)
        validation_confusion_matrix = metrics.confusion_matrix(y_validation_kfold, validation_predictions)
        
        test_predictions = (model.predict(x_test_data) > 0.5).astype(int)
        test_confusion_matrix = metrics.confusion_matrix(y_test_data, test_predictions)
        
        if show_progress:
            print(f'Cross validation fold: {index + 1}/{number_of_folds}')
        
        return {
            'model': model,
            'training_confusion_matrix': training_confusion_matrix,
            'validation_confusion_matrix': validation_confusion_matrix,
            'test_confusion_matrix': test_confusion_matrix
        }
    
    cv_results = list(map(process_fold, enumerate(kfold.split(x_training_validation_data, y_training_validation_data))))
    
    ensemble_model = EnsembleModel(list(map(lambda r: r['model'], cv_results)))
    
    test_predictions = (ensemble_model.predict(x_test_data) > 0.5).astype(int)
    test_confusion_matrix = metrics.confusion_matrix(y_test_data, test_predictions)
    
    results = {
        'ensemble': {
            'model':ensemble_model,
            'test_confusion_matrix': test_confusion_matrix
        },
        'cross_validation': cv_results
    }
    results['summary'] = cross_validation_ensemble_summary(results)
    
    return results
