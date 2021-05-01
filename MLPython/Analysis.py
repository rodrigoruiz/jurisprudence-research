
import numpy as np
import pandas as pd


def print_data(data):
    print('-- Data shape --')
    print(data.shape)
    print()
    
    print('-- Columns statistics --')
    
    # print('-- Missing data --')
    s = data.isnull().sum().to_frame('missing_data_count')
    s['missing_data_percentage_from_total'] = (s['missing_data_count'] / data.shape[0] * 100).round(decimals = 2).astype(str) + '%'
    # display(s)
    
    # print('-- Unique values --')
    u = data.nunique().to_frame('unique_values_count')
    u['unique_values_percentage_from_total'] = (u['unique_values_count'] / data.shape[0] * 100).round(decimals = 2).astype(str) + '%'
    # display(u)
    
    display(pd.concat([s, u], axis = 1))
    
    display(data.head())

def print_class_count(data):
    counts = np.unique(data, return_counts = True)
    class_counts = dict(list(zip(counts[0], counts[1])))
    total = counts[1].sum()
    
    for key, value in class_counts.items():
        print(f'{key}: {value} - {100 * value / total}%')

def test():
    print('hi2')
