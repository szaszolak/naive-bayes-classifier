
# coding: utf-8

import pandas as pd
import numpy as np

file_name = input('Enter csv file name: ')
chess = pd.read_csv(file_name)


columns = list(chess)

mapers = {}
for column in columns:
    values = chess[column].dropna(inplace=False).unique()
    maper = {}
    for index, value in enumerate(values):
        maper[value]=index
    mapped = chess[column].map(maper, na_action='ignore')
    chess[column] = mapped
    mapers[column] = maper

non_empty = chess.dropna(axis=0, inplace=False)

missing_columns = chess[chess.isnull().any(axis=1)]

targets = chess.columns[chess.isnull().any().nonzero()]

def bayes_propability(data_frame_slice_positive, data_frame_slice_negative, total_rows, target_row):
    positive_rows = data_frame_slice_positive.shape[0]
    negative_rows = data_frame_slice_negative.shape[0]
    positive_propabilities = np.array(np.zeros(target_row.columns.shape[0] + 1))
    negative_propabilities = np.array(np.zeros(target_row.columns.shape[0] + 1))
    
    for index, feature in enumerate(target_row.columns):
        positive_propability = data_frame_slice_positive[data_frame_slice_positive[feature]==target_row[feature].values[0]].shape[0] / positive_rows
        negative_propability = data_frame_slice_negative[data_frame_slice_negative[feature]==target_row[feature].values[0]].shape[0] / negative_rows
        positive_propabilities[index] = positive_propability
        negative_propabilities[index] = negative_propability
    
    negative_propabilities[target_row.columns.shape[0]] = negative_rows / total_rows
    positive_propabilities[target_row.columns.shape[0]] = positive_rows / total_rows
    
    positive_product =  np.prod(positive_propabilities)
    negative_product =  np.prod(negative_propabilities)
    return positive_product / (positive_product + negative_product)


for target in targets:
    maper = mapers[target]
    for label in maper:
        value = maper[label]
        features = chess.columns[~chess.columns.isin(targets)]
        non_empty[non_empty[target]==value][features]
        print(target, label)
        print(bayes_propability(non_empty[non_empty[target]==value][features], 
                          non_empty[non_empty[target]!=value][features],
                          non_empty.shape[0],
                          missing_columns[features]))
        
