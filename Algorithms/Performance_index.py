import pandas as pd
import numpy as np

# read csv
file_path = 'different_rfe_result.csv'
df = pd.read_csv(file_path)

# normalize the data
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# normoiolize the coloumns that are needed
df['Norm Test accuracy'] = normalize(df['TEST Accuracy'])
df['Norm Time rfe'] = normalize(df['Total rfe-time'])
df['Norm Time train'] = normalize(df['Total train-time'])
df['Norm Time total'] = normalize(df['Total runtime'])

# set the weights
weights = {
    'Norm Test accuracy': 1,
    'Norm Time rfe': 0,  # weight of time is negative
    'Norm Time train': 0,
    'Norm Time total': -1
}

# 计算综合性能评分
df['Performance Index'] = (
    weights['Norm Test accuracy'] * df['Norm Test accuracy'] +
    weights['Norm Time rfe'] * df['Norm Time rfe'] +
    weights['Norm Time train'] * df['Norm Time train'] +
    weights['Norm Time total'] * df['Norm Time total']
)
# 查看结果
df[['N_feature', 'Performance Index']].sort_values(by='Performance Index', ascending=False)
df.to_csv('diff_result_pi.csv', index=False)