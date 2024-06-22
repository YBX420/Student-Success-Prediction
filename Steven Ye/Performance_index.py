import pandas as pd
import numpy as np

# 读取CSV文件
file_path = 'result_rfe.csv'
df = pd.read_csv(file_path)

# 标准化函数
def normalize(series):
    return (series - series.min()) / (series.max() - series.min())

# 对需要的列进行标准化处理
df['Norm Best cross'] = normalize(df['Best cross-validation score'])
df['Norm Validation accuracy'] = normalize(df['Validation accuracy'])
df['Norm Test accuracy'] = normalize(df['Test accuracy'])
df['Norm Train accuracy'] = normalize(df['Train accuracy'])
df['Norm General accuracy'] = normalize(df['General accuracy'])
df['Norm Time rfe'] = normalize(df['Time rfe'])
df['Norm Time train'] = normalize(df['Time train'])
df['Norm Time total'] = normalize(df['Time total'])

# 设定权重（假设各指标同等重要，可以根据实际需要调整权重）
weights = {
    'Norm Best cross': 1,
    'Norm Validation accuracy': 0.1,
    'Norm Test accuracy': 0.1,
    'Norm Train accuracy': 0.8,
    'Norm General accuracy': 1,
    'Norm Time rfe': -0.5,  # 时间指标的权重为负，因为时间越少越好
    'Norm Time train': -0.5,
    'Norm Time total': 0
}

# 计算综合性能评分
df['Performance Index'] = (
    weights['Norm Best cross'] * df['Norm Best cross'] +
    weights['Norm Validation accuracy'] * df['Norm Validation accuracy'] +
    weights['Norm Test accuracy'] * df['Norm Test accuracy'] +
    weights['Norm General accuracy'] * df['Norm General accuracy'] +
    weights['Norm Time rfe'] * df['Norm Time rfe'] +
    weights['Norm Time train'] * df['Norm Time train'] +
    weights['Norm Time total'] * df['Norm Time total']
)
# 查看结果
df[['N_feature', 'Performance Index']].sort_values(by='Performance Index', ascending=False)
df.to_csv('result_PI.csv', index=False)