import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# 读取CSV文件
file_path = 'student_data_classified.csv'  # 请确保文件路径正确
data = pd.read_csv(file_path)

# 确认数据的前几行
print(data.head())

# 确认列名
print("Available columns:", data.columns.tolist())

# 准备特征数据和目标变量
data = data.drop(columns=['Unemployment rate'])
data = data.drop(columns=['Output'])
data = data.drop(columns=['GDP'])
features = data
target = data['GDP_class']

# 将目标变量转换为数值标签
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# 标准化特征数据
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 创建UMAP模型
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

# 拟合模型并将数据转换为2D
data_2d = reducer.fit_transform(features_scaled)

# 绘制2D数据，根据编码后的y值着色
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=target_encoded, cmap='nipy_spectral', s=20)

# 创建带有原始类别名称的图例
legend1 = plt.legend(*scatter.legend_elements(), title="GDP Classes")
plt.gca().add_artist(legend1)

# 将编码标签映射回原始标签以创建图例
legend_labels = label_encoder.inverse_transform(np.unique(target_encoded))
handles, _ = scatter.legend_elements()
legend2 = plt.legend(handles, legend_labels, title="GDP Classes")
plt.gca().add_artist(legend2)

plt.title('UMAP projection of the dataset')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.show()
