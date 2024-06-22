import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt


# 加载示例数据集
file_path = '../student_data_classified.csv'
data = pd.read_csv(file_path, delimiter=',')

# 确认数据列名
print("Available columns:", data.columns.tolist())

# drop the three orignal data and remains the selected ones.
data = data.drop(columns=['GDP', 'Unemployment rate', 'Output','GDP_random'])
X = data.drop(['GDP_class'], axis=1)
y = data['GDP_class']

# 将目标变量转换为数值标签
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 分割数据集
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=0)

# 标准化数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 创建并训练随机森林模型
best_para = {'max_depth' : 30, 'n_estimators' : 1000, 'random_state':0}

best_model = RandomForestClassifier(**best_para)
best_model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = best_model.predict(X_test)

# 生成混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 使用seaborn绘制混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()