import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import sklearn.tree as tree
import graphviz
from tqdm import tqdm

# 读取数据
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

# 使用GridSearchCV寻找最优参数
param_grid = {
    'n_estimators': [100, 200, 500, 1000],  # 树的数量
    'max_depth': [None, 10, 30, 50, 100],  # 树的最大深度
    'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],  # 最小不纯度减少
    'min_samples_split': [2, 5, 10, 20]  # 最小样本分裂数
}


rfc = RandomForestClassifier(random_state=0)
grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=3)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best parameters found: ", grid_search.best_params_)


# 使用最佳参数训练模型
print("Best estimators found: ", grid_search.best_estimator_)

best_model = RandomForestClassifier(**grid_search.best_params_)
best_model.fit(X_train, y_train)

# 评估模型
ypred_train = best_model.predict(X_train)
ypred_val = best_model.predict(X_val)
ypred_test = best_model.predict(X_test)

acc_train = accuracy_score(y_train, ypred_train)
prec_train = precision_score(y_train, ypred_train, average='weighted')
rec_train = recall_score(y_train, ypred_train, average='weighted')

acc_val = accuracy_score(y_val, ypred_val)
prec_val = precision_score(y_val, ypred_val, average='weighted')
rec_val = recall_score(y_val, ypred_val, average='weighted')

acc_test = accuracy_score(y_test, ypred_test)
prec_test = precision_score(y_test, ypred_test, average='weighted')
rec_test = recall_score(y_test, ypred_test, average='weighted')

print('TRAIN')
print('Accuracy:', acc_train)
print('Precision:', prec_train)
print('Recall:', rec_train)

print('VALIDATION')
print('Accuracy:', acc_val)
print('Precision:', prec_val)
print('Recall:', rec_val)

print('TEST')
print('Accuracy:', acc_test)
print('Precision:', prec_test)
print('Recall:', rec_test)

# 可视化随机森林
# estimator = best_model.estimators_[0]
# dot_data = tree.export_graphviz(estimator, out_file=None,
#                                 feature_names=X.columns,
#                                 class_names=label_encoder.classes_,
#                                 filled=True, rounded=True,
#                                 special_characters=True)
# graph = graphviz.Source(dot_data)
# graph.format = 'png'
# graph.render('decision_tree')

# 绘制不同参数下模型的准确度图
def plot_accuracy(param_range, train_scores, val_scores, xlabel, ylabel):
    plt.figure()
    plt.plot(param_range, train_scores, 'b', label='Training Data')
    plt.plot(param_range, val_scores, 'r', label='Validation Data')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# 示例：max_depth
depths = range(1, 30)
train_scores = []
val_scores = []

for d in tqdm(depths, desc="Training models with different max_depth"):
    rfc = RandomForestClassifier(max_depth=d, random_state=0)
    rfc.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, rfc.predict(X_train)))
    val_scores.append(accuracy_score(y_val, rfc.predict(X_val)))

plot_accuracy(depths, train_scores, val_scores, 'Max Depth', 'Accuracy')
