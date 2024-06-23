import pandas as pd
import numpy as np
import time
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score

start_time = time.time()

file_path = '../student_data_classified.csv'
data = pd.read_csv(file_path, delimiter=',')

# 确认数据列名
print("Available columns:", data.columns.tolist())

# 假设你的数据集中需要预测的目标列是 'GDP_Class'
# 丢弃 'GDP_Class' 列作为特征集，'GDP_Class' 作为目标变量
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

# 定义XGBoost分类器


# 定义XGBoost的参数网格
param_grid = {
    'n_estimators': [100, 200, 500, 1000],  # Number of gradient boosted trees
    'max_depth': [None, 10, 30, 50, 100],  # Depth of each tree
    'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage used to prevent overfitting
    'subsample': [0.5, 0.75, 1],  # Subsample ratio of the training instances
    'colsample_bytree': [0.5, 0.75, 1],  # Subsample ratio of columns when constructing each tree
}
xgb_clf = XGBClassifier(eval_metric='mlogloss', random_state=42)
# 使用GridSearchCV进行超参数调优
grid_search = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=10, scoring='accuracy', verbose=3, n_jobs=-1)

# 在训练数据上拟合模型
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳得分
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)


#params_best = {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1000, 'subsample': 1}
best_model = XGBClassifier(**grid_search.best_params_,eval_metric='mlogloss', random_state=42)
best_model.fit(X_train,y_train)

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

# 计算并输出总运行时间
end_time = time.time()
total_time = end_time - start_time
print(f"Total runtime of the script: {total_time} seconds")
