# Import necessary libraries
import pandas as pd
import numpy as np
import time
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Start timing
start_time = time.time()

file_path = 'student_data_classified.csv'
data = pd.read_csv(file_path, delimiter=',')

# 确认数据列名
print("Available columns:", data.columns.tolist())

# 假设你的数据集中需要预测的目标列是 'GDP_Class'
# 丢弃 'GDP_Class' 列作为特征集，'GDP_Class' 作为目标变量
data = data.drop(columns=['GDP', 'Unemployment rate', 'Output'])
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


# Initialize the base estimator for RFE with specified parameters
base_estimator = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=42)

# Perform Recursive Feature Elimination (RFE) to reduce to 100 features
selector = RFE(estimator=base_estimator, n_features_to_select=15, step=10, verbose=3)
X_train_reduced = selector.fit_transform(X_train, y_train)

# Define a pipeline with only Random Forest classifier (as dimensionality reduction is done by RFE)
pipeline = Pipeline([
    ('rf', RandomForestClassifier(random_state=42))  # Random Forest classifier
])

# Define the parameter grid for the Random Forest classifier
param_grid = {
    'rf__n_estimators': [100, 200, 500, 1000,2000],  # Number of trees in the forest
    'rf__max_depth': [None, 10, 30, 50, 100, 200]  # Maximum depth of the trees
}

# Setup GridSearchCV with the pipeline
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring='accuracy', verbose=3, n_jobs=-1)

# Fit the model on the training data with reduced features
grid_search.fit(X_train_reduced, y_train)

# Print the best parameters and the best score
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation score:", grid_search.best_score_)

# End timing
end_time = time.time()

# Calculate total runtime
total_time = end_time - start_time

# Print total runtime
print(f"Total runtime of the script: {total_time} seconds")
