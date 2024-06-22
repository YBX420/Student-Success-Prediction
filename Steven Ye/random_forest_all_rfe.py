
# Import necessary libraries
import pandas as pd
import numpy as np
import time
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, accuracy_score

# Start timing
start_time = time.time()

# Load all the data
file_path = '../student_data_classified.csv'
data = pd.read_csv(file_path, delimiter=',')


print("Available columns:", data.columns.tolist())

data = data.drop(columns=['GDP', 'Unemployment rate', 'Output','GDP_random'])
X = data.drop(['GDP_class'], axis=1)
y = data['GDP_class']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

param_grid = {
    'rf__n_estimators': [100, 200, 300],  # Number of trees in the forest
    'rf__max_depth': [None, 10, 20,30]  # Maximum depth of the trees
}
base_estimator = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=42)
best_params_list = []
scores_list = []
times_list = []
start_time_total = time.time()

for n_features in range(35, 1, -5):
    rfe = RFE(estimator=base_estimator, n_features_to_select=n_features)
    X_train_reduced = rfe.fit_transform(X_train, y_train)

    pipeline = Pipeline([
        ('rf', RandomForestClassifier(random_state=42))
    ])

    grid_search = GridSearchCV(pipeline, param_grid, cv=10, scoring='accuracy', verbose=3, n_jobs=-1)

    start_time = time.time()

    grid_search.fit(X_train, y_train)

    end_time = time.time()

    best_params_list.append((n_features, grid_search.best_params_))
    times_list.append((n_features, end_time - start_time))
    scores_list.append((n_features, grid_search.best_score_))


end_time_total = time.time()

# 打印每次RFE的最佳参数和所用时间
for n_features, best_params in best_params_list:
    print(f"Number of Features: {n_features}, Best Parameters: {best_params}")

for n_features, time_used in times_list:
    print(f"Number of Features: {n_features}, Time Used (seconds): {time_used}")

for n_features, score in scores_list:
    print(f"Number of Features: {n_features}, Scores Get: {score}")

with open('best_params.txt', 'w') as f:
    for n_features, best_params in best_params_list:
        f.write(f"{best_params}\n")

with open('times.txt', 'w') as f:
    for n_features, time_used in times_list:
        f.write(f"{time_used}\n")

with open('scores.txt', 'w') as f:
    for n_features, score in scores_list:
        f.write(f"{score}\n")

# 打印总时间
print("Total Time Used (seconds):", end_time_total - start_time_total)


'''
params_best = grid_search.best_params_
best_model = pipeline.set_params(**params_best)
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
'''

