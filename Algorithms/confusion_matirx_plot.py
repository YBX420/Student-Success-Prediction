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


# add the dataset
file_path = '../student_data_classified.csv'
data = pd.read_csv(file_path, delimiter=',')

# affirm the columns
print("Available columns:", data.columns.tolist())

# drop the three orignal data and remains the selected ones.
data = data.drop(columns=['GDP', 'Unemployment rate', 'Output','GDP_random'])
X = data.drop(['GDP_class'], axis=1)
y = data['GDP_class']

# encode all the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# slipt the dataset
X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)
X_val, X_test, y_val, y_test = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=0)

# standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 创建并训练随机森林模型
best_para = {'max_depth': 30, 'min_impurity_decrease': 0.0, 'min_samples_split': 2, 'n_estimators': 1000}

best_model = RandomForestClassifier(**best_para)
best_model.fit(X_train, y_train)

# predict it on the model
y_pred = best_model.predict(X_test)

# change the cm
cm = confusion_matrix(y_test, y_pred)
print(cm)
n = 10

# use the seaborn to plot the graph
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
output_directory = f'confusion_matrix_{n}.png'
plt.savefig(output_directory)