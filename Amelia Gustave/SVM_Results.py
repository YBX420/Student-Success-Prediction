import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
students = pd.read_csv('dataset.csv')

# Define the mapping
status_mapping = {'Graduate': 1, 'Dropout': 0, 'Enrolled': 2}

# Apply the mapping to the last column
students.iloc[:, -1] = students.iloc[:, -1].map(status_mapping)

# Remove the enrolled students, so we only look at dropouts and graduates
students = students[students['Target'] != 2]

students.drop('GDP', inplace = True, axis = 1)
students.drop('Inflation rate', inplace = True, axis = 1)
students.drop('Unemployment rate', inplace = True, axis = 1)

students['Target'] = pd.to_numeric(students['Target'], errors='coerce')

# Assume the last column is the target variable and the rest are features
X = students.iloc[:, :-1]
y = students.iloc[:, -1]

# Check if the target variable is continuous
if y.dtype in ['float64', 'float32', 'int64', 'int32'] and len(set(y)) > 20:
    # Convert continuous target to discrete values
    y = pd.qcut(y, q=4, labels=False)  # Example: Convert to 4 discrete classes


# Encode target labels with value between 0 and n_classes-1
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=21)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)


# Create an SVM classifier (Hypertuning - adjust parameters as needed)
clf = svm.SVC(kernel='rbf', gamma = 0.001, C = 4)

# Fit the model on training data
clf.fit(X_train, y_train)

# Predictions on training set
y_train_pred = clf.predict(X_train)
# Predictions on testing set
y_test_pred = clf.predict(X_test)
# Predictions on validation set
y_val_pred = clf.predict(X_val)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
train_precision = precision_score(y_train, y_train_pred)
train_recall = recall_score(y_train, y_train_pred)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)

# Print results
print("Training set:")
print(f"Accuracy: {train_accuracy:.2f}, Precision: {train_precision:.2f}, Recall: {train_recall:.2f}")
print("Testing set:")
print(f"Accuracy: {test_accuracy:.2f}, Precision: {test_precision:.2f}, Recall: {test_recall:.2f}")
print("Validation set:")
print(f"Accuracy: {val_accuracy:.2f}, Precision: {val_precision:.2f}, Recall: {val_recall:.2f}")

from sklearn.metrics import confusion_matrix, classification_report

# Calculate confusion matrix and classification report for training set
train_cm = confusion_matrix(y_train, y_train_pred)
train_cr = classification_report(y_train, y_train_pred)

# Calculate confusion matrix and classification report for testing set
test_cm = confusion_matrix(y_test, y_test_pred)
test_cr = classification_report(y_test, y_test_pred)

# Calculate confusion matrix and classification report for validation set
val_cm = confusion_matrix(y_val, y_val_pred)
val_cr = classification_report(y_val, y_val_pred)


'''
# Print confusion matrix and classification report
print("Training set:")
print("Confusion Matrix:")
print(train_cm)
print("Classification Report:")
print(train_cr)

print("Testing set:")
print("Confusion Matrix:")
print(test_cm)
print("Classification Report:")
print(test_cr)

print("Validation set:")
print("Confusion Matrix:")
print(val_cm)
print("Classification Report:")
print(val_cr)
'''
