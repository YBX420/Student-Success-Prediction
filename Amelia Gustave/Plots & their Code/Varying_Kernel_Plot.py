import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import precision_score, accuracy_score, recall_score
import matplotlib.pyplot as plt
import numpy as np

# Load your dataset
students = pd.read_csv('dataset.csv')

# Define the mapping
status_mapping = {'Graduate': 1, 'Dropout': 0, 'Enrolled': 2}

# Apply the mapping to the last column
students['Target'] = students['Target'].map(status_mapping)

# Remove the enrolled students, so we only look at dropouts and graduates
students = students[students['Target'] != 2]

students.drop(['GDP', 'Inflation rate', 'Unemployment rate'], inplace=True, axis=1)

# Convert target to numeric (if necessary)
students['Target'] = pd.to_numeric(students['Target'], errors='coerce')

# Assume the last column is the target variable and the rest are features
X = students.iloc[:, :-1]
y = students.iloc[:, -1]

# Encode target labels with value between 0 and n_classes-1
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=21)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define the kernel types to evaluate
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

# Lists to store precision, accuracy, and recall scores
train_precisions = []
val_precisions = []
train_accuracies = []
val_accuracies = []
train_recalls = []
val_recalls = []

# Function to calculate metrics
def get_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    return precision, accuracy, recall

# Evaluate each kernel
for kernel in kernels:
    # Train the model
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train, y_train)
    
    # Predict and calculate metrics for training set
    y_train_pred = svm.predict(X_train)
    train_precision, train_accuracy, train_recall = get_metrics(y_train, y_train_pred)
    train_precisions.append(train_precision)
    train_accuracies.append(train_accuracy)
    train_recalls.append(train_recall)
    
    # Predict and calculate metrics for validation set
    y_val_pred = svm.predict(X_val)
    val_precision, val_accuracy, val_recall = get_metrics(y_val, y_val_pred)
    val_precisions.append(val_precision)
    val_accuracies.append(val_accuracy)
    val_recalls.append(val_recall)

# Plotting the metrics on separate pages
# Precision
plt.figure(figsize=(8, 6))
plt.plot(kernels, train_precisions, marker='o', label='Training Precision')
plt.plot(kernels, val_precisions, marker='o', label='Validation Precision')
plt.xlabel('Kernel Type')
plt.ylabel('Precision')
plt.title('Precision Scores of Training and Validation Sets with Different Kernels')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Accuracy
plt.figure(figsize=(8, 6))
plt.plot(kernels, train_accuracies, marker='o', label='Training Accuracy')
plt.plot(kernels, val_accuracies, marker='o', label='Validation Accuracy')
plt.xlabel('Kernel Type')
plt.ylabel('Accuracy')
plt.title('Accuracy Scores of Training and Validation Sets with Different Kernels')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Recall
plt.figure(figsize=(8, 6))
plt.plot(kernels, train_recalls, marker='o', label='Training Recall')
plt.plot(kernels, val_recalls, marker='o', label='Validation Recall')
plt.xlabel('Kernel Type')
plt.ylabel('Recall')
plt.title('Recall Scores of Training and Validation Sets with Different Kernels')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
