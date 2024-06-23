import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

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

# As the last column is the target variable and the rest are features
X = students.iloc[:, :-1]
y = students.iloc[:, -1]

# Check if the target variable is continuous
if y.dtype in ['float64', 'float32', 'int64', 'int32'] and len(set(y)) > 20:
    # Convert continuous target to discrete values
    y = pd.qcut(y, q=4, labels=False)  # Example: Convert to 4 discrete classes


# Encode target labels with value between 0 and n_classes-1
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the parameter grid for tuning
# Different values of each hyperparameter to check
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly']
}

# Initialize the SVM classifier
svm = SVC(random_state=42)

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
print("Best parameters found: ", grid_search.best_params_)

# Use the best estimator to make predictions
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
