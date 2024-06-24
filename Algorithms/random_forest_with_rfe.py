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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
def run_rfe(n):
    # Start timing
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


    # Initialize the base estimator for RFE with specified parameters
    base_estimator = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=42)

    # Perform Recursive Feature Elimination (RFE) to reduce to 15 features
    selector = RFE(estimator=base_estimator, n_features_to_select=n,step=5, verbose=3)
    X_train_reduced = selector.fit_transform(X_train, y_train)

    rfe_time = time.time()
    # Define a pipeline with only Random Forest classifier (as dimensionality reduction is done by RFE)
    pipeline = Pipeline([
        ('rf', RandomForestClassifier(random_state=42))  # Random Forest classifier
    ])

    # Define the parameter grid for the Random Forest classifier

    param_grid = {
        'rf__n_estimators': [100, 200, 500, 1000],  # 树的数量
        'rf__max_depth': [None, 10, 30, 50, 100],  # 树的最大深度
        'rf__min_impurity_decrease': [0.0, 0.01, 0.05, 0.1],  # 最小不纯度减少
        'rf__min_samples_split': [2, 5, 10, 20]  # 最小样本分裂数
    }

    # Setup GridSearchCV with the pipeline
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=10, scoring='accuracy', verbose=3, n_jobs=-1)

    # Fit the model on the training data with reduced features
    grid_search.fit(X_train_reduced, y_train)



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

    cm = confusion_matrix(y_test, ypred_test)
    print(cm)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'confusion_matrix_second_{n}.png')

    # Print the best parameters and the best score
    end_time = time.time()

    # Calculate total runtime
    total_time = end_time - start_time
    rfe_cal_time = rfe_time- start_time
    train_time = end_time - rfe_time

    output_file_path = 'different_rfe_result.csv'
    df = pd.read_csv(output_file_path)
    new_row = pd.DataFrame({
        'N_feature': [n],
        'Best cross-validation' : [grid_search.best_score_],
        'TRAIN Accuracy': [acc_train],
        'TRAIN Precision': [prec_train],
        'TRAIN Recall': [rec_train],
        'VALIDATION Accuracy': [acc_val],
        'VALIDATION Precision': [prec_val],
        'VALIDATION Recall': [rec_val],
        'TEST Accuracy': [acc_test],
        'TEST Precision': [prec_test],
        'TEST Recall': [rec_test],
        'Total runtime' : [total_time],
        'Total rfe-time' : [rfe_cal_time],
        'Total train-time': [train_time]
    })
    df = pd.concat([df,new_row], ignore_index=True)
    df.to_csv(output_file_path, index=False)

    # with open('results.txt','a') as f:
    #     print(f"The n_feature testing is {n}")
    #     print("Best parameters found:", grid_search.best_params_)
    #     print("Best cross-validation score:", grid_search.best_score_)

    #     print('TRAIN')
    #     print('Accuracy:', acc_train) 
    #     print('Precision:', prec_train)
    #     print('Recall:', rec_train)

    #     print('VALIDATION')
    #     print('Accuracy:', acc_val)
    #     print('Precision:', prec_val)
    #     print('Recall:', rec_val)

    #     print('TEST')
    #     print('Accuracy:', acc_test)
    #     print('Precision:', prec_test)
    #     print('Recall:', rec_test)

    # # End timing

    # # Print total runtime
    # with open('results.txt','a') as f:
    #     print(f"Total runtime of the script: {total_time} seconds")
    #     print(f"Total runtime of the rfe: {rfe_cal_time} seconds")
    #     print(f"Total runtime of the train: {train_time} seconds")

