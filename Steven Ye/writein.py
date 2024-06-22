data_to_write = [
    "Best parameters found: {'n_estimators': 100, 'max_depth': 20}",
    "Best cross-validation score: 0.95",
    "TRAIN",
    "Accuracy: 0.98",
    "Precision: 0.97",
    "Recall: 0.96",
    "VALIDATION",
    "Accuracy: 0.94",
    "Precision: 0.93",
    "Recall: 0.92",
    "TEST",
    "Accuracy: 0.93",
    "Precision: 0.92",
    "Recall: 0.91",
    "Total runtime of the script: 300 seconds"
]

# 打开文件，追加模式
with open('results.txt', 'a') as f:
    for line in data_to_write:
        f.write(line + '\n')  # 每行写入文件，并换行

print("Results have been written to results.txt")
