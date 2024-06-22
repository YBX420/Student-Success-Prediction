import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'result_rfe.csv'
df = pd.read_csv(file_path)

# 检查数据
print(df.head())

def n_feature_time_rfe():
    # draw time_rfe by n_feature 
    plt.figure(figsize=(10, 6))
    plt.plot(df['N_feature'], df['Time rfe'], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('RFE Time (seconds)')
    plt.title('RFE Time vs. Number of Features')
    plt.grid(True)
    plt.show()

def n_feature_time_train():
    # 绘制time_rfe和n_feature的关系图
    plt.figure(figsize=(10, 6))
    plt.plot(df['N_feature'], df['Time train'], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Train Time (seconds)')
    plt.title('Train Time vs. Number of Features')
    plt.grid(True)
    plt.show()

def n_feature_total_train():
    # 绘制time_rfe和n_feature的关系图
    plt.figure(figsize=(10, 6))
    plt.plot(df['N_feature'], df['Time total'], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Total Time (seconds)')
    plt.title('Total Time vs. Number of Features')
    plt.grid(True)
    plt.show()

def n_feature_all_time():
    plt.figure(figsize=(12, 8))
    plt.plot(df['N_feature'], df['Time rfe'], marker='o', label='RFE Time')
    plt.plot(df['N_feature'], df['Time train'], marker='x', label='Train Time')
    plt.plot(df['N_feature'], df['Time total'], marker='s', label='Total Time')

    plt.xlabel('Number of Features')
    plt.ylabel('Time (seconds)')
    plt.title('RFE, Train, and Total Time vs. Number of Features')
    plt.legend()
    plt.grid(True)
    plt.show()

def n_feature_all_accuracy():
    plt.figure(figsize=(12, 8))
    plt.plot(df['N_feature'], df['Train accuracy'], marker='o', label='Train accuracy')
    plt.plot(df['N_feature'], df['Validation accuracy'], marker='x', label='Validation accuracy')
    plt.plot(df['N_feature'], df['Test accuracy'], marker='s', label='Test accuracy')
    plt.plot(df['N_feature'], df['General accuracy'], marker='+', label='General accuracy')

    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    plt.title('Train, Validation, Test, and General Accuracy vs. Number of Features')
    plt.legend()
    plt.grid(True)
    plt.show()


def n_feature_accuracy():
    # 绘制time_rfe和n_feature的关系图
    plt.figure(figsize=(10, 6))
    plt.plot(df['N_feature'], df['General accuracy'], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('General accuracy')
    plt.title('General accuracy vs. Number of Features')
    plt.grid(True)
    plt.show()

def n_feature_PI():
    # draw time_rfe by n_feature 
    file_path = 'result_PI.csv'
    df = pd.read_csv(file_path)
    plt.figure(figsize=(10, 6))
    plt.plot(df['N_feature'], df['Performance Index'], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Performance Index')
    plt.title('Performance Index vs. Number of Features')
    plt.grid(True)
    plt.show()

n_feature_PI()