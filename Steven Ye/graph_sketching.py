import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'different_rfe_result.csv'
df = pd.read_csv(file_path)

# 检查数据
print(df.head())

def n_feature_time_rfe():
    # draw time_rfe by n_feature 
    plt.figure(figsize=(10, 6))
    plt.plot(df['N_feature'], df['Total rfe-time'], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('RFE Time (seconds)')
    plt.title('RFE Time vs. Number of Features')
    plt.grid(True)
    plt.show()

def n_feature_time_train():
    # 绘制time_rfe和n_feature的关系图
    plt.figure(figsize=(10, 6))
    plt.plot(df['N_feature'], df['Total train-time'], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Train Time (seconds)')
    plt.title('Train Time vs. Number of Features')
    plt.grid(True)
    plt.show()

def n_feature_total_train():
    # 绘制time_rfe和n_feature的关系图
    plt.figure(figsize=(10, 6))
    plt.plot(df['N_feature'], df['Total runtime'], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Total Time (seconds)')
    plt.title('Total Time vs. Number of Features')
    plt.grid(True)
    plt.show()

def n_feature_all_time():
    plt.figure(figsize=(12, 8))
    plt.plot(df['N_feature'], df['Total rfe-time'], marker='o', label='RFE Time')
    plt.plot(df['N_feature'], df['Total train-time'], marker='x', label='Train Time')
    plt.plot(df['N_feature'], df['Total runtime'], marker='s', label='Total Time')

    plt.xlabel('Number of Features')
    plt.ylabel('Time (seconds)')
    plt.title('RFE, Train, and Total Time vs. Number of Features')
    plt.legend()
    plt.grid(True)
    plt.show()


def n_feature_accuracy():
    # 绘制time_rfe和n_feature的关系图
    plt.figure(figsize=(10, 6))
    plt.plot(df['N_feature'], df['TEST Accuracy'], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Test accuracy')
    plt.title('Test accuracy vs. Number of Features')
    plt.grid(True)
    plt.show()

def accuracy_time():
    # 绘制time_rfe和n_feature的关系图
    plt.figure(figsize=(10, 6))
    plt.plot(df['Total runtime'], df['TEST Accuracy'], marker='o')
    plt.ylabel('Test Accuracy')
    plt.xlabel('Total runtime')
    plt.title('Total Runtime vs. Test Accuracy')
    plt.grid(True)
    plt.show()

def accuracy_time_total():
    print(df.head())
    n_feature = df['N_feature']
    total_run_time = df['Total runtime']
    test_accuracy = df['TEST Accuracy']

    # 创建一个图形对象
    fig, ax1 = plt.subplots()

    # 绘制Total run time和n_feature的关系
    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('Total Run Time (seconds)', color='tab:blue')
    ax1.plot(n_feature, total_run_time, color='tab:blue', marker='o', label='Total Run Time')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 创建第二个y轴，共享x轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy', color='tab:red')
    ax2.plot(n_feature, test_accuracy, color='tab:red', marker='x', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 添加图例
    fig.tight_layout()  # 调整布局以防止标签重叠
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 保存图像
    plt.show()


# n_feature_time_rfe()
# n_feature_time_train()
n_feature_total_train()
n_feature_accuracy()
#accuracy_time_total()