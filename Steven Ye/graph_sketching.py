import pandas as pd
import matplotlib.pyplot as plt

# read the csv file
file_path = 'diff_result_pi.csv'
df = pd.read_csv(file_path)

# check the data
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
    # plot the time_rfe and the n_feature
    plt.figure(figsize=(10, 6))
    plt.plot(df['N_feature'], df['Total train-time'], marker='o')
    plt.xlabel('Number of Features')
    plt.ylabel('Train Time (seconds)')
    plt.title('Train Time vs. Number of Features')
    plt.grid(True)
    plt.show()

# def n_feature_total_train():
#     # 绘制time_rfe和n_feature的关系图
#     plt.figure(figsize=(10, 6))
#     plt.plot(df['N_feature'], df['Total runtime'], marker='o')
#     plt.xlabel('Number of Features')
#     plt.ylabel('Total Time (seconds)')
#     plt.title('Total Time vs. Number of Features')
#     plt.grid(True)
#     plt.show()

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

def n_feature_total_train():
    n_feature = df['N_feature']
    runtime = df['Total runtime']

    # create the figure
    plt.figure(figsize=(10, 6))

    # plot the features
    plt.plot(n_feature, runtime, marker='o', label='Total runtime')

    # label the maximum and minimum
    max_accuracy_index = runtime.idxmax()
    min_accuracy_index = runtime.idxmin()

    plt.plot(n_feature[max_accuracy_index], runtime[max_accuracy_index], 'rx', markersize = 24)  # 红色X标记最大值
    plt.plot(n_feature[min_accuracy_index], runtime[min_accuracy_index], 'gx', markersize = 24)  # 绿色X标记最小值

    # add the label
    plt.xlabel('Number of Features')
    plt.ylabel('Run Time')
    plt.title('Run Time vs. Number of Features')
    plt.grid(True)
    plt.show()

def n_feature_accuracy():
    n_feature = df['N_feature']
    test_accuracy = df['TEST Accuracy']

    # add the figures
    plt.figure(figsize=(10, 6))

    # plot the graph
    plt.plot(n_feature, test_accuracy, marker='o', label='Test Accuracy')

    # label the max and min
    max_accuracy_index = test_accuracy.idxmax()
    min_accuracy_index = test_accuracy.idxmin()

    plt.plot(n_feature[max_accuracy_index], test_accuracy[max_accuracy_index], 'rx', markersize = 24)  # 红色X标记最大值
    plt.plot(n_feature[min_accuracy_index], test_accuracy[min_accuracy_index], 'gx', markersize = 24)  # 绿色X标记最小值

    # add the labels
    plt.xlabel('Number of Features')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs. Number of Features')
    plt.grid(True)
    plt.show()

def accuracy_time():
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

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Number of Features')
    ax1.set_ylabel('Total Run Time (seconds)', color='tab:blue')
    ax1.plot(n_feature, total_run_time, color='tab:blue', marker='o', label='Total Run Time')
    ax1.tick_params(axis='y', labelcolor='tab:blue')


    ax2 = ax1.twinx()
    ax2.set_ylabel('Test Accuracy', color='tab:red')
    ax2.plot(n_feature, test_accuracy, color='tab:red', marker='x', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:red')


    fig.tight_layout()  
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')


    plt.show()

def performance_index():
    n_feature = df['N_feature']
    performance_index = df['Performance Index']

    # add the figures
    plt.figure(figsize=(10, 6))

    # plot the graph
    plt.plot(n_feature, performance_index, marker='o', label='Test Accuracy')

    # label the max and min
    max_accuracy_index = performance_index.idxmax()
    min_accuracy_index = performance_index.idxmin()

    plt.plot(n_feature[max_accuracy_index], performance_index[max_accuracy_index], 'rx', markersize = 24)  # 红色X标记最大值
    plt.plot(n_feature[min_accuracy_index], performance_index[min_accuracy_index], 'gx', markersize = 24)  # 绿色X标记最小值

    # add the labels
    plt.xlabel('Number of Features')
    plt.ylabel('Performance Index')
    plt.title('Performance Index vs. Number of Features')
    plt.grid(True)
    plt.show()

# n_feature_time_rfe()
# n_feature_time_train()
#n_feature_all_time()
#n_feature_total_train()
#n_feature_accuracy()
performance_index()
#accuracy_time_total()