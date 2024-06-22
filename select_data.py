import pandas as pd
import random

file_path = 'student_data.csv'


data = pd.read_csv(file_path)

# 选择GDP列并查看其描述性统计信息
gdp_column = 'GDP'
unemployment = 'Unemployment rate'
print(data[gdp_column].describe())
print(data[unemployment].describe())

# 计算分位数
q1 = data[gdp_column].quantile(0.33)
q2 = data[gdp_column].quantile(0.66)
q1_u = data[unemployment].quantile(0.33)
q2_u = data[unemployment].quantile(0.66)

print(f"33rd percentile (Q1): {q1}")
print(f"66th percentile (Q2): {q2}")

# 定义分类函数
def classify_gdp(gdp):
    if gdp < q1:
        return '-1'
    elif gdp < q2:
        return '0'
    else:
        return '1'
    
def classify_un(un):
    if un < q1_u:
        return '-1'
    elif un < q2_u:
        return '0'
    else:
        return '1'

def classify_drop(ot):
    if ot == 'Dropout':
        return '1'
    else:
        return '0'
    
def classify_gra(ot):
    if ot == 'Graduate':
        return '1'
    else:
        return '0'

def classify_enroll(ot):
    if ot == 'Enrolled':
        return '1'
    else:
        return '0' 

def classify_random(ran):
    return random.randint(-1,1)

# 应用分类函数

data['Unemployment rate_class'] = data[unemployment].apply(classify_un)
data['GDP_class'] = data[gdp_column].apply(classify_gdp)
data['Graduated_class'] = data['Output'].apply(classify_gra)
data['Dropout_class'] = data['Output'].apply(classify_drop)
data['Enrolled_class'] = data['Output'].apply(classify_enroll)
data['GDP_random'] = data[gdp_column].apply(classify_random)
data = data.drop(columns=['Inflation rate'])


#data = data[data['Output'] != 'Enrolled']
# 查看分类结果

# 保存到新的CSV文件
data.to_csv('student_data_classified.csv', index=False)
