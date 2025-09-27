#  多元Logistic回归算法
#2.1  载入分析所需要的库和模块
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
#2.2  数据读取及观察
data=pd.read_csv('C:/Users/Administrator/.spyder-py3/数据6.1.csv')
data.info()
len(data.columns) 
data.columns 
data.shape
data.dtypes
data.isnull().values.any() 
data.isnull().sum() 
data.head()
data.V1.value_counts()

#3  描述性分析及图形绘制
#3.1  描述性分析
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data.describe()
#3.2  绘制条形图
sns.countplot(x=data['V1'])#绘制data['V1']的计数条线图
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题。
plt.title("收入档次条线图")#设置直方图标题为"收入档次直方图"
#3.3  绘制箱线图
sns.boxplot(x='V1', y='V2', data=data, palette="Blues")

sns.boxplot(x='V1', y='V3', data=data, palette="Blues")

sns.boxplot(x='V1', y='V4', data=data, palette="Blues")

#4数据处理
# 4.1  区分分类特征和连续特征并进行处理
def data_encoding(data):
    data = data[["V1",'V2',"V3","V4"]]
    Discretefeature=[]
    Continuousfeature=['V2',"V3","V4"]
    df = pd.get_dummies(data,columns=Discretefeature)
    df[Continuousfeature]=(df[Continuousfeature]-df[Continuousfeature].mean())/(df[Continuousfeature].std())
    df["V1"]=data[["V1"]]
    return df
data=data_encoding(data)
#4.2  将样本示例全集分割为训练样本和测试样本
X = data.drop(['V1'],axis=1)#设置特征变量，即除V1之外的全部变量
y = data['V1']#设置响应变量，即V1
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, stratify=y, random_state=123)

#5  建立多元Logistic回归算法模型
#5.1  模型估计
model=LogisticRegression(multi_class='multinomial',solver = 'newton-cg', C=1e10, max_iter=1e3)
model.fit(X_train, y_train)

model.n_iter_#显示模型的迭代次数
model.intercept_#显示模型的截距项
model.coef_#显示模型的回归系数

#6.5.2  模型性能分析
model.score(X_test, y_test)#计算模型预测的准确率
prob = model.predict_proba(X_test)#计算响应变量预测分类概率
prob[:5]#显示前5个样本示例的响应变量预测分类概率。

np.set_printoptions(suppress=True)
prob = model.predict_proba(X_test)#计算响应变量预测分类概率
prob[:5]#显示前5个样本示例的响应变量预测分类概率

pred = model.predict(X_test)#计算响应变量预测分类类型
pred[:5]#显示前5个样本示例的响应变量预测分类类型

table = confusion_matrix(y_test, pred)#基于测试样本输出混淆矩阵
table#显示混淆矩阵

sns.heatmap(table,cmap='Reds', annot=True)
plt.tight_layout()#输出混淆矩阵热图,cmap='r'表示使用红色系， annot=True表示在格子上显示数字

print(classification_report(y_test, pred))#输出详细的预测效果指标

cohen_kappa_score(y_test, pred)#计算kappa得分

