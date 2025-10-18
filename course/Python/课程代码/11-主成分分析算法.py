#  主成分分析算法
#11.2.2  载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#11.2.3  变量设置及数据处理
data=pd.read_csv('数据11.1.csv')
X = data.iloc[:,1:11]#设置分析所需要的特征变量
X.info()
len(X.columns) 
X.columns 
X.shape
X.dtypes
X.isnull().values.any() 
X.isnull().sum() 
X.head(10)
scaler = StandardScaler()
scaler.fit(X)
X_s = scaler.transform(X)
X_s = pd.DataFrame(X_s, columns=X.columns)
#11.2.4  特征变量相关性分析
print(X_s.corr(method='pearson'))  
plt.subplot(1,1,1)
sns.heatmap(X_s.corr(), annot=True)

#11.3  主成分分析算法示例
#11.3.1  主成分提取及特征值、方差贡献率计算
model = PCA()#将模型设置为主成分分析算法
model.fit(X_s)#基于X_s的数据，使用fit方法进行拟合
np.set_printoptions(suppress=True)#不以科学计数法显示，而是直接显示数字
model.explained_variance_#计算提取的各个主成分的特征值
model.explained_variance_ratio_#计算各主成分的方差贡献率

#11.3.2  绘制碎石图观察各主成分特征值
plt.plot(model.explained_variance_, 'o-')#绘制碎石图观察各主成分特征值
plt.axhline(model.explained_variance_[2], color='r', linestyle='--', linewidth=2)
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plt.xlabel('主成分')#将图中x轴的标签设置为'主成分'
plt.ylabel('特征值')#将图中y轴的标签设置为'特征值'
plt.title('各主成分特征值碎石图')#将图的标题设置为'各主成分特征值碎石图'
#11.3.3  绘制碎石图观察各主成分方差贡献率
plt.plot(model.explained_variance_ratio_, 'o-')
plt.axhline(model.explained_variance_ratio_[2], color='r', linestyle='--', linewidth=2)
plt.xlabel('主成分')#将图中x轴的标签设置为'主成分'
plt.ylabel('方差贡献率')#将图中y轴的标签设置为'方差贡献率'
plt.title('各主成分方差贡献率碎石图')#将图的标题设置为'各主成分方差贡献率碎石图'
#11.3.4  绘制碎石图观察主成分累积方差贡献率
plt.plot(model.explained_variance_ratio_.cumsum(), 'o-')
plt.xlabel('主成分')#将图中x轴的标签设置为'主成分'
plt.ylabel('累积方差贡献率')#将图中y轴的标签设置为'累积方差贡献率'
plt.axhline(0.85, color='r', linestyle='--', linewidth=2)
plt.title('主成分累积方差贡献率碎石图')#将图的标题设置为'主成分累积方差贡献率碎石图'

#11.3.5  计算样本示例的主成分得分
scores = pd.DataFrame(model.transform(X_s), columns=['Comp' + str(n) for n in range(1, 11)])
pd.set_option('display.max_columns', None)#显示完整的列，如果不运行该代码，中间有省略号。
scores.head(10)
#11.3.6  绘制二维图形展示样本示例在前两个主成分上的得分

V1=pd.Series(data.V1)
plt.figure(figsize=(9,9))
plt.rcParams['axes.unicode_minus']=False# 解决图表中负号不显示问题。
sns.scatterplot(x='Comp1', y='Comp2',hue=V1, style=V1,data=scores)
plt.title('样本示例的主成分得分')

#11.3.7  绘制三维图形展示样本示例在前三个主成分上的得分
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['axes.unicode_minus']=False# 解决图表中负号不显示问题。
ax.scatter(scores['Comp1'], scores['Comp2'], scores['Comp3'],c='r')
ax.set_xlabel('Comp1')
ax.set_ylabel('Comp2')
ax.set_zlabel('Comp3')

#11.3.8  输出特征向量矩阵，观察主成分载荷
model.components_#计算主成分载荷，观察每个变量对于主成分的影响
round(pd.DataFrame(model.components_.T, index=X_s.columns, columns=['Comp' + str(n) for n in range(1, 11)]), 2)
