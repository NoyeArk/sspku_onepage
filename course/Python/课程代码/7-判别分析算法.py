#判别分析算法
#2.1  载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
#2.2  线性判别分析降维优势展示
#绘制三维数据的分布图
X, y = make_classification(n_samples=500, n_features=3, n_redundant=0,
                           n_classes=3, n_informative=2, n_clusters_per_class=1,
                           class_sep=0.5, random_state=100)#生成三类三维特征的数据
plt.rcParams['axes.unicode_minus']=False# 解决负号不显示问题
fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=20, azim=20)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', c=y)
#使用PCA进行降维
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_)
X_new = pca.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
plt.show()
#使用LDA进行降维
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)
X_new = lda.transform(X)
plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)
plt.show()#降维后样本特征信息之间的关系得以保留
#2.3  数据读取及观察
data=pd.read_csv('数据7.1.csv')
data.info()
len(data.columns) 
data.columns 
data.shape
data.dtypes
data.isnull().values.any() 
data.isnull().sum() 
data.head()
data.V1.value_counts()

#3  特征变量相关性分析
X = data.drop(['V1'],axis=1)#设置特征变量，即除V1之外的全部变量
y = data['V1']#设置响应变量，即V1
X.corr()
sns.heatmap(X.corr(), cmap='Blues', annot=True)

#4  使用样本示例全集开展线性判别分析
#4.1  模型估计及性能分析
# 使用样本示例全集开展LDA
model = LinearDiscriminantAnalysis()#使用LDA算法
model.fit(X, y)#使用fit方法进行拟合
model.score(X, y)
model.priors_
model.means_
np.set_printoptions(suppress=True)#不以科学计数法显示，而是直接显示数字
model.coef_#输出模型系数
model.intercept_#输出模型截距项
model.explained_variance_ratio_#输出可解释方差比例
model.scalings_

lda_scores = model.fit(X, y).transform(X)
lda_scores.shape
lda_scores[:5, :]

LDA_scores = pd.DataFrame(lda_scores, columns=['LD1', 'LD2'])
LDA_scores['网点类型'] = data['V1']
LDA_scores.head()

d = {0: '未转型网点', 1: '一般网点', 2: '精品网点'}
LDA_scores['网点类型'] = LDA_scores['网点类型'].map(d) 
LDA_scores.head()
plt.rcParams['axes.unicode_minus']=False# 解决图表中负号不显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题。
sns.scatterplot(x='LD1', y='LD2', data=LDA_scores, hue='网点类型')

#4.2  运用两个特征变量绘制LDA决策边界图
#安装mlxtend
pip --default-timeout=123 install mlxtend#大家在运行时把前面的“#”去掉，可能时间较长，需耐心等待 
from mlxtend.plotting import plot_decision_regions#导入plot_decision_regions


X2 = X.iloc[:,0:2]#仅选取V2存款规模、V3EVA作为特征变量
model = LinearDiscriminantAnalysis()#使用LDA算法
model.fit(X2, y)#使用fit方法进行拟合
model.score(X2, y)
model.explained_variance_ratio_

plot_decision_regions(np.array(X2), np.array(y), model)
plt.xlabel('存款规模')#将x轴设置为'存款规模'
plt.ylabel('EVA')#将y轴设置为'EVA'
plt.title('LDA决策边界')#将标题设置为'LDA决策边界'


#5  使用分割样本开展线性判别分析
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, stratify=y, random_state=123)
model = LinearDiscriminantAnalysis()#使用LDA算法
model.fit(X_train, y_train)#基于训练样本使用fit方法进行拟合
model.score(X_test, y_test)#基于测试样本计算模型预测的准确率

prob = model.predict_proba(X_test)
prob[:5]

pred = model.predict(X_test)
pred[:5]

confusion_matrix(y_test, pred)#输出测试样本的混淆矩阵

print(classification_report(y_test, pred))

cohen_kappa_score(y_test, pred)

#6  使用分割样本开展二次判别分析
#6.1  模型估计
model = QuadraticDiscriminantAnalysis()#使用QDA算法
model.fit(X_train, y_train)#基于训练样本使用fit方法进行拟合
model.score(X_test, y_test)#计算模型预测的准确率

prob = model.predict_proba(X_test)
prob[:5]

pred = model.predict(X_test)
pred[:5]

confusion_matrix(y_test, pred)

print(classification_report(y_test, pred))

cohen_kappa_score(y_test, pred)#计算cohen_kappa得分

#6.2  运用两个特征变量绘制QDA决策边界图
X2 = X.iloc[:, 0:2]
model = QuadraticDiscriminantAnalysis()
model.fit(X2, y)
model.score(X2, y)

plot_decision_regions(np.array(X2), np.array(y), model)
plt.xlabel('存款规模')#将x轴设置为'存款规模'
plt.ylabel('EVA')#将y轴设置为'EVA'
plt.title('QDA决策边界')#将标题设置为'QDA决策边界'

