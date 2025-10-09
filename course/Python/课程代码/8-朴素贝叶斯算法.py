#第八章  朴素贝叶斯算法
# 2.2  载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_decision_regions
#3  高斯朴素贝叶斯算法示例
#3.1  数据读取及观察
data=pd.read_csv('数据8.1.csv')
data.info()
data.isnull().values.any()
data.V1.value_counts()
data.V1.value_counts(normalize=True)
#3.2  将样本示例全集分割为训练样本和测试样本
X = data.drop(['V1'],axis=1)#设置特征变量，即除V1之外的全部变量
y = data['V1']#设置响应变量，即V1
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, stratify=y, random_state=123)
#3.3  高斯朴素贝叶斯算法拟合
model = GaussianNB()
model.fit(X_train, y_train)
model.score(X_test, y_test) 
#3.4  绘制ROC曲线
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plot_roc_curve(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('高斯朴素贝叶斯ROC曲线')
#3.5  运用两个特征变量绘制高斯朴素贝叶斯决策边界图
X2 = X.iloc[:, 0:2]
model = GaussianNB()
model.fit(X2, y)
model.score(X2, y)
plt.rcParams['axes.unicode_minus']=False# 解决图表中负号不显示问题。
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plot_decision_regions(np.array(X2), np.array(y), model)
plt.xlabel('存款规模')#将x轴设置为'存款规模'
plt.ylabel('EVA')#将y轴设置为'EVA'
plt.title('高斯朴素贝叶斯决策边界')#将标题设置为'高斯朴素贝叶斯决策边界'

#4  多项式、补集、二项式朴素贝叶斯算法示例
#4.1  数据读取及观察
data=pd.read_csv('数据8.2.csv')
data.V1.value_counts()#观察样本示例全集中响应变量的分类计数值
data.V1.value_counts(normalize=True)#观察样本示例全集中响应变量的分类占比
#4.2  将样本示例全集分割为训练样本和测试样本
X = data.drop(['V1'],axis=1)#设置特征变量，即除V1之外的全部变量
y = data['V1']#设置响应变量，即V1
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, stratify=y, random_state=123)
#4.3  多项式、补集、二项式朴素贝叶斯算法拟合
#1、多项朴素贝叶斯方法
model = MultinomialNB(alpha=0)#采取多项朴素贝叶斯方法，不进行拉普拉斯修正
model.fit(X_train, y_train)#基于训练样本，使用fit方进行拟合
model.score(X_test, y_test)#基于测试样本，计算模型预测准确率

model = MultinomialNB(alpha=1)#采取多项朴素贝叶斯方法，进行拉普拉斯修正
model.fit(X_train, y_train)#基于训练样本，使用fit方进行拟合
model.score(X_test, y_test)#基于测试样本，计算模型预测准确率
#2、补集朴素贝叶斯方法
model = ComplementNB(alpha=1)#采取补集朴素贝叶斯方法，进行拉普拉斯修正
model.fit(X_train, y_train)#基于训练样本，使用fit方进行拟合
model.score(X_test, y_test)#基于测试样本，计算模型预测准确率
#3、二项朴素贝叶斯方法
model = BernoulliNB(alpha=1)#采取二项朴素贝叶斯方法，进行拉普拉斯修正
model.fit(X_train, y_train)#基于训练样本，使用fit方进行拟合
model.score(X_test, y_test)#基于测试样本，计算模型预测准确率

model = BernoulliNB(binarize=2, alpha=1)#采取二项朴素贝叶斯方法，设置参数binarize=2，进行拉普拉斯修正
model.fit(X_train, y_train)#基于训练样本，使用fit方进行拟合
model.score(X_test, y_test)#基于测试样本，计算模型预测准确率

# 4.4  寻求二项式朴素贝叶斯算法拟合的最优参数
#1、通过将样本分割为训练样本、验证样本、测试样本的方式寻找最优参数
X_trainval, X_test, y_trainval, y_test =  train_test_split(X, y, test_size=0.3, stratify=y, random_state=10)#随机抽取30%的样本作为测试集
X_train, X_val, y_train, y_val =  train_test_split(X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=100)#从剩余的70%的样本集（训练集+验证集）中随机抽取20%作为验证集
y_train.shape, y_val.shape, y_test.shape#观察样本集形状，得到各样本集的样本容量。
best_val_score = 0
for binarize in np.arange(0, 5.5, 0.5):
    for alpha in np.arange(0, 1.1, 0.1):
        model = BernoulliNB(binarize=binarize, alpha=alpha)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val) 
        if score > best_val_score:
            best_val_score = score
            best_val_parameters = {'binarize': binarize, 'alpha': alpha}
best_val_score#计算验证集的最优预测准确率
best_val_parameters#得到最优参数
model = BernoulliNB(**best_val_parameters)#使用前面得到的最优参数构建伯努利朴素贝叶斯模型
model.fit(X_trainval, y_trainval)#使用前述70%的样本集（训练集+验证集）进行伯努利朴素贝叶斯估计
model.score(X_test, y_test) #输出基于测试样本得到的预测准确率
   
#2、采用10折交叉验证方法寻找最优参数
param_grid = {'binarize': np.arange(0, 5.5, 0.5), 'alpha': np.arange(0, 1.1, 0.1)}#定义字典形式的参数网络
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)#保持每折子样本中响应变量各类别数据占比相同
model = GridSearchCV(BernoulliNB(), param_grid, cv=kfold)#构建伯努利朴素贝叶斯模型，使用上步得到的参数网络，使用10折交叉验证方法进行交叉验证
model.fit(X_trainval, y_trainval)#使用前述70%的样本集（训练集+验证集）进行伯努利朴素贝叶斯估计
model.score(X_test, y_test)#输出基于测试样本得到的预测准确率
model.best_params_#输出最优参数
model.best_score_ #计算最优预测准确率
outputs = pd.DataFrame(model.cv_results_)#得到每个参数组合的详细交叉验证信息，并转化为数据框形式 
pd.set_option('display.max_columns', None)
outputs.head(3)#展示前3行
scores = np.array(outputs.mean_test_score).reshape(11,11)#将平均预测准确率组成11*11矩阵
ax = sns.heatmap(scores, cmap='Oranges', annot=True, fmt='.3f')# 绘制热图将10折交叉验证方法寻找最优参数过程可视化
ax.set_xlabel('binarize')
ax.set_xticklabels([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5,5])
ax.set_ylabel('alpha')
ax.set_yticklabels([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
plt.tight_layout()

#4.5  最优二项式朴素贝叶斯算法模型性能评价

prob = model.predict_proba(X_test)
prob[:5]
pred = model.predict(X_test)
pred[:5]
print(confusion_matrix(y_test, pred))
print(classification_report(y_test,pred))
cohen_kappa_score(y_test, pred)#计算kappa得分
