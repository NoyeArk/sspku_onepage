# 神经网络算法
#17.2.2载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_roc_curve

#17.3回归神经网络算法示例
#17.3.1变量设置及数据处理
#变量设置及数据处理
data=pd.read_csv('数据15.1.csv')
X = data.iloc[:,1:]#设置特征变量
y = data.iloc[:,0]#设置响应变量
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, random_state=10)
#数据标准化
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
X_train_s = pd.DataFrame(X_train_s, columns=X_train.columns)
X_test_s = pd.DataFrame(X_test_s, columns=X_test.columns)

#17.3.2单隐藏层的多层感知机算法
model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(5,), random_state=10, max_iter=3000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
model.n_iter_
model.intercepts_
model.coefs_

#17.3.3神经网络特征变量重要性水平分析
perm= permutation_importance(model, X_test_s, y_test, n_repeats=10, random_state=10)
dir(perm)
sorted_index =perm.importances_mean.argsort()
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plt.barh(range(X_train.shape[1]),perm.importances_mean[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index])
plt.xlabel('特征变量重要性水平均值')
plt.ylabel('特征变量')
plt.title('神经网络特征变量重要性水平分析')
plt.tight_layout()
#17.3.4绘制部分依赖图与个体条件期望图
plt.rcParams['axes.unicode_minus']=False# 解决图表中负号不显示问题。
PartialDependenceDisplay.from_estimator(model, X_train_s, ['roe','rdgrow'], kind='average')#绘制部分依赖图简称PDP图
PartialDependenceDisplay.from_estimator(model, X_train_s, ['roe','rdgrow'],kind='individual')#绘制个体条件期望图（ICE Plot）
PartialDependenceDisplay.from_estimator(model, X_train_s, ['roe','rdgrow'],kind='both')#绘制部分依赖图和个体条件期望图

#17.3.5拟合优度随神经元个数变化的可视化展示
models = []
for n_neurons in range(1, 50):
    model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(n_neurons,), random_state=10, max_iter=5000)
    model.fit(X_train_s, y_train)
    models.append(model)
train_scores = [model.score(X_train_s, y_train) for model in models]
test_scores = [model.score(X_test_s, y_test) for model in models]
fig, ax = plt.subplots()
ax.set_xlabel("神经元个数")
ax.set_ylabel("拟合优度")
ax.set_title("训练样本和测试样本的拟合优度随神经元个数变化情况")
ax.plot(range(1, 50), train_scores, marker='o', label="训练样本")
ax.plot(range(1, 50), test_scores, marker='o', label="测试样本")
ax.legend()
plt.show()

#17.3.6 通过K折交叉验证寻求单隐藏层最优神经元个数
param_grid = {'hidden_layer_sizes':[(1,),(2,),(3,),(4,),(5,),(10,),(15,),(20,)]}
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(MLPRegressor(solver='lbfgs', random_state=10, max_iter=2000), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model = model.best_estimator_
model.score(X_test_s, y_test)

PartialDependenceDisplay.from_estimator(model, X_train_s, ['roe','rdgrow'], kind='average')#绘制部分依赖图简称PDP图
PartialDependenceDisplay.from_estimator(model, X_train_s, ['roe','rdgrow'],kind='individual')#绘制个体条件期望图（ICE Plot）
PartialDependenceDisplay.from_estimator(model, X_train_s, ['roe','rdgrow'],kind='both')#绘制部分依赖图和个体条件期望图

#17.3.7双隐藏层的多层感知机算法
model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(5, 3), random_state=10, max_iter=3000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

best_score = 0
best_sizes = (1, 1)
for i in range(1, 5):
    for j in range(1, 5):
        model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=(i, j), random_state=10, max_iter=2000)
        model.fit(X_train_s, y_train)
        score = model.score(X_test_s, y_test)
        if best_score < score:
            best_score = score
            best_sizes = (i, j)
best_score
best_sizes

#17.3.8最优模型拟合效果图形展示
pred = model.predict(X_test_s)#对响应变量进行预测
t = np.arange(len(y_test))#求得响应变量在测试样本中的个数，以便绘制图形。
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plt.plot(t, y_test, 'r-', linewidth=2, label=u'原值')#绘制响应变量原值曲线。
plt.plot(t, pred, 'g-', linewidth=2, label=u'预测值')#绘制响应变量预测曲线。
plt.legend(loc='upper right')#将图例放在图的右上方。
plt.grid()
plt.show()

#17.4二分类神经网络算法示例
#17.4.1变量设置及数据处理
#变量设置及数据处理
data=pd.read_csv('数据13.1.csv')
X = data.iloc[:,1:]#设置特征变量
y = data.iloc[:,0]#设置响应变量
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, stratify=y,random_state=10)
#数据标准化
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
X_train_s = pd.DataFrame(X_train_s, columns=X_train.columns)
X_test_s = pd.DataFrame(X_test_s, columns=X_test.columns)
#17.4.2单隐藏层二分类问题神经网络算法
model = MLPClassifier(solver='lbfgs',activation='relu',hidden_layer_sizes=(3,), random_state=10, max_iter=2000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
model.n_iter_

model=MLPClassifier(solver='sgd',learning_rate_init=0.01,learning_rate='constant',tol=0.0001,activation='relu',hidden_layer_sizes=(3,),random_state=10,max_iter=2000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

#17.4.3双隐藏层二分类问题神经网络算法
model = MLPClassifier(solver='lbfgs',activation='relu',hidden_layer_sizes=(3, 2), random_state=10, max_iter=2000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
model.n_iter_

#17.4.4早停策略减少过拟合问题

model = MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(20, 20), random_state=10, early_stopping=True, validation_fraction=0.25, max_iter=2000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
model.n_iter_


#17.4.5正则化（权重衰减）策略减少过拟合问题

model = MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(20, 20), random_state=10, alpha=0.1, max_iter=2000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
model.n_iter_

model = MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(20, 20), random_state=10, alpha=1, max_iter=2000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

model = MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(20, 20), random_state=10, alpha=0.001, max_iter=2000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

#17.4.6模型性能评价
np.set_printoptions(suppress=True)#不以科学计数法显示，而是直接显示数字
prob = model.predict_proba(X_test_s)
prob[:5]
pred = model.predict(X_test_s)
pred[:5]
print(confusion_matrix(y_test, pred))
print(classification_report(y_test,pred))
cohen_kappa_score(y_test, pred)#计算kappa得分

#17.4.7绘制ROC曲线
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plot_roc_curve(model, X_test_s, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('二分类问题神经网络算法ROC曲线')#将标题设置为''二分类问题神经网络算法ROC曲线'

#17.4.8运用两个特征变量绘制二分类神经网络算法决策边界图
X2_test_s = X_test_s.iloc[:, [2,5]]#仅选取workyears、debtratio作为特征变量
model=MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(20,20),random_state=10,alpha=0.1,max_iter=2000)
model.fit(X2_test_s,y_test)
model.score(X2_test_s,y_test)
plot_decision_regions(np.array(X2_test_s), np.array(y_test), model)
plt.xlabel('debtratio')#将x轴设置为'debtratio'
plt.ylabel('workyears')#将y轴设置为'workyears'
plt.title('二分类问题神经网络算法决策边界')#将标题设置为'二分类问题神经网络算法决策边界'


#17.5多分类神经网络算法示例
#17.5.1变量设置及数据处理
#变量设置及数据处理
data=pd.read_csv('数据15.2.csv')
X = data.iloc[:,1:]#设置特征变量
y = data.iloc[:,0]#设置响应变量
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3,stratify=y, random_state=10)

#数据标准化
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
X_train_s = pd.DataFrame(X_train_s, columns=X_train.columns)
X_test_s = pd.DataFrame(X_test_s, columns=X_test.columns)
#17.5.2单隐藏层多分类问题神经网络算法
model=MLPClassifier(solver='sgd',learning_rate_init=0.01,learning_rate='constant',tol=0.0001,activation='relu',hidden_layer_sizes=(3,),random_state=10,max_iter=2000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

#17.5.3双隐藏层多分类问题神经网络算法
model=MLPClassifier(solver='sgd',learning_rate_init=0.01,learning_rate='constant',tol=0.0001,activation='relu',hidden_layer_sizes=(3,2),random_state=10,max_iter=2000)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)


#17.5.4模型性能评价
np.set_printoptions(suppress=True)#不以科学计数法显示，而是直接显示数字
pred = model.predict(X_test_s)
pred[:5]
print(confusion_matrix(y_test, pred))
print(classification_report(y_test,pred))
cohen_kappa_score(y_test, pred)#计算kappa得分

#17.4.5运用两个特征变量绘制多分类神经网络算法决策边界图
X2_train_s = X_train_s.iloc[:, [3,5]]#仅选取income、consume作为特征变量
X2_test_s = X_test_s.iloc[:, [3,5]]#仅选取income、consume作为特征变量
model=MLPClassifier(solver='sgd',learning_rate_init=0.01,learning_rate='constant',tol=0.0001,activation='relu',hidden_layer_sizes=(3,2),random_state=10,max_iter=2000)
model.fit(X2_train_s, y_train)
model.score(X2_test_s, y_test)#运行结果为0.8557377049180328。
plt.rcParams['font.sans-serif']=['SimHei']#解决图表中中文显示问题
plt.rcParams['axes.unicode_minus']=False#解决图表中负号不显示问题。
plot_decision_regions(np.array(X2_test_s),np.array(y_test),model)
plt.xlabel('income')#将x轴设置为'income'
plt.ylabel('consume')#将y轴设置为'consume'
plt.title('多分类神经网络算法决策边界')#将标题设置为'多分类神经网络算法决策边界'，
