#  K近邻算法
# 10.2.2  载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.metrics import mean_squared_error
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import plot_roc_curve

#10.3  回归问题K近邻算法示例
#10.3.1  变量设置及数据处理
data=pd.read_csv('数据4.1.csv')
X = data.drop(['V1'],axis=1)#设置特征变量，即除V1之外的全部变量
y = data['V1']#设置响应变量，即V1
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, random_state=123)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
#10.3.2  构建K近邻回归算法模型
#K近邻算法(K=1)
model = KNeighborsRegressor(n_neighbors=1)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)
pred
mean_squared_error(y_test, pred)
model.score(X_test_s, y_test)
#K近邻算法(K=17)
model = KNeighborsRegressor(n_neighbors=17)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)
pred
mean_squared_error(y_test, pred)
model.score(X_test_s, y_test)
#K近邻算法(K=9)
model = KNeighborsRegressor(n_neighbors=9)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)
pred
mean_squared_error(y_test, pred)
model.score(X_test_s, y_test)
#10.3.3  如何选择最优的K值
scores = []
ks = range(1, 17)
for k in ks:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train_s, y_train)
    score = model.score(X_test_s, y_test)
    scores.append(score)
max(scores)
index_max = np.argmax(scores)
index_max
print(f'最优K值: {ks[index_max]}')
#K近邻算法(选取最优K的图形展示)
plt.rcParams['font.sans-serif'] = ['SimHei']#本代码的含义是解决图表中中文显示问题。
plt.plot(ks, scores, 'o-')
plt.xlabel('K')
plt.axvline(ks[index_max], linewidth=1, linestyle='--', color='k')
plt.ylabel('拟合优度')
plt.title('不同K取值下的拟合优度')
plt.tight_layout()
#10.3.4  最优模型拟合效果图形展示
model = KNeighborsRegressor(n_neighbors=4)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)
mean_squared_error(y_test, pred)
model.score(X_test_s, y_test)
t = np.arange(len(y_test))
plt.rcParams['font.sans-serif'] = ['SimHei']#本代码的含义是解决图表中中文显示问题。
plt.plot(t, y_test, 'r-', linewidth=2, label=u'原值')
plt.plot(t, pred, 'g-', linewidth=2, label=u'预测值')
plt.legend(loc='upper right')
plt.grid()
plt.show()

   
#10.4 分类问题K近邻算法示例
#10.4.1  变量设置及数据处理
data=pd.read_csv('数据8.1.csv')
X = data.drop(['V1'],axis=1)#设置特征变量，即除V1之外的全部变量
y = data['V1']#设置响应变量，即V1
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, random_state=123)
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
#10.4.2  构建K近邻分类算法模型
#K近邻算法(K=1)
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)
model.score(X_test_s, y_test)
#K近邻算法(K=33)
model = KNeighborsClassifier(n_neighbors=33)
model.fit(X_train_s, y_train)
pred = model.predict(X_test_s)
model.score(X_test_s, y_test)
#10.4.3  如何选择最优的K值
scores = []
ks = range(1, 33)
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_s, y_train)
    score = model.score(X_test_s, y_test)
    scores.append(score)
max(scores)
index_max = np.argmax(scores)
print(f'最优K值: {ks[index_max]}')
#K近邻算法(选取最优K的图形展示)
plt.rcParams['font.sans-serif'] = ['SimHei']#本代码的含义是解决图表中中文显示问题。
plt.plot(ks, scores, 'o-')#绘制K取值和模型预测准确率的关系图
plt.xlabel('K')#设置X轴标签为“K”
plt.axvline(ks[index_max], linewidth=1, linestyle='--', color='k')
plt.ylabel('预测准确率')
plt.title('不同K取值下的预测准确率')
plt.tight_layout()
#10.4.4  最优模型拟合效果图形展示
model = KNeighborsClassifier(n_neighbors=9)#选取前面得到的最优K值9构建K近邻算法模型
model.fit(X_train_s, y_train)#基于训练样本进行拟合
pred = model.predict(X_test_s)#对响应变量进行预测
t = np.arange(len(y_test))#求得响应变量在测试样本中的个数，以便绘制图形。
plt.rcParams['font.sans-serif'] = ['SimHei']#本代码的含义是解决图表中中文显示问题。
plt.plot(t, y_test, 'r-', linewidth=2, label=u'原值')#绘制响应变量原值曲线。
plt.plot(t, pred, 'g-', linewidth=2, label=u'预测值')#绘制响应变量预测曲线。
plt.legend(loc='upper right')#将图例放在图的右上方。
plt.grid()
plt.show()
#10.4.5  绘制K近邻分类算法ROC曲线
scaler = StandardScaler()
scaler.fit(X)
X_s = scaler.transform(X)
plt.rcParams['font.sans-serif'] = ['SimHei']#本代码的含义是解决图表中中文显示问题。
plot_roc_curve(model,X_s, y)#本代码的含义是绘制ROC曲线，并计算AUC值。
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)#本代码的含义是在图中增加45度黑色虚线，以便观察ROC曲线性能。
plt.title('K近邻算法ROC曲线')#本代码的含义是设置标题为'K近邻算法ROC曲线'。

#10.4.6  运用两个特征变量绘制K近邻算法决策边界图
X2 = X.iloc[:, 0:2]#仅选取V2存款规模、V3EVA作为特征变量
model = KNeighborsClassifier(n_neighbors=9)#使用K近邻算法，K=9
scaler = StandardScaler()
scaler.fit(X2)
X2_s = scaler.transform(X2)
model.fit(X2_s, y)#使用fit方法进行拟合
model.score(X2_s, y)#计算模型预测准确率

plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plot_decision_regions(np.array(X2_s), np.array(y), model)
plt.xlabel('存款规模')#将x轴设置为'存款规模'
plt.ylabel('EVA')#将y轴设置为'EVA'
plt.title('K近邻算法决策边界')#将标题设置为'K近邻算法决策边界'

#10.4.7  普通KNN算法、带权重KNN、指定半径KNN三种算法对比
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=9)))
models.append(('KNN with weights', KNeighborsClassifier(n_neighbors=9, weights='distance')))
models.append(('Radius Neighbors', RadiusNeighborsClassifier(radius=100)))
#基于验证集法
results = []
for name, model in models:
    model.fit(X_train_s, y_train)
    results.append((name, model.score(X_test_s, y_test)))
for i in range(len(results)):
    print('name: {}; score: {}'.format(results[i][0], results[i][1]))
    
#基于10折交叉验证法
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=9)))
models.append(('KNN with weights', KNeighborsClassifier(n_neighbors=9, weights='distance')))
models.append(('Radius Neighbors', RadiusNeighborsClassifier(radius=10000)))
results = []
for name, model in models:
    kfold = KFold(n_splits=10)
    cv_result = cross_val_score(model, X_s, y, cv=kfold)
    results.append((name, cv_result))
for i in range(len(results)):
    print('name: {}; cross_val_score: {}'.format(results[i][0], results[i][1].mean()))
