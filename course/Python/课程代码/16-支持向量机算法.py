#  支持向量机算法
#16.2.2  载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions

#16.3  回归支持向量机算法示例
#16.3.1  变量设置及数据处理
data=pd.read_csv('数据15.1.csv')
X = data.iloc[:,1:]#设置特征变量
y = data.iloc[:,0]#设置响应变量
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, random_state=10)
#数据标准化
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
#16.3.2  回归支持向量机算法（默认参数）
# 线性核函数算法
model = SVR(kernel='linear')
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
# 多项式核函数算法
model = SVR(kernel='poly')
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
# 径向基函数算法
model = SVR(kernel='rbf')#选择模型所使用的核函数为径向基函数rbf
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
# sigmod核函数算法
model = SVR(kernel='sigmoid')#选择模型所使用的核函数为径向基函数rbf
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

#16.3.3  通过K折交叉验证寻求最优参数
#通过K折交叉验证寻求最优参数（线性核函数算法）
param_grid = {'C': [0.01, 0.1, 1, 10, 50, 100, 150], 'epsilon': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVR(kernel='linear'), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model = model.best_estimator_
len(model.support_)
model.support_vectors_
model.score(X_test_s, y_test)
#通过K折交叉验证寻求最优参数（多项式核函数算法）
param_grid = {'C': [0.01, 0.1, 1, 10, 50, 100, 150], 'epsilon': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVR(kernel='poly'), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model = model.best_estimator_
model.score(X_test_s, y_test)
#通过K折交叉验证寻求最优参数（sigmod核函数算法）
param_grid = {'C': [0.01, 0.1, 1, 10, 50, 100, 150], 'epsilon': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVR(kernel='sigmoid'), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model = model.best_estimator_
model.score(X_test_s, y_test)
#通过K折交叉验证寻求最优参数（径向基函数算法）
param_grid = {'C': [0.01, 0.1, 1, 10, 50, 100, 150], 'epsilon': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]}
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model = model.best_estimator_
model.score(X_test_s, y_test)

#16.3.4  最优模型拟合效果图形展示
pred = model.predict(X_test_s)#对响应变量进行预测
t = np.arange(len(y_test))#求得响应变量在测试样本中的个数，以便绘制图形。
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plt.plot(t, y_test, 'r-', linewidth=2, label=u'原值')#绘制响应变量原值曲线。
plt.plot(t, pred, 'g-', linewidth=2, label=u'预测值')#绘制响应变量预测曲线。
plt.legend(loc='upper right')#将图例放在图的右上方。
plt.grid()
plt.show()


#16.4 二分类支持向量机算法示例
#16.4.1  变量设置及数据处理
#变量设置及数据处理
data=pd.read_csv('数据13.1.csv')
X = data.iloc[:,1:]#设置特征变量
y = data.iloc[:,0]#设置响应变量
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, random_state=10)
#数据标准化
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)
#16.4.2  二分类支持向量机算法（默认参数）
# 线性核函数算法
model = SVC(kernel="linear", random_state=10)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
# 多项式核函数算法
model = SVC(kernel="poly", degree=2, random_state=10)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
# 多项式核函数算法
model = SVC(kernel="poly", degree=3, random_state=10)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
# 径向基函数算法
model = SVC(kernel="rbf", random_state=10)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
# sigmod核函数算法
model = SVC(kernel="sigmoid",random_state=10)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
#16.4.3  通过K折交叉验证寻求最优参数
#通过K折交叉验证寻求最优参数（线性核函数算法）
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel="linear", random_state=123), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model.score(X_test_s, y_test)
#通过K折交叉验证寻求最优参数（多项式核函数算法）
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel="poly", random_state=123), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model.score(X_test_s, y_test)
#通过K折交叉验证寻求最优参数（径向基函数算法）
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel="rbf", random_state=123), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model.score(X_test_s, y_test)
#通过K折交叉验证寻求最优参数（sigmod核函数算法）
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel="sigmoid", random_state=123), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model.score(X_test_s, y_test)
#16.4.4  模型性能评价
np.set_printoptions(suppress=True)#不以科学计数法显示，而是直接显示数字
pred = model.predict(X_test_s)
pred[:5]
print(confusion_matrix(y_test, pred))
print(classification_report(y_test,pred))
cohen_kappa_score(y_test, pred)#计算kappa得分
#16.4.5  绘制ROC曲线
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plot_roc_curve(model, X_test_s, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('二分类支持向量机算法ROC曲线')#将标题设置为'二分类支持向量机算法ROC曲线'
#16.4.6  运用两个特征变量绘制二分类支持向量机算法决策边界图
X2 = X.iloc[:, [2,5]]#仅选取workyears、debtratio作为特征变量
scaler = StandardScaler()
X2_s =scaler.fit_transform(X2)
model = SVC(kernel="rbf",gamma=0.01, C=10,random_state=10)
model.fit(X2_s ,y)
model.score(X2_s ,y)
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plt.rcParams['axes.unicode_minus']=False# 解决图表中负号不显示问题。
plot_decision_regions(np.array(X2_s ), np.array(y), model)
plt.xlabel('debtratio')#将x轴设置为'debtratio'
plt.ylabel('workyears')#将y轴设置为'workyears'
plt.title('二分类支持向量机算法决策边界')#将标题设置为'二分类支持向量机算法决策边界'

model = SVC(kernel="rbf",gamma=0.01, C=50000,random_state=10)
model.fit(X2_s ,y)
model.score(X2_s ,y)
plot_decision_regions(np.array(X2_s ), np.array(y), model)
plt.xlabel('debtratio')#将x轴设置为'debtratio'
plt.ylabel('workyears')#将y轴设置为'workyears'
plt.title('二分类支持向量机算法决策边界')#将标题设置为'二分类支持向量机算法决策边界'

model = SVC(kernel="rbf",gamma=0.01, C=0.5,random_state=10)
model.fit(X2_s ,y)
model.score(X2_s ,y)
plot_decision_regions(np.array(X2_s ), np.array(y), model)
plt.xlabel('debtratio')#将x轴设置为'debtratio'
plt.ylabel('workyears')#将y轴设置为'workyears'
plt.title('二分类支持向量机算法决策边界')#将标题设置为'二分类支持向量机算法决策边界'

model = SVC(kernel="rbf",gamma=10, C=10,random_state=10)
model.fit(X2_s ,y)
model.score(X2_s ,y)
plot_decision_regions(np.array(X2_s ), np.array(y), model)
plt.xlabel('debtratio')#将x轴设置为'debtratio'
plt.ylabel('workyears')#将y轴设置为'workyears'
plt.title('二分类支持向量机算法决策边界')#将标题设置为'二分类支持向量机算法决策边界'

model = SVC(kernel="rbf",gamma=0.001, C=10,random_state=10)
model.fit(X2_s ,y)
model.score(X2_s ,y)
plot_decision_regions(np.array(X2_s ), np.array(y), model)
plt.xlabel('debtratio')#将x轴设置为'debtratio'
plt.ylabel('workyears')#将y轴设置为'workyears'
plt.title('二分类支持向量机算法决策边界')#将标题设置为'二分类支持向量机算法决策边界'

#16.5 多分类支持向量机算法示例

#16.5.1  变量设置及数据处理
data=pd.read_csv('数据15.2.csv')
X = data.iloc[:,1:]#设置特征变量
y = data.iloc[:,0]#设置响应变量
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3,stratify=y, random_state=10)

#数据标准化
scaler = StandardScaler()
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

#16.5.2  多分类支持向量机算法（一对一）
classifier = svm.SVC(C=1000, kernel='rbf', gamma='scale', decision_function_shape='ovo')# ovr:一对多策略，ovo表示一对一
classifier.fit(X_train_s, y_train)
print("训练集：", classifier.score(X_train_s, y_train))#计算svc分类器对训练集的准确率
print("测试集：", classifier.score(X_test_s, y_test))#计算svc分类器对测试集的准确率
print('train_decision_function:\n', classifier.decision_function(X_train_s))#查看内部决策函数，返回的是样本到超平面的距离
print('predict_result:\n', classifier.predict(X_train_s))# 查看预测结果

#运用两个特征变量绘制多分类支持向量机算法决策边界图
X2 = X.iloc[:, [3,5]]#仅选取workyears、debtratio作为特征变量
X2_s =scaler.fit_transform(X2)
model = SVC(kernel="rbf",C=10,random_state=10)
model.fit(X2_s ,y)
model.score(X2_s ,y)
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plt.rcParams['axes.unicode_minus']=False# 解决图表中负号不显示问题。
plot_decision_regions(np.array(X2_s ), np.array(y), model)
plt.xlabel('income')#将x轴设置为'income'
plt.ylabel('consume')#将y轴设置为'consume'
plt.title('多分类支持向量机算法决策边界')#将标题设置为'多分类支持向量机算法决策边界'

#16.5.3  多分类支持向量机算法（默认参数）
# 线性核函数算法
model = SVC(kernel="linear", random_state=123)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
# 多项式核函数算法
model = SVC(kernel="poly", degree=2, random_state=123)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
# 多项式核函数算法
model = SVC(kernel="poly", degree=3, random_state=123)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
# 径向基函数算法
model = SVC(kernel='rbf', random_state=123)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)
# sigmod核函数算法
model = SVC(kernel="sigmoid",random_state=123)
model.fit(X_train_s, y_train)
model.score(X_test_s, y_test)

#16.5.4  通过K折交叉验证寻求最优参数
#通过K折交叉验证寻求最优参数（线性核函数算法）
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1, 10]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel='linear',random_state=123), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model.score(X_test_s, y_test)
#通过K折交叉验证寻求最优参数（多项式核函数算法）
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1, 10]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel='poly',random_state=123), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model.score(X_test_s, y_test)
#通过K折交叉验证寻求最优参数（径向基函数算法）
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1, 10]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel='rbf',random_state=123), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model.score(X_test_s, y_test)
#通过K折交叉验证寻求最优参数（sigmod核函数算法）
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1, 10]}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(SVC(kernel='sigmoid',random_state=123), param_grid, cv=kfold)
model.fit(X_train_s, y_train)
model.best_params_
model.score(X_test_s, y_test)
#16.5.5  模型性能评价
np.set_printoptions(suppress=True)#不以科学计数法显示，而是直接显示数字
pred = model.predict(X_test_s)
pred[:5]
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test, pred),cmap='Blues', annot=True)
plt.tight_layout()
print(classification_report(y_test,pred))
cohen_kappa_score(y_test, pred)#计算kappa得分
