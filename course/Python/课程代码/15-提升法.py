#第十五章  提升法
#15.2.2  载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_roc_curve
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LogisticRegression
#15.3  回归提升法示例
#15.3.1  变量设置及数据处理
data=pd.read_csv('数据15.1.csv')
X = data.iloc[:,1:]#设置特征变量
y = data.iloc[:,0]#设置响应变量
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, random_state=10)
# 15.3.2  线性回归算法观察
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
#15.3.3  回归提升法（默认参数）
model = GradientBoostingRegressor(random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)
#15.3.4  使用随机搜索寻求最优参数
param_distributions = {'n_estimators': range(100, 300), 'max_depth': range(1, 8),'subsample': np.linspace(0.1,1,10), 'learning_rate': np.linspace(0.1, 1, 10)}
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = RandomizedSearchCV(estimator=GradientBoostingRegressor(random_state=1),param_distributions=param_distributions, cv=kfold, n_iter=100, random_state=1)
model.fit(X_train, y_train)
model.best_params_
model = model.best_estimator_
model.score(X_test, y_test)
#15.3.5  绘制图形观察模型均方误差随基学习器数量变化情况
scores = []
for n_estimators in range(1, 201):
    model = GradientBoostingRegressor(n_estimators=n_estimators, subsample=0.8, max_depth=5, learning_rate=0.2, random_state=10)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    scores.append(mse)
index = np.argmin(scores)
range(1, 201)[index]
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plt.plot(range(1, 201), scores)
plt.axvline(range(1, 201)[index], linestyle='--', color='k', linewidth=1)
plt.xlabel('基学习器数量')
plt.ylabel('MSE')
plt.title('模型均方误差随基学习器数量变化情况')
print(scores)
#15.3.6  绘制图形观察模型拟合优度随基学习器数量变化情况
ScoreAll = []
for n_estimators in range(1, 201):
    model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    ScoreAll.append([n_estimators,score])
ScoreAll = np.array(ScoreAll)
print(ScoreAll)
max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] #找出最高得分对应的索引
print("最优参数以及最高得分:",ScoreAll[max_score])  
plt.figure(figsize=[20,5])
plt.xlabel('n_estimators')
plt.ylabel('拟合优度')
plt.title('拟合优度随n_estimators变化情况')
plt.plot(ScoreAll[:,0],ScoreAll[:,1])
plt.show()
# 15.3.7  回归问题提升法特征变量重要性水平分析
sorted_index = model.feature_importances_.argsort()
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plt.barh(range(X_train.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index])
plt.xlabel('特征变量重要性水平')
plt.ylabel('特征变量')
plt.title('回归问题提升法特征变量重要性水平分析')
plt.tight_layout()
#15.3.8  绘制部分依赖图与个体条件期望图
plt.rcParams['axes.unicode_minus']=False# 解决图表中负号不显示问题。
PartialDependenceDisplay.from_estimator(model, X_train, ['roe','netincomeprofit'], kind='average')#绘制部分依赖图简称PDP图
PartialDependenceDisplay.from_estimator(model, X_train, ['roe','netincomeprofit'],kind='individual')#绘制个体条件期望图（ICE Plot）
PartialDependenceDisplay.from_estimator(model, X_train, ['roe','netincomeprofit'],kind='both')#绘制部分依赖图和个体条件期望图
#15.3.9  最优模型拟合效果图形展示
pred = model.predict(X_test)#对响应变量进行预测
t = np.arange(len(y_test))#求得响应变量在测试样本中的个数，以便绘制图形。
plt.plot(t, y_test, 'r-', linewidth=2, label=u'原值')#绘制响应变量原值曲线。
plt.plot(t, pred, 'g-', linewidth=2, label=u'预测值')#绘制响应变量预测曲线。
plt.legend(loc='upper right')#将图例放在图的右上方。
plt.grid()
plt.show()
#15.3.10 XGBoost回归提升法
#conda install -c anaconda py-xgboost#安装时大家需要把本行代码最前面的#号去掉
import xgboost as xgb
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=147, max_depth=8, 
         subsample=0.6, colsample_bytree=0.8, learning_rate=0.7, random_state=10)
model.fit(X_train, y_train)
model.score(X_test, y_test)

model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=147, random_state=10)
model.fit(X_train, y_train)
model.score(X_test, y_test)
#寻求最优n_estimators
ScoreAll = []
for n_estimators in range(1, 151):
    model =xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, random_state=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    ScoreAll.append([n_estimators,score])
ScoreAll = np.array(ScoreAll)
print(ScoreAll)
max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] #找出最高得分对应的索引
print("最优参数以及最高得分:",ScoreAll[max_score])  


#绘制图形观察“训练样本和测试样本的拟合优度随n_estimators变化情况”
models = []
for n_estimators in range(1, 100):
    model =xgb.XGBRegressor(objective='reg:squarederror', n_estimators=n_estimators, random_state=10)
    model.fit(X_train, y_train)
    models.append(model)
train_scores = [model.score(X_train, y_train) for model in models]
test_scores = [model.score(X_test, y_test) for model in models]
fig, ax = plt.subplots()
ax.set_xlabel("n_estimators")
ax.set_ylabel("拟合优度")
ax.set_title("训练样本和测试样本的拟合优度随n_estimators变化情况")
ax.plot(range(1, 100), train_scores, marker='o', label="训练样本")
ax.plot(range(1, 100), test_scores, marker='o', label="测试样本")
ax.legend()
plt.show()

#15.4 二分类提升法示例
#15.4.1  变量设置及数据处理
data=pd.read_csv('C:/Users/Administrator/.spyder-py3/数据13.1.csv')
X = data.iloc[:,1:]#设置特征变量
y = data.iloc[:,0]#设置响应变量
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3, stratify=y,random_state=10)

#15.4.2  AdaBoost算法
model = AdaBoostClassifier(random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

#15.4.3  二分类提升法（默认参数）
model = GradientBoostingClassifier(random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

#15.3.4  使用随机搜索寻求最优参数
param_distributions = {'n_estimators': range(1, 300), 'max_depth': range(1, 10),
                       'subsample': np.linspace(0.1,1,10), 'learning_rate': np.linspace(0.1, 1, 10)}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=10),
              param_distributions=param_distributions, n_iter=10, cv=kfold, random_state=10)
model.fit(X_train, y_train)
model.best_params_
model = model.best_estimator_
model.score(X_test, y_test)


# 15.3.5  二分类问题提升法特征变量重要性水平分析
sorted_index = model.feature_importances_.argsort()
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plt.barh(range(X_train.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index])
plt.xlabel('特征变量重要性水平')
plt.ylabel('特征变量')
plt.title('二分类问题提升法特征变量重要性水平分析')
plt.tight_layout()

#15.3.6  绘制部分依赖图与个体条件期望图
PartialDependenceDisplay.from_estimator(model, X_train, ['workyears','debtratio'], kind='average')#绘制部分依赖图简称PDP图
PartialDependenceDisplay.from_estimator(model, X_train, ['workyears','debtratio'],kind='individual')#绘制个体条件期望图（ICE Plot）
PartialDependenceDisplay.from_estimator(model, X_train, ['workyears','debtratio'],kind='both')#绘制个体条件期望图（ICE Plot）

#15.4.7  模型性能评价
np.set_printoptions(suppress=True)#不以科学计数法显示，而是直接显示数字
prob = model.predict_proba(X_test)
prob[:5]
pred = model.predict(X_test)
pred[:5]
print(confusion_matrix(y_test, pred))
print(classification_report(y_test,pred))
cohen_kappa_score(y_test, pred)#计算kappa得分
#15.4.8  绘制ROC曲线
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plot_roc_curve(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('二分类问题提升法ROC曲线')#将标题设置为''二分类问题提升法ROC曲线'
#15.4.9  运用两个特征变量绘制二分类提升法决策边界图
X2 = X.iloc[:, [2,5]]#仅选取workyears、debtratio作为特征变量
model = GradientBoostingClassifier(random_state=123)
model.fit(X2,y)
model.score(X2,y)
plot_decision_regions(np.array(X2), np.array(y), model)
plt.xlabel('debtratio')#将x轴设置为'debtratio'
plt.ylabel('workyears')#将y轴设置为'workyears'
plt.title('二分类问题提升法决策边界')#将标题设置为'二分类问题提升法决策边界'

#15.4.10 XGBoost二分类提升法
#conda install -c anaconda py-xgboost
import xgboost as xgb
model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, max_depth=6, 
         subsample=0.6, colsample_bytree=0.8, learning_rate=0.1, random_state=0)
model.fit(X_train, y_train)
model.score(X_test, y_test)

ScoreAll = []
for n_estimators in range(1, 151):
    model =xgb.XGBClassifier(objective='binary:logistic', n_estimators=n_estimators, max_depth=6, 
         subsample=0.6, colsample_bytree=0.8, learning_rate=0.1, random_state=0)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    ScoreAll.append([n_estimators,score])
ScoreAll = np.array(ScoreAll)
print(ScoreAll)
max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] #找出最高得分对应的索引
print("最优参数以及最高得分:",ScoreAll[max_score])  

prob = model.predict_proba(X_test)
prob[:5]
pred = model.predict(X_test)
pred[:5]
print(confusion_matrix(y_test, pred))
print(classification_report(y_test,pred))
cohen_kappa_score(y_test, pred)#计算kappa得分


#15.5 多分类提升法示例

#15.5.1  变量设置及数据处理
data=pd.read_csv('C:/Users/Administrator/.spyder-py3/数据15.2.csv')
X = data.iloc[:,1:]#设置特征变量
y = data.iloc[:,0]#设置响应变量
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3,stratify=y, random_state=10)
#15.5.2  多元Logistic回归算法观察
model=LogisticRegression(multi_class='multinomial',solver = 'newton-cg', C=1e10, max_iter=1e3)
model.fit(X_train, y_train)
model.score(X_test, y_test)

#15.5.3  多分类提升法（默认参数）
model = GradientBoostingClassifier(random_state=123)
model.fit(X_train, y_train)
model.score(X_test, y_test)

#15.5.4  使用随机搜索寻求最优参数
param_distributions = {'n_estimators': range(100, 200),'max_depth': range(1, 9),
                       'subsample': np.linspace(0.1, 1, 10),'learning_rate': np.linspace(0.1, 1, 10)}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = RandomizedSearchCV(estimator=GradientBoostingClassifier(random_state=10),
              param_distributions=param_distributions, cv=kfold, random_state=10)
model.fit(X_train, y_train)
model.best_params_
model = model.best_estimator_
model.score(X_test, y_test)

# 15.5.5  多分类问题提升法特征变量重要性水平分析
sorted_index = model.feature_importances_.argsort()
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
plt.barh(range(X_train.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index])
plt.xlabel('特征变量重要性水平')
plt.ylabel('特征变量')
plt.title('多分类问题提升法特征变量重要性水平分析')
plt.tight_layout()

#15.5.6  绘制部分依赖图与个体条件期望图
PartialDependenceDisplay.from_estimator(model, X_train, ['income','consume'], target=3,kind='average')#绘制部分依赖图简称PDP图
PartialDependenceDisplay.from_estimator(model, X_train, ['income','consume'],target=3,kind='individual')#绘制个体条件期望图（ICE Plot）
PartialDependenceDisplay.from_estimator(model, X_train, ['income','consume'],target=3,kind='both')#绘制个体条件期望图（ICE Plot）

#15.5.7  模型性能评价
prob = model.predict_proba(X_test)
prob[:5]
pred = model.predict(X_test)
pred[:5]
print(confusion_matrix(y_test, pred))
sns.heatmap(confusion_matrix(y_test, pred),cmap='Blues', annot=True)
plt.tight_layout()
print(classification_report(y_test,pred))
cohen_kappa_score(y_test, pred)#计算kappa得分

#15.3.8 XGBoost多分类提升法
#conda install -c anaconda py-xgboost
import xgboost as xgb


model = xgb.XGBClassifier(objective='multi:softprob', n_estimators=100, max_depth=5, 
         subsample=0.9, colsample_bytree=0.8, learning_rate=0.5, random_state=0)
model.fit(X_train, y_train)
model.score(X_test, y_test)

ScoreAll = []
for n_estimators in range(1, 101):
    model =xgb.XGBClassifier(objective='multi:softprob', n_estimators=n_estimators, random_state=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    ScoreAll.append([n_estimators,score])
ScoreAll = np.array(ScoreAll)
print(ScoreAll)
max_score = np.where(ScoreAll==np.max(ScoreAll[:,1]))[0][0] #找出最高得分对应的索引
print("最优参数以及最高得分:",ScoreAll[max_score])  

