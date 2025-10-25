#  随机森林算法
# 14.2.2  载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_roc_curve
from sklearn.inspection import PartialDependenceDisplay
from mlxtend.plotting import plot_decision_regions

# 14.3 分类问题随机森林算法示例
# 14.3.1  变量设置及数据处理
data = pd.read_csv("数据13.1.csv")
X = data.iloc[:, 1:]  # 设置特征变量
y = data.iloc[:, 0]  # 设置响应变量
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=10
)
# 14.3.2  二元Logistic回归、单颗分类决策树算法观察
model = LogisticRegression(C=1e10, max_iter=1000, fit_intercept=True)
model.fit(X_train, y_train)
model.score(X_test, y_test)
# 单颗分类决策树算法
model = DecisionTreeClassifier()
path = model.cost_complexity_pruning_path(X_train, y_train)
param_grid = {"ccp_alpha": path.ccp_alphas}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
model = GridSearchCV(DecisionTreeClassifier(random_state=10), param_grid, cv=kfold)
model.fit(X_train, y_train)
print("最优alpha值：", model.best_params_)
model = model.best_estimator_
print("最优预测准确率：", model.score(X_test, y_test))
# 14.3.3  装袋法分类算法
model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(random_state=10),
    n_estimators=300,
    max_samples=0.8,
    random_state=0,
)
model.fit(X_train, y_train)
model.score(X_test, y_test)
# 14.3.4  随机森林分类算法
model = RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=10)
model.fit(X_train, y_train)
model.score(X_test, y_test)
# 14.3.5  寻求max_features最优参数
scores = []
for max_features in range(1, X.shape[1] + 1):
    model = RandomForestClassifier(
        max_features=max_features, n_estimators=300, random_state=10
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)
index = np.argmax(scores)
range(1, X.shape[1] + 1)[index]
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正常显示中文
plt.plot(range(1, X.shape[1] + 1), scores, "o-")
plt.axvline(range(1, X.shape[1] + 1)[index], linestyle="--", color="k", linewidth=1)
plt.xlabel("最大特征变量数")
plt.ylabel("最优预测准确率")
plt.title("预测准确率随选取的最大特征变量数变化情况")
print(scores)

# 14.3.6  寻求n_estimators最优参数
ScoreAll = []
for i in range(100, 300, 10):
    model = RandomForestClassifier(max_features=2, n_estimators=i, random_state=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    ScoreAll.append([i, score])
ScoreAll = np.array(ScoreAll)
print(ScoreAll)
max_score = np.where(ScoreAll == np.max(ScoreAll[:, 1]))[0][0]  # 找出最高得分对应的索引
print("最优参数以及最高得分:", ScoreAll[max_score])
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正常显示中文
plt.figure(figsize=[20, 5])
plt.xlabel("n_estimators")
plt.ylabel("预测准确率")
plt.title("预测准确率随n_estimators变化情况")
plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
plt.show()
# 进一步寻求n_estimators最优参数
ScoreAll = []
for i in range(190, 210):
    model = RandomForestClassifier(max_features=2, n_estimators=i, random_state=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    ScoreAll.append([i, score])
ScoreAll = np.array(ScoreAll)
print(ScoreAll)
max_score = np.where(ScoreAll == np.max(ScoreAll[:, 1]))[0][0]  # 找出最高得分对应的索引
print("最优参数以及最高得分:", ScoreAll[max_score])
plt.figure(figsize=[20, 5])
plt.xlabel("n_estimators")
plt.ylabel("预测准确率")
plt.title("预测准确率随n_estimators变化情况")
plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
plt.show()

# 14.3.7  随机森林特征变量重要性水平分析
sorted_index = model.feature_importances_.argsort()
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决图表中中文显示问题
plt.barh(range(X_train.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index])
plt.xlabel("特征变量重要性水平")
plt.ylabel("特征变量")
plt.title("随机森林特征变量重要性水平分析")
plt.tight_layout()
# 14.3.8  绘制部分依赖图与个体条件期望图
PartialDependenceDisplay.from_estimator(
    model, X_train, ["workyears", "debtratio"], kind="average"
)  # 绘制部分依赖图简称PDP图
PartialDependenceDisplay.from_estimator(
    model, X_train, ["workyears", "debtratio"], kind="individual"
)  # 绘制个体条件期望图（ICE Plot）
PartialDependenceDisplay.from_estimator(
    model, X_train, ["workyears", "debtratio"], kind="both"
)  # 绘制个体条件期望图（ICE Plot）
# 14.3.9  模型性能评价
prob = model.predict_proba(X_test)
prob[:5]
pred = model.predict(X_test)
pred[:5]
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
cohen_kappa_score(y_test, pred)  # 计算kappa得分
# 14.3.10  绘制ROC曲线
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决图表中中文显示问题
plot_roc_curve(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, "k--", linewidth=1)
plt.title("随机森林分类树算法ROC曲线")  # 将标题设置为''随机森林分类树算法ROC曲线'
# 14.3.11  运用两个特征变量绘制随机森林算法决策边界图
X2 = X.iloc[:, [2, 5]]  # 仅选取workyears、debtratio作为特征变量
model = RandomForestClassifier(n_estimators=300, max_features=1, random_state=1)
model.fit(X2, y)
model.score(X2, y)
plot_decision_regions(np.array(X2), np.array(y), model)
plt.xlabel("debtratio")  # 将x轴设置为'debtratio'
plt.ylabel("workyears")  # 将y轴设置为'workyears'
plt.title("随机森林算法决策边界")  # 将标题设置为'随机森林算法决策边界'

# 14.4  回归问题随机森林算法示例
# 14.4.1  变量设置及数据处理
data = pd.read_csv("数据13.2.csv")
X = data.iloc[:, 1:]  # 设置特征变量
y = data.iloc[:, 0]  # 设置响应变量
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=10
)
# 14.4.2  线性回归、单颗回归决策树算法观察
# 线性回归算法
model = LinearRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
# 单颗回归决策树算法
param_grid = {"ccp_alpha": path.ccp_alphas}
kfold = KFold(n_splits=10, shuffle=True, random_state=10)
model = GridSearchCV(DecisionTreeRegressor(random_state=10), param_grid, cv=kfold)
model.fit(X_train, y_train)
print("最优alpha值：", model.best_params_)
model = model.best_estimator_
print("最优拟合优度：", model.score(X_test, y_test))
# 14.4.3  装袋法回归算法
model = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(random_state=10),
    n_estimators=300,
    max_samples=0.9,
    oob_score=True,
    random_state=0,
)
model.fit(X_train, y_train)
model.score(X_test, y_test)
# 14.4.4  随机森林回归算法
max_features = int(X_train.shape[1] / 3)
max_features
model = RandomForestRegressor(
    n_estimators=300, max_features=max_features, random_state=10
)
model.fit(X_train, y_train)
model.score(X_test, y_test)

# 14.4.5  寻求max_features最优参数
scores = []
for max_features in range(1, X.shape[1] + 1):
    model = RandomForestRegressor(
        max_features=max_features, n_estimators=300, random_state=123
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    scores.append(score)
index = np.argmax(scores)
range(1, X.shape[1] + 1)[index]
plt.plot(range(1, X.shape[1] + 1), scores, "o-")
plt.axvline(range(1, X.shape[1] + 1)[index], linestyle="--", color="k", linewidth=1)
plt.xlabel("最大特征变量数")
plt.ylabel("拟合优度")
plt.title("拟合优度随选取的最大特征变量数变化情况")
print(scores)
# 14.4.6  寻求n_estimators最优参数
ScoreAll = []
for i in range(10, 100, 10):
    model = RandomForestRegressor(max_features=1, n_estimators=i, random_state=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    ScoreAll.append([i, score])
ScoreAll = np.array(ScoreAll)
print(ScoreAll)
max_score = np.where(ScoreAll == np.max(ScoreAll[:, 1]))[0][0]  # 找出最高得分对应的索引
print("最优参数以及最高得分:", ScoreAll[max_score])
plt.figure(figsize=[20, 5])
plt.xlabel("n_estimators")
plt.ylabel("拟合优度")
plt.title("拟合优度随n_estimators变化情况")
plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
plt.show()

# 进一步寻求n_estimators最优参数
ScoreAll = []
for i in range(10, 30):
    model = RandomForestRegressor(max_features=1, n_estimators=i, random_state=10)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    ScoreAll.append([i, score])
ScoreAll = np.array(ScoreAll)
print(ScoreAll)
max_score = np.where(ScoreAll == np.max(ScoreAll[:, 1]))[0][0]  # 找出最高得分对应的索引
print("最优参数以及最高得分:", ScoreAll[max_score])
plt.figure(figsize=[20, 5])
plt.xlabel("n_estimators")
plt.ylabel("拟合优度")
plt.title("拟合优度随n_estimators变化情况")
plt.plot(ScoreAll[:, 0], ScoreAll[:, 1])
plt.show()

# 14.4.7  随机森林特征变量重要性水平分析
sorted_index = model.feature_importances_.argsort()
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决图表中中文显示问题
plt.barh(range(X_train.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index])
plt.xlabel("特征变量重要性水平")
plt.ylabel("特征变量")
plt.title("随机森林特征变量重要性水平分析")
plt.tight_layout()
# 14.4.8  绘制部分依赖图与个体条件期望图
PartialDependenceDisplay.from_estimator(
    model, X_train, ["stop10", "stop1"], kind="average"
)  # 绘制部分依赖图简称PDP图
PartialDependenceDisplay.from_estimator(
    model, X_train, ["stop10", "stop1"], kind="individual"
)  # 绘制个体条件期望图（ICE Plot）
PartialDependenceDisplay.from_estimator(
    model, X_train, ["stop10", "stop1"], kind="both"
)  # 绘制个体条件期望图（ICE Plot）
# 14.4.9  最优模型拟合效果图形展示
pred = model.predict(X_test)  # 对响应变量进行预测
t = np.arange(len(y_test))  # 求得响应变量在测试样本中的个数，以便绘制图形。
plt.plot(t, y_test, "r-", linewidth=2, label="原值")  # 绘制响应变量原值曲线。
plt.plot(t, pred, "g-", linewidth=2, label="预测值")  # 绘制响应变量预测曲线。
plt.legend(loc="upper right")  # 将图例放在图的右上方。
plt.grid()
plt.show()
