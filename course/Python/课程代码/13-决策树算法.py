# 决策树算法
# 13.2.2  载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import cohen_kappa_score
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LinearRegression

# 13.3 分类问题决策树算法示例
# 13.3.1  变量设置及数据处理
data = pd.read_csv("数据13.1.csv")
data.info()
data.isnull().values.any()
data.credit.value_counts()
data.credit.value_counts(normalize=True)
# 将样本示例全集分割为训练样本和测试样本
X = data.iloc[:, 1:]  # 设置特征变量
y = data.iloc[:, 0]  # 设置响应变量
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=10
)
# 13.3.2  未考虑成本-复杂度剪枝的决策树分类算法模型
# 一、当分裂准则为信息熵时
model = DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=10)
model.fit(X_train, y_train)
model.score(X_test, y_test)
plot_tree(
    model,
    feature_names=X.columns,
    node_ids=True,
    impurity=True,
    proportion=True,
    rounded=True,
    precision=3,
)
plt.savefig("out1.pdf")  # 有效解决显示不清晰的问题
# 模型性能评价
prob = model.predict_proba(X_test)
prob[:5]
pred = model.predict(X_test)
pred[:5]
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
cohen_kappa_score(y_test, pred)  # 计算kappa得分

# 二、当分裂准则为基尼指数时
model = DecisionTreeClassifier(
    criterion="gini", max_depth=2, random_state=10
)  # 采用基尼系数作为分裂准则，指定决策树的最大深度为2，设定随机数种子为100
model.fit(X_train, y_train)
model.score(X_test, y_test)
plot_tree(
    model,
    feature_names=X.columns,
    node_ids=True,
    impurity=True,
    proportion=True,
    rounded=True,
    precision=3,
)
plt.savefig("out2.pdf")


# 13.3.3  考虑成本-复杂度剪枝的决策树分类算法模型
model = DecisionTreeClassifier(random_state=10)
path = model.cost_complexity_pruning_path(X_train, y_train)
print("模型复杂度参数：", max(path.ccp_alphas))  # 输出最大的模型复杂度参数
print("模型不纯度：", max(path.impurities))  # 输出最大的模型不纯度
# 13.3.4  绘制图形观察"叶节点总不纯度随alpha值变化情况"
fig, ax = plt.subplots()
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决图表中中文显示问题
ax.plot(path.ccp_alphas, path.impurities, marker="o", drawstyle="steps-post")
ax.set_xlabel("有效的alpha（成本-复杂度剪枝参数值)")
ax.set_ylabel("叶节点总不纯度")
ax.set_title("叶节点总不纯度随alpha值变化情况")

# 13.3.5  绘制图形观察“节点数和树的深度随alpha值变化情况”
models = []
for ccp_alpha in path.ccp_alphas:
    model = DecisionTreeClassifier(random_state=10, ccp_alpha=ccp_alpha)
    model.fit(X_train, y_train)
    models.append(model)
print(
    "最后一棵决策时的节点数为: {} ;其alpha值为: {}".format(
        models[-1].tree_.node_count, path.ccp_alphas[-1]
    )
)  # 输出最path.ccp_alphas中最后一个值，即修剪整棵树的alpha值，只有一个节点
node_counts = [model.tree_.node_count for model in models]
depth = [model.tree_.max_depth for model in models]
fig, ax = plt.subplots(2, 1)
ax[0].plot(path.ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("节点数nodes")
ax[0].set_title("节点数nodes随alpha值变化情况")
ax[1].plot(path.ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("决策树的深度depth")
ax[1].set_title("决策树的深度随alpha值变化情况")
fig.tight_layout()

# 13.3.6  绘制图形观察“训练样本和测试样本的预测准确率随alpha值变化情况”
train_scores = [model.score(X_train, y_train) for model in models]
test_scores = [model.score(X_test, y_test) for model in models]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("预测准确率")
ax.set_title("训练样本和测试样本的预测准确率随alpha值变化情况")
ax.plot(
    path.ccp_alphas, train_scores, marker="o", label="训练样本", drawstyle="steps-post"
)
ax.plot(
    path.ccp_alphas, test_scores, marker="o", label="测试样本", drawstyle="steps-post"
)
ax.legend()
plt.show()

# 13.3.7  通过10折交叉验证法寻求最优alpha值
param_grid = {"ccp_alpha": path.ccp_alphas}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
model = GridSearchCV(DecisionTreeClassifier(random_state=10), param_grid, cv=kfold)
model.fit(X_train, y_train)
print("最优alpha值：", model.best_params_)
model = model.best_estimator_
print("最优预测准确率：", model.score(X_test, y_test))
plot_tree(
    model,
    feature_names=X.columns,
    node_ids=True,
    impurity=True,
    proportion=True,
    rounded=True,
    precision=3,
)
plt.savefig("out3.pdf")

# 13.3.8  决策树特征变量重要性水平分析
sorted_index = model.feature_importances_.argsort()
plt.barh(range(X_train.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index])
plt.xlabel("特征变量重要性水平")
plt.ylabel("特征变量")
plt.title("决策树特征变量重要性水平分析")
plt.tight_layout()

# 13.3.9  绘制ROC曲线
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决图表中中文显示问题
plot_roc_curve(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, "k--", linewidth=1)
plt.title("决策树分类树算法ROC曲线")  # 将标题设置为'决策树分类树算法ROC曲线'

# 13.3.10  运用两个特征变量绘制决策树算法决策边界图
X2 = X.iloc[:, [2, 5]]  # 仅选取workyears、debtratio作为特征变量
model = DecisionTreeClassifier(random_state=100)
path = model.cost_complexity_pruning_path(X2, y)
param_grid = {"ccp_alpha": path.ccp_alphas}
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
model = GridSearchCV(DecisionTreeClassifier(random_state=100), param_grid, cv=kfold)
model.fit(X2, y)  # 使用fit方法进行拟合
model.score(X2, y)  # 计算模型预测准确率
plot_decision_regions(np.array(X2), np.array(y), model)
plt.xlabel("debtratio")  # 将x轴设置为'debtratio'
plt.ylabel("workyears")  # 将y轴设置为'workyears'
plt.title("决策树算法决策边界")  # 将标题设置为'决策树算法决策边界'


# 13.4  回归问题决策树算法示例
# 13.4.1  变量设置及数据处理
data = pd.read_csv("数据13.2.csv")
data.info()
data.isnull().values.any()
# 将样本示例全集分割为训练样本和测试样本
X = data.iloc[:, 1:]  # 设置特征变量
y = data.iloc[:, 0]  # 设置响应变量
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=10
)

# 13.4.2  未考虑成本-复杂度剪枝的决策树回归算法模型
model = DecisionTreeRegressor(max_depth=2, random_state=10)
model.fit(X_train, y_train)
print("拟合优度：", model.score(X_test, y_test))
plot_tree(model, feature_names=X.columns, node_ids=True, rounded=True, precision=3)
plt.savefig("out4.pdf")  # 有效解决显示不清晰的问题
print("文本格式的决策树：", export_text(model, feature_names=list(X.columns)))

# 13.4.3  考虑成本-复杂度剪枝的决策树回归算法模型
model = DecisionTreeRegressor(random_state=10)
path = model.cost_complexity_pruning_path(X_train, y_train)
print("模型复杂度参数：", max(path.ccp_alphas))  # 输出最大的模型复杂度参数
print("模型总均方误差：", max(path.impurities))  # 输出最大的模型总均方误差
# 13.4.4  绘制图形观察"叶节点总均方误差随alpha值变化情况"
fig, ax = plt.subplots()
ax.plot(path.ccp_alphas, path.impurities, marker="o", drawstyle="steps-post")
ax.set_xlabel("有效的alpha（成本-复杂度剪枝参数值)")
ax.set_ylabel("叶节点总均方误差")
ax.set_title("叶节点总均方误差alpha值变化情况")

# 13.4.5  绘制图形观察“节点数和树的深度随alpha值变化情况”
models = []
for ccp_alpha in path.ccp_alphas:
    model = DecisionTreeRegressor(random_state=10, ccp_alpha=ccp_alpha)
    model.fit(X_train, y_train)
    models.append(model)
print(
    "最后一棵决策时的节点数为: {} ;其alpha值为: {}".format(
        models[-1].tree_.node_count, path.ccp_alphas[-1]
    )
)  # 输出最path.ccp_alphas中最后一个值，即修剪整棵树的alpha值，只有一个节点
node_counts = [model.tree_.node_count for model in models]
depth = [model.tree_.max_depth for model in models]
fig, ax = plt.subplots(2, 1)
ax[0].plot(path.ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("节点数nodes")
ax[0].set_title("节点数nodes随alpha值变化情况")
ax[1].plot(path.ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("决策树的深度depth")
ax[1].set_title("决策树的深度随alpha值变化情况")
fig.tight_layout()

# 13.4.6  绘制图形观察“训练样本和测试样本的拟合优度随alpha值变化情况”
train_scores = [model.score(X_train, y_train) for model in models]
test_scores = [model.score(X_test, y_test) for model in models]
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("拟合优度")
ax.set_title("训练样本和测试样本的拟合优度随alpha值变化情况")
ax.plot(
    path.ccp_alphas, train_scores, marker="o", label="训练样本", drawstyle="steps-post"
)
ax.plot(
    path.ccp_alphas, test_scores, marker="o", label="测试样本", drawstyle="steps-post"
)
ax.legend()
plt.show()

# 13.4.7 通过10折交叉验证法寻求最优alpha值并开展特征变量重要性水平分析
# 通过10折交叉验证法寻求最优alpha值
param_grid = {"ccp_alpha": path.ccp_alphas}
kfold = KFold(n_splits=10, shuffle=True, random_state=10)
model = GridSearchCV(DecisionTreeRegressor(random_state=10), param_grid, cv=kfold)
model.fit(X_train, y_train)
print("最优alpha值：", model.best_params_)
model = model.best_estimator_
print("最优拟合优度：", model.score(X_test, y_test))
print("决策树深度：", model.get_depth())
print("叶节点数目：", model.get_n_leaves())
print("每个变量的重要性：", model.feature_importances_)

# 决策树特征变量重要性水平分析
sorted_index = model.feature_importances_.argsort()
plt.barh(range(X_train.shape[1]), model.feature_importances_[sorted_index])
plt.yticks(np.arange(X_train.shape[1]), X_train.columns[sorted_index])
plt.xlabel("特征变量重要性水平")
plt.ylabel("特征变量")
plt.title("决策树特征变量重要性水平分析")
plt.tight_layout()

plot_tree(
    model,
    feature_names=X.columns,
    node_ids=True,
    impurity=True,
    proportion=True,
    rounded=True,
    precision=3,
)
plt.savefig("out5.pdf")  # 有效解决显示不清晰的问题
# 13.4.8  最优模型拟合效果图形展示
pred = model.predict(X_test)  # 对响应变量进行预测
t = np.arange(len(y_test))  # 求得响应变量在测试样本中的个数，以便绘制图形。
plt.plot(t, y_test, "r-", linewidth=2, label="原值")  # 绘制响应变量原值曲线。
plt.plot(t, pred, "g-", linewidth=2, label="预测值")  # 绘制响应变量预测曲线。
plt.legend(loc="upper right")  # 将图例放在图的右上方。
plt.grid()
plt.show()

# 13.4.9  构建线性回归算法模型进行对比
model = LinearRegression().fit(X_train, y_train)
model.score(X_test, y_test)
