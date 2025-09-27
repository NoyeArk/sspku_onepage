# 线性回归算法
#1  载入分析所需要的模块和函数
import pandas as pd#载入pandas模块，并简称为pd
import numpy as np#载入numpy模块，并简称为np
import matplotlib.pyplot as plt#载入matplotlib.pyplot模块，并简称为plt
import seaborn as sns#载入seaborn模块，并简称为sns
from scipy import stats#载入stats模块
from scipy.stats import probplot#载入probplot模块
import statsmodels.formula.api as smf#载入statsmodels.formula.api模块，并简称为smf
from sklearn.linear_model import LinearRegression#载入LinearRegression模块
from sklearn.model_selection import train_test_split#载入train_test_split模块
from sklearn.metrics import mean_squared_error, r2_score#载入mean_squared_error, r2_score模块
#2  数据读取及观察
data=pd.read_csv('数据4.1.csv')
data.info()
len(data.columns) # 本命令的含义是列出数据集中变量的数量
data.columns # 本命令的含义是列出数据集中的变量
data.shape # 列出数据集的形状
data.dtypes # 观察数据集中各个变量的数据类型
data.isnull().values.any() # 检查数据集是否有缺失值
data.isnull().sum() # 逐个变量检查数据集是否有缺失值

#3 描述性分析
data.describe()#对数据集进行描述性分析
data.describe().round(2)#只保留两位小数
data.describe().round(2).T#只保留两位小数并转置
data.mean()#对数据集中的变量求均值
data.var()#对数据集中的变量求方差
data.std()#对数据集中的变量求标准差
data.cov()#对数据集中的变量求协方差矩阵

#4  图形绘制
#4.1 直方图
plt.figure(figsize=(20,10))#figsize用来设置图形的大小，figsize = (a, b)，其中a为图形的宽，b为图形的高，单位为英寸。本例中图形的宽为20英寸, 高为10英寸。
plt.subplot(1,2,1)#本代码的含义是指定作图位置。可以把figure理解成画布，subplot就是将figure中的图像划分为几块，每块当中显示各自的图像，有利于进行比较。一般使用格式：subplot(m,n,p)，m为行数即在同一画面创建m行个图形位置，n为列数即在同一画面创建n列个图形位置，本例中把绘图窗口分成1行2列2块区域，然后在每个区域分别作图，p为位数即在同一画面的m行，n列的图形位置，p=1表示从左到右从上到下的第一个位置。
plt.hist(data['invest'], density=False)#绘制invest变量的直方图，参数density为True和False，分别代表是否进行归一化处理。
plt.title("Histogram of 'invest'")#将invest变量的直方图的标题设定为 Histogram of invest。
plt.subplot(1,2,2)#在figure画布从左到右从上到下的第二个位置作图
plt.hist(data['profit'], density=False)#绘制profit变量的直方图，不进行归一化处理。
plt.title("Histogram of 'profit'")#将profit变量的直方图的标题设定为 Histogram of profit。

# 使用sns.histplot绘制直方图
sns.displot(data['invest'],bins=10,kde=True)
plt.title("Histogram of 'invest'")
sns.displot(data['profit'],bins=10,kde=True)
plt.title("Histogram of 'profit'")

#4.2  密度图
plt.subplot(1,2,1)#指定作图位置
sns.kdeplot(data['invest'],shade=True)#绘制invest变量的密度图，shade=True表示密度曲线下方的面积用阴影填充
plt.title("Density distribution of 'invest'")#将invest变量的密度图的标题设定为Density distribution of 'invest'
plt.subplot(1,2,2)#指定作图位置
sns.kdeplot(data['profit'],shade=True)#绘制profit变量的密度图，显示核密度曲线，shade=True表示密度曲线下方的面积用阴影填充
plt.title("Density distribution of 'profit'")#将invest变量的密度图的标题设定为Density distribution of ''profit''

#4.3 箱线图
plt.figure(figsize=(9,6))#figsize用来设置图形的大小
plt.subplot(1,2,1)#指定作图位置
plt.boxplot(data['invest'])#绘制invest变量的箱图
plt.title("Boxlpot of 'invest'")#标题设定为Boxlpot of 'invest'
plt.subplot(1,2,2)#指定作图位置
plt.boxplot(data['profit'])#绘制profit变量的箱图
plt.title("Boxlpot of 'profit'")#标题设定为Boxlpot of 'profit'。

#4.4.4  小提琴图
plt.subplot(1,2,1)#指定作图位置
sns.violinplot(data['invest'])#绘制invest变量的小提琴图
plt.title("Violin plot of 'invest'")#标题设定为Violin plot of 'invest'
plt.subplot(1,2,2)#指定作图位置
sns.violinplot(data['profit'])#绘制profit变量的小提琴图
plt.title("Violin plot of 'profit'")#标题设定为Violin plot of 'profit'

#4.5 正态 QQ 图
plt.figure(figsize=(12,6))#figsize用来设置图形的大小
plt.subplot(1,2,1)#指定作图位置
probplot(data['invest'], plot=plt)#绘制invest变量的正态 QQ 图
plt.title("Q-Q plot of 'invest'")#标题设定为Q-Q plot of 'invest'
plt.subplot(1,2,2)#指定作图位置
probplot(data['profit'], plot=plt)#绘制profit变量的正态 QQ 图
plt.title("Q-Q plot of 'profit'")#标题设定为Q-Q plot of 'profit'

# 4.6 散点图和点线图
plt.figure(figsize=(12,6))#设定图形的宽为12英寸，图形的高为6英寸
plt.subplot(1,3,1)#指定作图位置。在同一画面创建1行3列个图形位置，首先在从左到右的第一个位置作图
sns.scatterplot(data=data, x="invest", y="profit", hue="invest", alpha=0.6)#绘制invest和profit的散点图，使用数据集为data，x横轴为invest，y纵轴为profit，参数hue的作用就是在图像中将输出的散点图按照hue指定的变量（invest）的颜色种类进行区分,alpha为散点的透明度，取值为0到1
plt.title("Scatter plot")#将散点图的标题设定为Scatter plot
plt.subplot(1,3,2)#指定作图位置
sns.lineplot(data=data, x="invest", y="profit")#绘制invest和profit的线图
plt.title("Line plot of invest, profit")#将标题设定为Line plot of invest, profit
plt.subplot(1,3,3)#指定作图位置
sns.lineplot(data=data)#绘制全部变量的线图
plt.title('Line Plot')#将标题设定为Line Plot

#4.7  热力图
plt.figure(figsize=(10, 10))#设置图形大小
plt.subplot(1, 2, 1)#指定作图位置
sns.heatmap(data=data, cmap="YlGnBu", annot = True)#基于data数据绘制热力图，cmap="YlGnBu"用来设置热力图的颜色色系，annot=True表示在热力图每个方格写入数据。
plt.title("Heatmap using seaborn")#指定作图标题
plt.subplot(1, 2, 2)#指定作图位置
plt.imshow(data, cmap ="YlGnBu")#实现热图绘制
plt.title("Heatmap using matplotlib")#指定作图标题

#4.8 回归拟合图
sns.regplot( x="invest", y="profit",data=data )#以"invest"为特征变量，"profit"为响应变量，绘制回归拟合图

#4.9  联合分布图
sns.jointplot(x = "invest", y = "profit", kind = "reg", data = data)#基于数据data绘制联合分布图，x轴为invest，y轴为profit，绘图类型为reg
plt.title("Joint plot using sns")#为图表设置标题
# kind参数可以是hex, kde, scatter, reg, hist。当kind='reg'时，它显示最佳拟合线。

#5  正态性检验

#5.1  Shapiro-Wilk test检验
Ho = '数据服从正态分布'#定义原假设
Ha = '数据不服从正态分布'#定义备择假设
alpha = 0.05#定义显著性P值
def normality_check(data):
    for columnName, columnData in data.items():
        print("Shapiro test for {columnName}".format(columnName=columnName))
        res = stats.shapiro(columnData)
        pValue = round(res[1], 2)
        if pValue > alpha:
            print("pvalue = {pValue} > {alpha}. 不能拒绝原假设. {Ho}".format(pValue=pValue, alpha=alpha, Ho=Ho))
        else:
            print("pvalue = {pValue} <= {alpha}. 拒绝原假设. {Ha}".format(pValue=pValue, alpha=alpha, Ha=Ha))
normality_check(data)

# 5.2 使用kstest检验数据是否服从正态分布
Ho = '数据服从正态分布'#定义原假设
Ha = '数据不服从正态分布'#定义备择假设
alpha = 0.05#定义显著性P值
def normality_check(data):
    for columnName, columnData in data.iteritems():
        print("kstest for {columnName}".format(columnName=columnName))
        res = stats.kstest(columnData,'norm')
        pValue = round(res[1], 2)
        if pValue > alpha:
            print("pvalue = {pValue} > {alpha}. 不能拒绝原假设. {Ho}".format(pValue=pValue, alpha=alpha, Ho=Ho))
        else:
            print("pvalue = {pValue} <= {alpha}. 拒绝原假设. {Ha}".format(pValue=pValue, alpha=alpha, Ha=Ha))
normality_check(data)

#6 相关性分析
print(data.corr(method='pearson')) #输出变量之间的皮尔逊相关系数矩阵
plt.subplot(1,1,1)
sns.heatmap(data.corr(), annot=True)# 绘制相关矩阵的热图
data1=pd.read_csv('C:/Users/Administrator/.spyder-py3/数据4.2.csv')
print(data1.corr(method='spearman')) #输出变量之间的斯皮尔曼等级相关系数矩阵
print(data1.corr(method='kendall')) #输出变量之间的肯德尔等级相关系数矩阵

#7  使用 smf 进行线性回归   
#7.1  使用 smf 进行线性回归
X = data.iloc[:, 2:6]#将数据集中的第3列至第5列作为自变量
y = data.iloc[:, 1:2] #将数据集中的第2列作为因变量 
model = smf.ols('y~X', data=data).fit()#使用线性回归模型，并进行训练
print(model.summary())#输出估计模型摘要
y_pred = model.predict(X)#将y_pred设定为模型因变量的预测值
y_lst = y.iloc[:, -1:].values.tolist()#将y_lst设定为模型因变量的实际值，并转换为列表形式
plt.scatter( y_lst, y_pred, color='blue')  # 绘制因变量实际值和拟合值的散点图
plt.plot( y_lst, y_pred, color='Red', linewidth=2 )  # 绘制因变量实际值和拟合值的的线图
plt.title('Simple Linear Regression')#设定图片标题为Simple Linear Regression
plt.xlabel('y')#设定x轴为y
plt.ylabel('y_pred')#设定y轴为y_pred
#7.2  多重共线性检验 
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
#7.3  解决多重共线性问题      
X = data.iloc[:, 3:6]#将数据集中的第4列至第5列作为自变量
y = data.iloc[:, 1:2] #将数据集中的第2列作为因变量 
model = smf.ols('y~X', data=data).fit()#使用线性回归模型，并进行训练
print(model.summary())#输出估计模型摘要
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
#4.7.4  绘制拟合回归平面
model = smf.ols('profit ~ labor + rd', data=data)
results = model.fit()
results.params
import matplotlib.pyplot as plt
xx = np.linspace(data.labor.min(), data.labor.max(), 100)
yy = np.linspace(data.rd.min(), data.rd.max(), 100)
xx.shape, yy.shape
XX, YY = np.meshgrid(xx,yy)
XX.shape, YY.shape
ZZ = results.params[0] + XX * results.params[1] +  YY * results.params[2] 
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(data.labor, data.rd, data.profit,c='r')
ax.plot_surface(XX, YY, ZZ, rstride=10, cstride=10, alpha=0.2, cmap='viridis')
ax.set_xlabel('labor')
ax.set_ylabel('rd')
ax.set_zlabel('profit')

#8  使用 sklearn 进行线性回归
#8.1  使用验证集法进行模型拟合
X = data.iloc[:, 3:6]#将数据集中的第4列至第5列作为自变量
y = data.iloc[:, 1:2] #将数据集中的第2列作为因变量 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)#将样本示例全集划分为训练样本和测试样本，测试样本占比为30%。
X_train.shape, X_test.shape, y_train.shape, y_test.shape#观察四个数据的形状
model = LinearRegression()#使用线性回归模型
model.fit(X_train, y_train)#基于训练样本拟合模型
model.coef_#计算上步估计得到的回归系数值
model.score(X_test, y_test)#观察模型在测试集中的拟合优度（可决系数）
pred = model.predict(X_test)#计算响应变量基于测试集的预测结果
pred.shape#观察数据形状
mean_squared_error(y_test, pred)#计算测试集的均方误差
r2_score(y_test, pred)#计算测试集的可决系数

#8.2  更换随机数种子，使用验证集法进行模型拟合
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)#更换随机数种子
model = LinearRegression().fit(X_train, y_train)#基于训练样本拟合线性回归模型
pred = model.predict(X_test)#计算响应变量基于测试集的预测结果
mean_squared_error(y_test, pred)#计算测试集的均方误差
r2_score(y_test, pred)#计算测试集的可决系数

# 8.3  使用10折交叉验证法进行模型拟合
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RepeatedKFold
X = data.iloc[:, 3:6]#将数据集中的第4列至第5列作为自变量
y = data.iloc[:, 1:2] #将数据集中的第2列作为因变量 
model = LinearRegression()#使用线性回归模型
kfold = KFold(n_splits=10,shuffle=True, random_state=1)#将样本示例全集分为10折
scores = cross_val_score(model, X, y, cv=kfold)#计算每一折的可决系数
scores#显示每一折的可决系数
scores.mean()#计算各折样本可决系数的均值
scores.std()#计算各折样本可决系数的标准差
scores_mse = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')#得到每个子样本的均方误差
scores_mse#显示各折样本的均方误差
scores_mse.mean()#计算各折样本均方误差的均值
# 更换随机数种子，并与上步得到结果进行对比，观察均方误差MSE大小
kfold = KFold(n_splits=10, shuffle=True, random_state=100)
scores_mse = -cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')#得到每个子样本的均方误差
scores_mse.mean()#计算各折样本均方误差的均值

#8.4  使用10折重复10次交叉验证法进行模型拟合
rkfold = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
scores_mse = -cross_val_score(model, X, y, cv=rkfold, scoring='neg_mean_squared_error')#得到每个子样本的均方误差
scores_mse.shape
scores_mse.mean()
# 绘制各子样本均方误差的直方图
sns.distplot(pd.DataFrame(scores_mse))
plt.xlabel('MSE')
plt.title('10-fold CV Repeated 10 Times')  

# 8.5  使用留一交叉验证法进行模型拟合
loo = LeaveOneOut()
scores_mse = -cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
scores_mse.mean()  
