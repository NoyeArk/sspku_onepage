#第九章  高维数据惩罚回归算法
# 9.2.1  载入分析所需要的模块和函数
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
#9.2.2  数据读取及观察
data=pd.read_csv('数据9.1.csv')
data.info()
data.isnull().values.any()
data.V1.value_counts()
#9.3  变量设置及数据处理
y = data['V1']#设置响应变量，即V1
X_pre= data.drop(['V1'],axis=1)#设置原始特征变量，即除V1之外的全部变量
scaler = StandardScaler()#调用StandardScaler()函数
X = scaler.fit_transform(X_pre)#对原始特征变量进行标准化处理
np.set_printoptions(suppress=True)#不以科学计数法显示，而是直接显示数字
np.mean(X,axis=0)#观察新生成特征变量的均值
np.std(X,axis=0)#观察新生成特征变量的标准差
#9.4  岭回归算法
#9.4.1  使用默认惩罚系数构建岭回归模型
model = Ridge()#建立岭回归算法模型
model.fit(X, y)#使用fit()方法进行拟合
model.score(X, y) #计算岭回归模型拟合优度（可决系数）
model.intercept_#输出岭回归的常数项
pd.DataFrame(model.coef_, index=X_pre.columns, columns=['Coefficient'])#以数据框形式展现回归系数
y_hat= model.predict(X)
pd.DataFrame(y_hat,y)#将响应变量的拟合值和实际值进行对比
#9.4.2  使用留一交叉验证法寻求最优惩罚系数构建岭回归模型
alphas = np.logspace(-4,4,100)#定义一个参数网络
model = RidgeCV(alphas=alphas)#将惩罚参数设置为参数网络中的取值
model.fit(X, y)#使用fit()方法进行拟合
model.alpha_
model.score(X, y) #计算岭回归模型拟合优度（可决系数）
model = RidgeCV(alphas=np.linspace(50, 150,1000))
model.fit(X, y)#使用fit()方法进行拟合
model.alpha_
model.score(X, y) #计算岭回归模型拟合优度（可决系数）
#9.4.3 使用K折交叉验证法寻求最优惩罚系数构建岭回归模型
alphas=np.linspace(10, 100,1000)#定义一个参数网络
kfold = KFold(n_splits=10, shuffle=True, random_state=1)#定义10折随机分组样本
model = RidgeCV(alphas=alphas, cv=kfold)#将惩罚参数设置为参数网络中的取值
model.fit(X, y)#使用fit()方法进行拟合
model.alpha_
model.score(X, y) #计算岭回归模型拟合优度（可决系数）
model.intercept_#输出岭回归的常数项
pd.DataFrame(model.coef_, index=X_pre.columns, columns=['Coefficient'])#以数据框形式展现回归系数
y_hat= model.predict(X)#得到响应变量的拟合值
pd.DataFrame(y_hat,y)#将响应变量的拟合值和实际值进行对比
#9.4.4 划分训练样本和测试样本下的最优岭回归模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = RidgeCV(alphas=np.linspace(10, 100,1000))#将惩罚参数设置为参数网络中的取值
model.fit(X_train, y_train)#使用fit()方法进行拟合
model.alpha_#获得最优alpha值
model.score(X_test, y_test) #计算岭回归模型拟合优度（可决系数）

#9.5  Lasso回归算法
#9.5.1  使用随机选取惩罚系数构建岭回归模型
model = Lasso(alpha=0.2)#建立Lasso回归算法模型
model.fit(X, y)#使用fit()方法进行拟合
model.score(X, y) #计算Lasso回归模型拟合优度（可决系数）
model.intercept_#输出Lasso回归的常数项
pd.DataFrame(model.coef_, index=X_pre.columns, columns=['Coefficient'])
y_hat= model.predict(X)#得到响应变量的拟合值
pd.DataFrame(y_hat,y)#将响应变量的拟合值和实际值进行对比
#9.5.2  使用留一交叉验证法寻求最优惩罚系数构建Lasso回归模型
alphas=np.linspace(0, 0.3,100)#定义一个参数网络
model = LassoCV(alphas=alphas)#将惩罚参数设置为参数网络中的取值
model.fit(X, y)#使用fit()方法进行拟合
model.alpha_
model.score(X, y) #计算Lasso回归模型拟合优度（可决系数）

#9.5.3 使用K折交叉验证法寻求最优惩罚系数构建Lasso回归模型
alphas=np.linspace(0, 0.3,100)#定义一个参数网络
kfold = KFold(n_splits=10, shuffle=True, random_state=1)#定义10折随机分组样本
model = LassoCV(alphas=alphas, cv=kfold)#将惩罚参数设置为参数网络中的取值
model.fit(X, y)#使用fit()方法进行拟合
model.alpha_
model.score(X, y) #计算Lasso回归模型拟合优度（可决系数）
model.intercept_#输出Lasso回归的常数项
pd.DataFrame(model.coef_, index=X_pre.columns, columns=['Coefficient'])

#9.5.4 划分训练样本和测试样本下的最优Lasso回归模型
model =LassoCV(alphas=np.linspace(0, 0.3,100))#将惩罚参数设置为参数网络中的取值
model.fit(X_train, y_train)#使用fit()方法进行拟合
model.alpha_#获得最优alpha值，运行结果为0.02727272727272727。
model.score(X_test, y_test) #计算岭回归模型拟合优度（可决系数）

# 9.6  弹性网回归算法
#9.6.1  使用随机选取惩罚系数构建弹性网回归模型
model = ElasticNet(alpha=1, l1_ratio=0.1)
model.fit(X, y)#使用fit()方法进行拟合
model.score(X, y) #计算弹性网回归模型拟合优度（可决系数）
model.intercept_#输出弹性网回归的常数项
pd.DataFrame(model.coef_, index=X_pre.columns, columns=['Coefficient'])
y_hat= model.predict(X)#得到响应变量的拟合值
pd.DataFrame(y_hat, y)#将响应变量的拟合值和实际值进行对比
#9.6.2 使用K折交叉验证法寻求最优惩罚系数构建弹性网回归模型
alphas = np.logspace(-3, 0, 100)
kfold = KFold(n_splits=10, shuffle=True, random_state=1)
model = ElasticNetCV(cv=kfold, alphas = alphas, l1_ratio=[0.0001, 0.001, 0.01, 0.1, 0.5, 1])
model.fit(X, y)#使用fit()方法进行拟合
model.alpha_
model.l1_ratio_
model.score(X, y) #计算弹性网回归模型拟合优度（可决系数）
model.intercept_#输出弹性网回归的常数项
pd.DataFrame(model.coef_, index=X_pre.columns, columns=['Coefficient'])
#9.6.3 划分训练样本和测试样本下的最优弹性网回归模型
model = ElasticNetCV(cv=kfold, alphas =np.logspace(-3, 0, 100), l1_ratio=[0.0001, 0.001, 0.01, 0.1, 0.5, 1])
model.fit(X_train, y_train)#使用fit()方法进行拟合
model.alpha_#获得最优alpha值，运行结果为0.1。
model.l1_ratio_#输出正则项L1最优惩罚系数，运行结果为：0.5。
model.score(X_test, y_test) 

