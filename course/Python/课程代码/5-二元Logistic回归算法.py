#  二元Logistic回归算法
#5.2  数据准备
# 5.2.1  载入分析所需要的模块和函数
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import plot_roc_curve

# 2 数据读取及观察
data=pd.read_csv('数据5.1.csv')
data.info()
len(data.columns) 
data.columns 
data.shape
data.dtypes
data.isnull().values.any() 
data.isnull().sum() 
data.head()

#3描述性分析
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data.describe()
data.groupby('V1').describe().unstack()#按照V1变量的取值分组对其他变量开展描述性分析
pd.crosstab(data.V3, data.V1)
pd.crosstab(data.V3, data.V1, normalize='index')

# 4 数据处理
# 4.1  区分分类特征和连续特征并进行处理
def data_encoding(data):
    data = data[["V1",'V2',"V3","V4","V5","V6","V7","V8","V9"]]
    Discretefeature=["V3"]
    Continuousfeature=['V2',"V4","V5","V6","V7","V8","V9"]
    df = pd.get_dummies(data,columns=Discretefeature)
    df[Continuousfeature]=(df[Continuousfeature]-df[Continuousfeature].mean())/(df[Continuousfeature].std())
    df["V1"]=data[["V1"]]
    return df
data=data_encoding(data)

# 4.2  将样本示例全集分割为训练样本和测试样本
X = data.drop(['V1','V3_5'],axis=1)#设置特征变量，即除V1、V3_5之外的全部变量
X['intercept'] = [1]*X.shape[0]#为X增加1列，设置模型中的常数项。
y = data['V1']#设置响应变量，即V1
print(data["V1"].value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=100)
X_train.head()
y_train.head()

#5  建立二元Logistic回归算法模型
#5.1  使用statsmodels建立二元Logistic回归算法模型
model = sm.Logit(y_train, X_train)
results = model.fit()
results.params
results.summary()
np.exp(results.params)   # 以几率比（Odds ratio）的形式输出二元Logistic回归模型的系数值
margeff = results.get_margeff()
margeff.summary()

#计算训练误差

table = results.pred_table() 
table

Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Error_rate = 1 - Accuracy
Error_rate

precision = table[1, 1] / (table[0, 1] + table[1, 1])
precision

recall = table[1, 1] / (table[1, 0] + table[1, 1])
recall

#计算测试误差

prob = results.predict(X_test)
pred = (prob >= 0.5)
table = pd.crosstab(y_test, pred, colnames=['Predicted'])
table

table = np.array(table)   

Accuracy = (table[0, 0] + table[1, 1]) / np.sum(table)
Accuracy

Error_rate = 1 - Accuracy
Error_rate

precision = table[1, 1] / (table[0, 1] + table[1, 1])
precision

recall = table[1, 1] / (table[1, 0] + table[1, 1])
recall

#5.2  使用sklearn建立二元Logistic回归算法模型
model =  LogisticRegression(C=1e10, fit_intercept=True)
model.fit(X_train, y_train)
print("训练样本预测准确率: {:.3f}".format(model.score(X_train, y_train)))#训练样本预测对的个数 / 总个数
print("测试样本预测准确率: {:.3f}".format(model.score(X_test, y_test)))#测试样本预测对的个数 / 总个数
model.coef_

predict_target=model.predict(X_test)
predict_target
predict_target_prob=model.predict_proba(X_test)  
predict_target_prob
predict_target_prob_lr=predict_target_prob[:,1]
df=pd.DataFrame({'prob':predict_target_prob_lr,'target':predict_target,'labels':list(y_test)})
df.head()
    
print('预测正确总数：')
print(sum(predict_target==y_test))
 
print('训练样本：')
predict_Target=model.predict(X_train)
print(metrics.classification_report(y_train,predict_Target))

print(metrics.confusion_matrix(y_train, predict_Target))   
 
print('测试样本：')
print(metrics.classification_report(y_test,predict_target))

print(metrics.confusion_matrix(y_test, predict_target))

#5.3  特征变量重要性水平分析
lr1=[i for item in model.coef_ for i in item]
lr1=np.array(lr1)
lr1
feature=list(X.columns)
feature
dic={}
for i in range(len(feature)):
  dic.update({feature[i]:lr1[i]})
dic
df=pd.DataFrame.from_dict(dic,orient='index',columns=['权重'])
df
df=df.reset_index().rename(columns={'index':'特征'})
df
df=df.sort_values(by='权重',ascending=False)
df
data_hight=df['权重'].values.tolist()
data_hight
data_x=df['特征'].values.tolist()
data_x

font = {'family': 'Times New Roman', 'size': 7, }
sns.set(font_scale=1.2)
plt.rc('font',family='Times New Roman')
plt.figure(figsize=(6,6))
plt.barh(range(len(data_x)), data_hight, color='#6699CC')
plt.yticks(range(len(data_x)),data_x,fontsize=12)
plt.tick_params(labelsize=12)
plt.xlabel('Feature importance',fontsize=14)
plt.title("LR feature importance analysis",fontsize = 14)
plt.show()

#5.5.4  绘制ROC曲线，计算AUC值
plot_roc_curve(model, X_test, y_test)
x = np.linspace(0, 1, 100)
plt.plot(x, x, 'k--', linewidth=1)
plt.title('ROC Curve (Test Set)')

#5.5.5  计算科恩kappa得分
cohen_kappa_score(y_test, pred)      
