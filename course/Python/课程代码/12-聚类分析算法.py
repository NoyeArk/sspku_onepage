#12.2  数据准备
#12.2.2载入分析所需要的库和模块
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

#12.2.3变量设置及数据处理
data=pd.read_csv('数据12.1.csv')
X = data.iloc[:,[2,5,6,7]]#设置分析所需要的特征变量
X.info()
len(X.columns) 
X.columns 
X.shape
X.dtypes
X.isnull().values.any() 
X.isnull().sum() 
X.head(10)
scaler = StandardScaler()
scaler.fit(X)
X_s = scaler.transform(X)
X_s = pd.DataFrame(X_s, columns=X.columns)
#12.2.4相关性分析
print(X_s.corr(method='pearson'))  
plt.subplot(1,1,1)
sns.heatmap(X_s.corr(), annot=True)

#12.3  划分聚类分析算法
#12.3.1使用K均值聚类分析方法对样本示例进行聚类(K=2)
model = KMeans(n_clusters=2, random_state=2)
model.fit(X_s)
model.labels_
pd.DataFrame(model.labels_.T, index=data.V1,columns=['聚类'])
model.cluster_centers_
model.inertia_

#12.3.2使用K均值聚类分析方法对样本示例进行聚类(K=3)
model = KMeans(n_clusters=3, random_state=2)
model.fit(X_s)
model.labels_
pd.DataFrame(model.labels_.T, index=data.V1,columns=['聚类'])
np.set_printoptions(suppress=True)#不以科学计数法显示，而是直接显示数字
model.cluster_centers_
model.inertia_

#12.3.3使用K均值聚类分析方法对样本示例进行聚类(K=4)
model = KMeans(n_clusters=4, random_state=3)
model.fit(X_s)
model.labels_
pd.DataFrame(model.labels_.T, index=data.V1,columns=['聚类'])
model.cluster_centers_
model.inertia_

#12.4层次聚类方法
#12.4.1最短联结法聚类分析
linkage_matrix = linkage(X_s, 'single')
plt.rcParams['font.sans-serif'] = ['SimHei']#解决图表中中文显示问题
dendrogram(linkage_matrix)
plt.title('最短联结法聚类分析树状图')

model = AgglomerativeClustering(n_clusters=3, linkage='single')
model.fit(X_s)
model.labels_
pd.DataFrame(model.labels_.T, index=data.V1,columns=['聚类'])

#12.4.2最长联结法聚类分析
linkage_matrix = linkage(X_s , 'complete')
dendrogram(linkage_matrix)
plt.title('最长联结法聚类分析树状图')

model = AgglomerativeClustering(n_clusters=3, linkage='complete')
model.fit(X_s )
model.labels_
pd.DataFrame(model.labels_.T, index=data.V1,columns=['聚类'])

#12.4.3平均联结法聚类分析
linkage_matrix = linkage(X_s, 'average')
dendrogram(linkage_matrix)
plt.title('平均联结法聚类分析树状图')

model = (AgglomerativeClustering(n_clusters=3, linkage='average'))
model.fit(X_s)
model.labels_
pd.DataFrame(model.labels_.T, index=data.V1,columns=['聚类'])

model = AgglomerativeClustering(n_clusters=3,linkage='average')
model.fit(X_s)
model.labels_
pd.DataFrame(model.labels_.T, index=data.V1,columns=['聚类'])

model = AgglomerativeClustering(n_clusters=3,linkage='average')
model.fit(X_s)
model.labels_
pd.DataFrame(model.labels_.T, index=data.V1,columns=['聚类'])

#12.4.4ward联结法聚类分析
linkage_matrix = linkage(X_s, 'ward')
dendrogram(linkage_matrix)
plt.title('ward联结法聚类分析树状图')

model = AgglomerativeClustering(n_clusters=3,linkage='ward')
model.fit(X_s)
model.labels_
pd.DataFrame(model.labels_.T, index=data.V1,columns=['聚类'])


#12.4.5重心联结法聚类分析
linkage_matrix = linkage(X_s, 'centroid')
dendrogram(linkage_matrix)
plt.title('重心联结法聚类分析树状图')

labels = fcluster(linkage_matrix, t=3, criterion='maxclust')
labels
pd.DataFrame(labels.T, index=data.V1,columns=['聚类'])


