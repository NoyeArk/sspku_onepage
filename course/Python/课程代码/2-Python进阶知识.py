     

# python numpy模块数组（array）
#数组的创建
import numpy as np#导入numpy模块并简称为np
array0= np.arange(10)#使用arange()函数定义一个一维数组array0
array0#查看数组array0
type(array0)#观察数组array0的类型
array0.ndim#观察数组array0的维度
array0.shape#观察数组array0的形状

np.zeros(9)#创建元素值全部为0的一维数组，元素个数为9
np.zeros((3, 3))#创建二维数组，3×3零矩阵
np.ones((3, 3))#创建二维数组，3×3矩阵，元素值全部为1
np.ones((3, 3, 3))#创建三维数组，形状为3×3×3，元素值全部为1

list1 = [0,1,2,3,4,5,6,7,8,9]#创建列表list1
array1 = np.array(list1)#将列表list1转化为数组形式，得到array1
array1#查看数组array1
array1 = array1.reshape(5, 2)#将一维数组array1转化为二维数组
print(array1)#查看更新后的数组array1
print(array1.ndim)#观察数组array1的维度
print(array1.shape)#观察数组array1的形状

list2 = [[0,1,2,3,4], [5,6,7,8,9]]#创建列表list2
array2= np.array(list2)#将列表list2转化为数组形式，得到array2
print(array2)#查看数组array2
print(array2.ndim)#观察数组array2的维度
print(array2.shape)#观察数组array2的形状
array2 = array2.reshape(5, 2)#将数组array2的形状改变成(5, 2)
'''
array2#查看改变形状后的数组array2
#数组的计算
list1 = [0,1,2,3,4,5,6,7,8,9]#创建列表list1
array1 = np.array(list1)#将列表list1转化为数组形式，得到array1
array1*3+1#将数据array1中的每个元素的值都乘以3再加1
np.sqrt(array1)#将数据array1中的每个元素的值都开平方
np.exp(array1)#将数据array1中的每个元素的值都进行指数运算
np.set_printoptions(suppress=True)#不以科学计数法显示，而是直接显示数字
np.exp(array1)#将数据array1中的每个元素的值都进行指数运算
type(np.exp)#观察函数np.exp（）的类型
array1=array1**2+array1+1#对array1使用公式进行数学运算，并更新array1
array1#查看更新后的数组array1
list2 = [[0,1,2,3,4], [5,6,7,8,9]]#创建列表list2
array2= np.array(list2)#将列表list2转化为数组形式，得到array2
array2#查看数组array2
np.sum(array2)#对数组array2的所有元素进行求和
np.mean(array2)#对数组array2的所有元素进行求均值
array2.mean(axis=0)#对数组array2的所有元素按列求均值
array2.mean(axis=1)#对数组array2的所有元素按行求均值
array2.cumsum()#对数组array2的所有元素求累积加总值

#矩阵运算
array1 = np.arange(2, 6).reshape(2, 2)#生成一维数组，并转化为2*2的矩阵array1
array1#查看矩阵array1
array1.T#转置矩阵array1
array1 * array1.T#将矩阵array1和置矩阵array1按元素相乘
np.dot(array1, array1.T) #将矩阵array1和置矩阵array1按照矩阵乘法规则相乘
from numpy.linalg import inv, eig#载入模块inv, eig
inv(array1)#求矩阵array1的逆矩阵
eigenvalues, eigenvectors = eig(array1)#求矩阵array1的特征值和特征向量
eigenvalues#查看矩阵array1的特征值
eigenvectors#查看矩阵array1的特征向量

#数组的排序
array4 = np.array([6, 5, 2, 3, 7, 5,1])#生成一维数组array4
array4.sort()#对数组array4进行排序
array4#查看排序后的数组array4
np.unique(array4)#查看数组array4中的非重复值

#数组的索引和切片
list1 = [0,1,2,3,4,5,6,7,8,9]#创建列表list1
array1 = np.array(list1)#将列表list1转化为数组形式，得到array1
array1[3]#索引数组array1中的第4个值
array1[3:7]#切片数组array1中的第4、5、6、7个值
array1[array1 >= 5]#查看数组array1中大于等于5的元素
np.where(array1 >=5, 1, 0)#将数组array1中大于等于5的元素设置为1，其他为0
array1[3:7] = 3 # 将数组array1中的第4、5、6、7个值统一设置为3
array1#查看数组array1

#2.5 python pandas模块序列（series）与数据框（DataFrame）
#2.5.1 序列（series）的相关操作
#一、创建序列（series）
import pandas as pd#导入pandas模块，并简称为pd
Series1 = pd.Series([1,3,5,6,7,6,7])#使用pd.Series()直接创建序列Series1
Series1#查看序列Series1
type(Series1)#查看序列Series1的类型
import numpy as np#导入numpy模块并简称为np
list1 = [1,3,5,6,7,6,7]#创建列表list1
array1 = np.array(list1)#将列表list1转化为数组形式，得到array1
Series1 = pd.Series(list1)#使用pd.Series()将列表list1转化成序列Series1
Series1#查看序列Series1
Series1 = pd.Series(array1)#使用pd.Series()将数组array1转化成序列Series1
Series1#查看序列Series1
#二、序列中元素的索引（index）和值（values）
Series1.index#查看序列Series1元素的索引（index）
Series1.values#查看序列Series1元素的值（values）
Series1[0]#查看序列Series1第1个元素的值
Series1[1]#查看序列Series1第2个元素的值
Series1[[0,2]]#查看序列Series1索引为0和2的元素的值
Series1.index = ['a', 'b', 'c', 'd', 'e', 'f', 'g']#修改序列Series1中元素的索引（index）
Series1#查看更新索引后的序列Series1
Series1['b']#查看更新索引后的序列Series1中的元素'b'
Series1[['a', 'e']]#查看更新索引后的序列Series1中的元素'a', 'e'
#三、序列中元素值的基本统计
Series1 = Series1.sort_values()#将序列Series1的元素按值大小进行排序
Series1#查看序列Series1
Series1.unique()#查看序列Series1中的非重复值
Series1.value_counts()#对序列Series1中的值进行计数统计
# 数据框（DataFrame）的相关操作
#一、创建数据框（DataFrame）
Dict1 = {'credit': [1, 1, 1, 1], 'age': [55, 55, 39, 37],'education': [2,4,3,2], 'workyears': [7.8, 2.6, 6.5, 10.9]}#创建字典Dict1
DataFrame1= pd.DataFrame(Dict1)#将字典Dict1转化为数据框形式
DataFrame1#查看数据框DataFrame1
type(DataFrame1)#查看数据框DataFrame1的类型

list1 = [[1, 55, 2, 7.8], [1, 55, 4, 2.6],[1, 39, 3, 6.5],[1, 37, 2, 10.9]]#创建列表list1
array1= np.array(list1)#将列表list1转化为数组形式，得到array1
array1#查看数组array1
DataFrame1= pd.DataFrame(array1, columns=['credit', 'age', 'education', 'workyears'])#将数组array1转化为数据框形式
DataFrame1#查看数据框DataFrame1
DataFrame1= pd.DataFrame(list1, columns=['credit', 'age', 'education', 'workyears'])#将列表list1转化为数据框形式
DataFrame1#查看数据框DataFrame1
#二、数据框索引（index）、列（columns）和值（values）
DataFrame1.index#查看数据框DataFrame1的索引（index）
DataFrame1.columns#查看数据框DataFrame1的列（columns）
DataFrame1.values#查看数据框DataFrame1的值（values）
DataFrame1 = DataFrame1.sort_values(by='workyears')#将数据框DataFrame1按照'workyears'列变量排序
DataFrame1#查看排序后的数据框DataFrame1
#三、提取数据框中的变量列
DataFrame1['workyears']#提取数据框DataFrame1中'workyears'列
type(DataFrame1['workyears'])#观察数据框中'workyears'列的类型
DataFrame1.workyears#提取数据框DataFrame1中'workyears'列
DataFrame1.loc[:,['education', 'workyears']]#提取数据框DataFrame1中所有行，列'education', 'workyears'
DataFrame1.loc[0,'workyears']#提取数据框DataFrame1中第一行（第一个样本示例），'workyears'的值
DataFrame1.loc[DataFrame1.workyears>6, :]#提取数据框DataFrame1中所有workyears>6的样本示例，所有列
DataFrame1.iloc[0, 1]#提取数据框DataFrame1中第一行（第一个样本示例），第二个变量的值
DataFrame1.iloc[1, 2:]#提取数据框DataFrame1中第二行（第二个样本示例），第三个（含）及以后变量的值
DataFrame1.iloc[:, 2:]#提取数据框DataFrame1中所有行，第三个（含）及以后变量列
#四、从数据框中提取子数据框
DataFrame1[['education', 'workyears']]#提取数据框DataFrame1中'education'、'workyears'列，形成子数据框
type(DataFrame1[['education', 'workyears']])#观察子数据框的类型
DataFrame1.education.value_counts()#观察数据框DataFrame1中education变量的取值计数情况
DataFrame1.age.value_counts()#观察数据框DataFrame1中age变量的取值计数情况
pd.crosstab(DataFrame1.education, DataFrame1.age)#针对'education'、'workyears'两个变量开展交叉表分析
#五、数据框中变量列的编辑操作
DataFrame1.columns = ['y', 'x1', 'x2', 'x3']#将数据框DataFrame1中的列名进行修改
DataFrame1#查看更改列名后的数据框DataFrame1
DataFrame1['x4'] = [6.6,3.6,7.8,10.4]#在数据框DataFrame1中增加1列'x4'
DataFrame1#查看更新后的数据框DataFrame1
DataFrame1['x5'] = np.array([6.6,3.6,7.8,10.4])#在数据框DataFrame1中增加一个数组作为列'x5'
DataFrame1#查看更新后的数据框DataFrame1
DataFrame1 = DataFrame1.drop('x5', axis=1)#在数据框DataFrame1中删除'x5'一列，axis=1表示去掉列
DataFrame1#查看更新后的数据框DataFrame1
DataFrame1 = DataFrame1.drop('x4', axis='columns')#在数据框DataFrame1中删除'x4'一列，是另一种去掉列的方式
DataFrame1#查看更新后的数据框DataFrame1
#输出结果，包括name、credit_rating、loan信息。
        





# python数据读取
#读取csv或者txt
import pandas as pd#导入pandas模块，并简称为pd
data=pd.read_csv('数据4.1.csv')
data=pd.read_csv('数据4.1.txt')
data=pd.read_csv('数据4.1.txt',sep='\t')#从设置路径中读取数据4.1，数据4.1为.csv格式
data=pd.read_csv('数据4.1.txt',sep='\s+')#从设置路径中读取数据4.1，数据4.1为.csv格式
data=pd.read_table('数据4.1.txt')
data=pd.read_csv('数据4.1.csv', header=None)
data=pd.read_csv('数据4.1.csv',names = ['V1', 'V2', 'V3', 'V4'])
data=pd.read_csv('数据4.1.csv', skiprows=[0],names = ['V1', 'V2', 'V3', 'V4'])

#读取excel数据
import pandas as pd#导入pandas模块，并简称为pd
data=pd.read_excel('数据4.1.xlsx')
data=pd.read_excel('数据4.1.xlsx',sheet_name='数据4.1副本') # 读取某个sheet表
data=pd.read_excel('数据4.1.xlsx')[['profit','invest']] # 读取并筛选几列

#读取spss数据
#pip install --upgrade pyreadstat#安装pyreadstat，大家在运行时把最前面的#号去掉
import pandas as pd#导入pandas模块，并简称为pd
data=pd.read_spss('数据7.1.sav')

#读取stata数据
import pandas as pd#导入pandas模块，并简称为pd
data=pd.read_stata('数据8.dta')
    
#python数据检索
data=pd.read_csv('数据2.1.csv')#读取数据2.1
data.describe()   # 描述统计
data.info()    # 基本信息
data.dtypes    #　列格式类型
data.head(2)    #前n行
data.tail(2)    #后ｎ行
data.shape    #维度
data.index  #索引
data.columns  #列名

#python数据缺失值处理
#查看数据集中的缺失值
data=pd.read_csv('数据2.1.csv')#读取数据2.1
data.isnull()  #判断是否是缺失值,不能省略括号
data.notnull()#判断是否是缺失值,不能省略括号
data.isnull().value_counts()#计算缺失值个数
data.isna().sum()  # 按列计算缺失值个数
data.isnull().sum().sort_values(ascending=False).head()#计算缺失值个数并排序
#2.9.2填充数据集中的缺失值
#一、用字符串'缺失数据'代替
data.fillna('缺失数据',inplace=True)  # 填充缺失值
data.isnull().value_counts()#重新计算缺失值个数
#二、用前后值填充缺失数据
data=pd.read_csv('数据2.1.csv')#重新读取数据2.1
data.fillna(method='pad') #类似于excel中用上一个单元格内容批量填充
data=pd.read_csv('数据2.1.csv')#重新读取数据2.1
data.fillna(method='bfill')#用后一个非缺失值（未来值）填充
#三、用变量均值或者中位数填充缺失数据
data=pd.read_csv('数据2.1.csv')#重新读取数据2.1
data.describe()#对数据2.1开展描述性分析，重点观察变量的均值、中位数
data.fillna(data.mean())#依据列变量的均值对列中的缺失数据进行填充
data=pd.read_csv('数据2.1.csv')#重新读取数据2.1
data.fillna(data.median())#依据列变量的中位数对列中的缺失数据进行填充
#四、用线性插值法填充缺失数据
data=pd.read_csv('数据2.1.csv')#重新读取数据2.1
data.interpolate()#使用线性插值法对列中的缺失数据进行填充
#删除数据集中的缺失值
data=pd.read_csv('数据2.1.csv')#重新读取数据2.1
data.dropna()#只要有列变量存在缺失值，则整行（整个样本示例）即被删除
data=pd.read_csv('数据2.1.csv')#重新读取数据2.1
data.dropna(how='all')#只有所有列变量都为缺失值，整行（整个样本示例）才被删除
data=pd.read_csv('数据2.1.csv')#重新读取数据2.1
data.dropna(axis=1)#针对某列变量，只要存在缺失值，整列就被删除
data=pd.read_csv('数据2.1.csv')#重新读取数据2.1
data.dropna(axis=1,how='all')#针对某列变量，只有所有样本示例均为缺失值，整列才被删除

#python数据重复值处理
#1.查看数据集中的重复值
data=pd.read_csv('数据2.2.csv')#读取数据2.2
data.duplicated()#找出数据2.2中的重复样本示例
data.duplicated('pb')#当变量'pb'相同时，即视为重复示例，找出数据2.2中的重复样本示例
data['pb'].unique() #找出数据2.2中'pb'变量的不重复样本示例
data['pb'].unique().tolist() # 以列表list形式展示 'pb'变量的不重复样本示例
data['pb'].nunique() #计算 'pb'变量的不重复值的个数 
#2去除数据集中的重复值
data.drop_duplicates()#将数据集中重复的样本示例去掉，默认保留重复值中第一个出现的示例
data.drop_duplicates(keep='last')#将数据集中重复的样本示例去掉，保留重复值中最后出现的示例
data.drop_duplicates(['roe'])#当变量'roe'相同时，即视为重复示例，将数据集中重复的样本示例去掉
data.drop_duplicates(['pb','roe'])#当变量'pb'和'roe'都相同时，视为重复示例，将数据集中重复的样本示例去掉

#python数据行列处理
#1 删除变量列、样本示例行
data=pd.read_csv('数据2.2.csv')#读取数据2.2
data.drop('pb',axis=1,inplace=True)#删除'pb'变量列，其中axis=1表示列，不创建新的对象,直接对原始对象进行修改
data.drop(labels=[0,3,5], axis=0)#删除编号为0、3、5的样本示例，axis=0表示行
#2 更改变量列名称、调整变量列顺序
data=pd.read_csv('数据2.2.csv')#读取数据2.2
data.columns= ['V1', 'V2', 'V3', 'V4', 'V5']#更改全部列名，需要注意列名的个数等于代码中变量个数
data.rename(columns = {'V1':'var1'},inplace=True) #更改单个列名，注意参数columns不能少
data = data[['var1','V3','V2','V4', 'V5']]#调整数据集中列的顺序
#3 改变列的数据格式
data=pd.read_csv('数据2.2.csv')#读取数据2.2
data['pb'] = data['pb'].astype('int') #将列变量'pb'的数据类型更改为整数型
data.dtypes#观察数据集中各变量的数据类型
data['pb'] = data['pb'].astype('float')#将列变量'pb'的数据类型再更改为浮点型
data.dtypes#观察数据集中各变量的数据类型
#4 多列转换
data=pd.read_csv('数据2.2.csv')#读取数据2.2
data[['pb','roe','debt']]=data[['pb','roe','debt']].astype(str)  #将'pb','roe','name'三列数据均转换成字符串格式
data.dtypes#观察数据集中各变量的数据类型，字符串在Pandas中的类型为object
data['roe'].apply(lambda x:isinstance(x,str))#判断格式，是否为字符串
#5 数据百分比格式转换
data=pd.read_csv('数据2.2.csv')#读取数据2.2
data['roe'] = data['roe'].apply(lambda x: '%.2f%%' % (x*100))#将变量'roe'的数据改成百分比格式
'''


