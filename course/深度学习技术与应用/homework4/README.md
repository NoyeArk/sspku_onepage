# 平时作业4：Code Generation

Code Generation是一个以自然语言为输入，输出一个代码片段的任务。要求该输出的代码片段能够完成自然语言输入所描述的编程任务。在通常情况下，自然语言输入的长度单位是一个语句，而相应的程序输出可以是一行代码、多行代码或一个完整的方法体。

CONCODE是一个较为经典的Code Generation任务的数据集。

本次作业的要求是：以CONCODE数据集为训练集和测试集，完成一个支持程序代码生成的深度神经网络。

一、任务数据集：

本次作业的数据集选用CodeXGlue数据集中与代码生成相关的子数据集CONCODE，数据相关的格式、基本状况可以参考如下的链接：

https://github.com/Dingjz/CodeXGLUE/tree/main/Text-Code/text-to-code

二、结果汇报

请提供你的【程序源代码】及【模型训练介绍PPT】，其中PPT应包含以下内容：

（1）请提供你所采用的模型结构的图示及相关说明；

（2）请提供你的模型在验证数据集和测试数据集上的结果，衡量指标采用：Exact Match 和 BLEU

（3）请提供能够体现你的训练过程的Learn Curve及相关说明。