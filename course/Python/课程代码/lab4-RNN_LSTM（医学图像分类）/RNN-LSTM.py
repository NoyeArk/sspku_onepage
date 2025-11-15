#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import keras.backend as K
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score

data_train = pd.read_csv("data.csv")
print(data_train.head())

print(data_train.dtypes)

print(data_train.isnull().any())

# 显示不同特征的分布情况
features = ['age']
fig = plt.subplots(figsize=(15, 15))
for i, j in enumerate(features):
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=j, data=data_train)
    plt.title("No. of age")
plt.show()

features = ['gender']
fig = plt.subplots(figsize=(15, 15))
for i, j in enumerate(features):
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=j, data=data_train)
    plt.title("No. of gender")
plt.show()

features = ['heart_failure']
fig = plt.subplots(figsize=(15, 15))
for i, j in enumerate(features):
    plt.subplots_adjust(hspace=1.0)
    sns.countplot(x=j, data=data_train)
    plt.title("No. of heart_failure")
plt.show()

df_tmp1 = data_train[
    ['age', 'gender', 'body_mass_index', 'heart_failure', 'hypertension', 'chronic_obstructic_pulmonary_disease',
     'chronic_liver_disease', 'diabetes_mellitus', 'chroinc_kidney_disease', 'charlson',
     'emergency', 'surgery', 'acute_kidney_disease']]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df_tmp1.corr(), cmap="YlGnBu", annot=True)
plt.show()

df_tmp2 = data_train[
    ['APSIII', 'SAPSII', 'non_renal_sofa-1', 'non_renal_sofa-3', 'non_renal_sofa', 'aki_stage', 'creatinine_baseline',
     'creatinine-1', 'creatinine-3', 'creatinine',
     'urine_output-1', 'urine_output-3', 'urine_output', 'diuretic', 'mechanical_ventalition', 'renal_toxic_drug',
     'acute_kidney_disease']]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.heatmap(df_tmp2.corr(), cmap="YlGnBu", annot=True)
plt.show()

data_test = pd.read_csv("test.csv")
print(data_test.head())

X_train = data_train.drop(['acute_kidney_disease'], axis=1)
y_train = data_train['acute_kidney_disease']

import keras.layers as layers

X_train = layers.Lambda(lambda X_train: K.expand_dims(X_train, axis=-1))(X_train)

print(X_train.shape)

X_test = data_test.drop(['acute_kidney_disease'], axis=1)
y_test = data_test['acute_kidney_disease']

X_test = layers.Lambda(lambda X_test: K.expand_dims(X_test, axis=-1))(X_test)

print(X_test.shape)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

lstm = Sequential()
lstm.add(LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm.add(LSTM(50))
lstm.add(Dense(10, activation='relu'))
lstm.add(Dense(1, activation='sigmoid'))
lstm.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['acc'])
history = lstm.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=64)
score = lstm.evaluate(X_test, y_test, batch_size=128)
print(lstm.summary())

from keras.utils import plot_model

plot_model(lstm, to_file='model.png')


def show_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Test acc')
    plt.title('Training and Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


y_pred = lstm.predict(X_test, batch_size=10)
y_pred = np.round(y_pred)
data_test['y_pred']=y_pred
data_test.to_excel('data_test_pred.xlsx')

print("验证集查准率: {:.2f}%".format(precision_score(y_test, y_pred) * 100))
print("验证集查全率: {:.2f}%".format(recall_score(y_test, y_pred) * 100))
print("验证集F1值: {:.2f}%".format(f1_score(y_test, y_pred) * 100))

show_history(history)


def roc_f(y_data, y_score, title):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, threshold = roc_curve(y_data, y_score)

    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    # plt.figure(figsize=(10, 10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ' RNN-LSTM Model ')
    plt.legend(loc="lower right")
    plt.show()


y_train_score = lstm.predict_proba(X_train)
y_test_score = lstm.predict_proba(X_test)

roc_f(y_train, y_train_score, 'Training')
roc_f(y_test, y_test_score, 'Test')
