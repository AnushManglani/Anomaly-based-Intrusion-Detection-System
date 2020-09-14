# -*- coding: utf-8 -*-
"""
Created on Sun May  5 21:59:55 2019

@author: Admin
"""

import numpy as np
import pandas as pd

data1 = pd.read_csv('C:/Users/Admin/Downloads/UNSW-NB15_1.csv', header=None)
data2 = pd.read_csv('C:/Users/Admin/Downloads/UNSW-NB15_2.csv', header=None)
data3 = pd.read_csv('C:/Users/Admin/Downloads/UNSW-NB15_3.csv', header=None)
data4 = pd.read_csv('C:/Users/Admin/Downloads/UNSW-NB15_4.csv', header=None)

#data4.columns = data4.columns.str.replace(' ', '')

data1.fillna(0, inplace=True)
data2.fillna(0, inplace=True)
data3.fillna(0, inplace=True)
data4.fillna(0, inplace=True)

d1 = pd.concat([data1, data2])
d2 = pd.concat([d1, data3])
d3 = pd.concat([d2, data4])

one_hot = pd.get_dummies(d3)
X = one_hot.drop([48, '47_0', '47_ Fuzzers', '47_Fuzzers', '47_Fuzzers', '47_Generic', '47_Reconnaissance', '47_Shellcode', '47_Worms', '47_Analysis', '47_Backdoor', '47_Backdoors', '47_DoS', '47_Exploits'], axis=1)

h = X.head()
print(h)

Y = d3.iloc[:, 48].values
print(X.shape, Y.shape)

from sklearn.model_selection import train_test_split
X_m, X_test, Y_m, Y_test = train_test_split(X, Y, test_size=0.10, random_state=2)
print(X_m.shape, Y_m.shape, X_test.shape, Y_test.shape)

X_train, X_dev, Y_train, Y_dev = train_test_split(X_m, Y_m, test_size=0.05, random_state=2)
print(X_train.shape, Y_train.shape, X_dev.shape, Y_dev.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_dev = sc.transform(X_dev)
X_test = sc.transform(X_test)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_dev = np.reshape(X_dev, (X_dev.shape[0], 1, X_dev.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

from keras.preprocessing import sequence
from IPython.display import SVG
from keras.utils import np_utils,plot_model
from keras.utils.vis_utils import model_to_dot
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import h5py
from keras import callbacks
from keras import callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from keras.models import load_model
from keras import optimizers



x_train = np.array(X_train)
x_test = np.array(X_test)
x_dev = np.array(X_dev)
y_train = np.array(Y_train)
y_test = np.array(Y_test)
y_dev = np.array(Y_dev)

batch_size = 32

model = Sequential()
model.add(LSTM(80,input_dim=296))  
model.add(Dropout(0.1))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
#del model
model = load_model('UNSW_adagrad.h5')
model.compile(loss='binary_crossentropy',optimizer='adagrad',metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=100, validation_data=(x_dev, y_dev))

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)



model.save('UNSW_adagrad.h5')

print(data4.isnull().any())