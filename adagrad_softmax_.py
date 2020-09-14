# -*- coding: utf-8 -*-
"""adagrad_softmax .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GCGLaIgk4NOhdk908Jmrd577EhFQlv5Q
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('D:/anomaly detection/NSLKDD/KDDTrain+.csv')
dataset_test = pd.read_csv('D:/anomaly detection/NSLKDD/KDDTest+.csv')

X = dataset_train.iloc[:, :-2].values
Y = dataset_train.iloc[:, 41].values

X_test = dataset_test.iloc[:, :-2].values
Y_test = dataset_test.iloc[:, 41].values

print(X.shape, X_test.shape, Y.shape, Y_test.shape)

# a = set(list(X_train[:, 2])) 
# b = set(list(X_dev[:, 2]))
# print(a-b)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X[:, 1] = LabelEncoder().fit_transform(X[:, 1])
X[:, 2] = LabelEncoder().fit_transform(X[:, 2])
X[:, 3] = LabelEncoder().fit_transform(X[:, 3])
X = OneHotEncoder(categorical_features=[1,2,3]).fit_transform(X).toarray()

X_test[:, 1] = LabelEncoder().fit_transform(X_test[:, 1])
X_test[:, 2] = LabelEncoder().fit_transform(X_test[:, 2])
X_test[:, 3] = LabelEncoder().fit_transform(X_test[:, 3])
X_test = OneHotEncoder(categorical_features=[1,2,3]).fit_transform(X_test).toarray()

# X_dev[:, 1] = LabelEncoder().fit_transform(X_dev[:, 1])
# X_dev[:, 2] = LabelEncoder().fit_transform(X_dev[:, 2])
# X_dev[:, 3] = LabelEncoder().fit_transform(X_dev[:, 3])
# X_dev = OneHotEncoder(categorical_features=[1,2,3]).fit_transform(X_dev).toarray()

print(X.shape, Y.shape, X_test.shape, Y_test.shape)

X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=0.15, random_state=2)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, X_dev.shape, Y_dev.shape)

dos = ['mailbomb', 'back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'worm']
probe = ['ipsweep', 'satan', 'nmap', 'portsweep', 'mscan', 'saint']
r2l = ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named']
u2r = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']

for i in range(0, len(Y_train)):
  if Y_train[i] == 'normal':
    Y_train[i] = 0
  elif Y_train[i] in dos:
    Y_train[i] = 1
  elif Y_train[i] in probe:
    Y_train[i] = 2
  elif Y_train[i] in r2l:
    Y_train[i] = 3
  elif Y_train[i] in u2r:
    Y_train[i] = 4

for i in range(0, len(Y_test)):
  if Y_test[i] == 'normal':
    Y_test[i] = 0
  elif Y_test[i] in dos:
    Y_test[i] = 1
  elif Y_test[i] in probe:
    Y_test[i] = 2
  elif Y_test[i] in r2l:
    Y_test[i] = 3
  elif Y_test[i] in u2r:
    Y_test[i] = 4
    
for i in range(0, len(Y_dev)):
  if Y_dev[i] == 'normal':
    Y_dev[i] = 0
  elif Y_dev[i] in dos:
    Y_dev[i] = 1
  elif Y_dev[i] in probe:
    Y_dev[i] = 2
  elif Y_dev[i] in r2l:
    Y_dev[i] = 3
  elif Y_dev[i] in u2r:
    Y_dev[i] = 4

print(set(list(Y_train)))
print(set(list(Y_test)))
print(set(list(Y_dev)))

print(Y_train.shape)
print(Y_train)

Y_train = LabelEncoder().fit_transform(Y_train)
Y_test = LabelEncoder().fit_transform(Y_test)
Y_dev = LabelEncoder().fit_transform(Y_dev)

print(Y_train.shape)
print(Y_train)

Y_train

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

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_dev = sc.transform(X_dev)
X_test = sc.transform(X_test)

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_dev = np.reshape(X_dev, (X_dev.shape[0], 1, X_dev.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

X_train.shape

x_train = np.array(X_train)
x_test = np.array(X_test)
x_dev = np.array(X_dev)
y_train1 = np.array(Y_train)
y_test1 = np.array(Y_test)
y_dev1 = np.array(Y_dev)

y_train= to_categorical(y_train1)
y_test= to_categorical(y_test1)
y_dev = to_categorical(y_dev1)

X_test.shape

batch_size = 32

model = Sequential()
model.add(LSTM(80,input_dim=122))  
model.add(Dropout(0.1))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(100, activation='relu'))
model.add(Dense(5))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',optimizer='adagrad',metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=50, validation_data=(x_dev, y_dev))

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)

model.summary()

model.get_weights()

model.save('adagrad_softmax.h5')

del model

model = load_model('my_model.h5')

model.fit(x_train, y_train, batch_size=batch_size, epochs=2, validation_data=(x_dev, y_dev))

sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=500, validation_data=(x_dev, y_dev))

loss, accuracy = model.evaluate(x_test, y_test)
print(accuracy)

model.save('my_model1.h5')

