#coding: utf-8
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

fname_train = '../../NNdata/img_100000.csv'
fname_test = '../../NNdata/rindex_100000.csv'
fname_val = '../../NNdata/1.5_0.015.csv'
data_train = np.loadtxt(fname_train, delimiter=',')
data_test = np.loadtxt(fname_test, delimiter=',')
data_val = np.reshape(np.loadtxt(fname_val, delimiter=','), (1, 200))
batch_size = 100

model = Sequential()
model.add(Dense(units=200, input_shape=(200,)))
model.add(Dense(units=200))
sgd = optimizers.SGD(lr=0.001)
mae = keras.losses.MeanAbsoluteError()
mse = keras.losses.MeanSquaredError()
model.compile(optimizer='sgd', loss='mae')
log_f = model.fit(x=data_train, y=data_test, batch_size=batch_size, epochs=100, \
		validation_split=0.2, shuffle=True)
op = model.predict_on_batch(x=data_val)
print(op)