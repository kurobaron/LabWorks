# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from multi_layer_net import MultiLayerNet
from optimizer import *

validation_split = 0.2
name = input('Input the simulation conditions : ')
fname_train = 'img_100000_new.csv'
fname_test = 'rindex_100000_new.csv'
data_train = np.loadtxt(fname_train, delimiter = ',')
data_test = np.loadtxt(fname_test, delimiter = ',')
indices = np.arange(data_train.shape[0])
np.random.shuffle(indices)
validation_size = int(data_train.shape[0]*validation_split)
x_train, x_test = data_train[indices[:-validation_size], :], data_train[indices[-validation_size:], :]
t_train, t_test = data_test[indices[:-validation_size], :], data_test[indices[-validation_size:], :]
hid_size = [200]
layer_num = len(hid_size) + 1
network = MultiLayerNet(input_size = x_train.shape[1], hidden_size_list = hid_size, 
	output_size = t_train.shape[1], weight_init_std = 0.01)
optimizer = SGD(0.001)
iters_num = 100000
train_size = x_train.shape[0]
test_size = x_test.shape[0]
batch_size = 100
train_loss = []
#train_acc = []
test_loss = []
#test_acc = []
iter_per_epoch = max(train_size/batch_size, 1)
values = {}
for key in network.indexes:
	values[key] = []
for i in range(iters_num):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]
	grad = network.gradient(x_batch, t_batch)
	optimizer.update(network.params, grad)
	loss = network.loss(x_batch, t_batch)
	train_loss.append(loss)
	batch_mask_val = np.random.choice(test_size, batch_size)
	x_batch_val = x_test[batch_mask_val]
	t_batch_val = t_test[batch_mask_val]
	loss_val = network.loss(x_batch_val, t_batch_val)
	test_loss.append(loss_val)
	for key, value in values.items():
		value.append(np.sum(network.params[key])/network.params[key].size)
	if i%iter_per_epoch == 0:

		"""train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc.append(train_acc)
		test_acc.append(test_acc)"""
		#print(train_acc, test_acc)
		print("iter number : {}/{}, loss = {}".format(i, iters_num, loss))

with open ('loss.csv', 'a') as f:
	writer = csv.writer(f)
	writer.writerow(["{}".format(name), "{}".format(loss), "{}".format(loss_val)])