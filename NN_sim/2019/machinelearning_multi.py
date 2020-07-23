# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
import time
from multi_layer_net import MultiLayerNet
from optimizer import *

validation_split = float(input('Input split rate : '))
name_train = input('Input the name of train file : ')
name_test = input('Input the name of test file : ')
name_sim = input('Input the name of file to simulate : ')
fname_train = name_train + '.csv'
fname_test = name_test + '.csv'
fname_sim = name_sim + '.csv'
data_train = np.loadtxt(fname_train, delimiter = ',')
data_test = np.loadtxt(fname_test, delimiter = ',')
data_sim = np.loadtxt(fname_sim, delimiter = ',')
data_sim = np.reshape(data_sim, (1,200))
indices = np.arange(data_train.shape[0])
np.random.shuffle(indices)
validation_size = int(data_train.shape[0]*validation_split)
x_train, x_test = data_train[indices[:-validation_size], :], data_train[indices[-validation_size:], :]
t_train, t_test = data_test[indices[:-validation_size], :], data_test[indices[-validation_size:], :]
#(x_train, t_train), (x_test, t_test) = (data_train[0], data_train[1]), (data_test[0], data_test[1])
hid_size = 200
layer_num = 2
network = MultiLayerNet(input_size = x_train.shape[1], hidden_size_list = [hid_size], 
	output_size = t_train.shape[1], weight_init_std = 0.01)
optimizer = Adam()
iters_num = 100000
train_size = x_train.shape[0]
test_size = x_test.shape[0]
batch_size = 100
learning_rate = 0.001
train_loss = []
#train_acc = []
test_loss = []
#test_acc = []
iter_per_epoch = max(train_size/batch_size, 1)
values = {}
w1_value = []
b1_value = []
w2_value = []
b2_value = []
w3_value = []
b3_value = []
for i in range(iters_num):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]
	grad = network.gradient(x_batch, t_batch)
	optimizer.update(network.params, grad)
	for key in ('W1', 'b1', 'W2', 'b2'):
		values[key] = np.sum(network.params[key])/network.params[key].size
		#network.params[key] -= learning_rate*grad[key]
	loss = network.loss(x_batch, t_batch)
	train_loss.append(loss)
	batch_mask_val = np.random.choice(test_size, batch_size)
	x_batch_val = x_test[batch_mask_val]
	t_batch_val = t_test[batch_mask_val]
	loss_val = network.loss(x_batch_val, t_batch_val)
	test_loss.append(loss_val)
	w1_value.append(values['W1'])
	b1_value.append(values['b1'])
	w2_value.append(values['W2'])
	b2_value.append(values['b2'])
	if i%iter_per_epoch == 0:
		"""w1_value.append(values['W1'])
		b1_value.append(values['b1'])
		w2_value.append(values['W2'])
		b2_value.append(values['b2'])
		train_loss.append(loss)
		test_loss.append(loss_val)"""

		"""train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc.append(train_acc)
		test_acc.append(test_acc)"""
		#print(train_acc, test_acc)
		print("iter number : {}/{}, loss = {}".format(i, iters_num, loss))
x = np.arange(iters_num)

"""if iter_per_epoch%iter_per_epoch == 0:
	x = np.arange(iters_num/iter_per_epoch)
else:
	x = np.arange(iters_num/iter_per_epoch + 1)"""

plt.rcParams['font.family'] ='sans-serif'#使用するフォント
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
plt.rcParams['font.size'] = 15 #フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ

y = train_loss
plt.plot(x, y)
plt.legend()
plt.xlabel('Iteration number')
#plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 100)
plt.show()

y = train_loss
plt.plot(x, y)
plt.legend()
plt.xlabel('Iteration number')
#plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 20)
plt.show()

y_val = test_loss
plt.plot(x, y_val, color = "#ff4500")
plt.legend()
plt.xlabel('Iteration number')
#plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 100)
plt.show()

y_val = test_loss
plt.plot(x, y_val, color = "#ff4500")
plt.legend()
plt.xlabel('Iteration number')
#plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 20)
plt.show()

y_w1 = w1_value
plt.plot(x, y_w1, color = "#008000")
plt.legend()
plt.xlabel('Iteration number')
#plt.xlabel('Epoch')
plt.ylabel('Mean of W1\'s values')
plt.show()

y_b1 = b1_value
plt.plot(x, y_b1, color = "#008000")
plt.legend()
plt.xlabel('Iteration number')
#plt.xlabel('Epoch')
plt.ylabel('Mean of b1\'s values')
plt.show()

y_w2 = w2_value
plt.plot(x, y_w2, color = "#483d8b")
plt.legend()
plt.xlabel('Iteration number')
#plt.xlabel('Epoch')
plt.ylabel('Mean of W2\'s values')
plt.show()

y_b2 = b2_value
plt.plot(x, y_b2, color = "#483d8b")
plt.legend()
plt.xlabel('Iteration number')
#plt.xlabel('Epoch')
plt.ylabel('Mean of b2\'s values')
plt.show()

np.savetxt(name_sim + "_iter_{}_h_{}_{}_{}_simulated.csv".format(iters_num, hid_size, layer_num, str(int(time.time()))), network.predict(data_sim), delimiter = ',')