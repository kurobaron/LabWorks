# coding: utf-8
import numpy as np
from gan_layers import *

class SequentialDis:
	def __init__(self):
		self.layersPre = None
		self.lastLayer = None
		self.layers = []
		self.trainable = True
		self.lastLayer = None

	def compile(self):
		for i in self.layersPre:
			if i[0] == 'Dense':
				affine = Affine(i[1], i[2], i[3], i[4])
				self.layers.append(affine)
			elif i[0] == 'Relu':
				relu = Relu()
				self.layers.append(relu)
			elif i[0] == 'LeakyRelu':
				leaky_relu = LeakyRelu(i[1])
				self.layers.append(leaky_relu)
			elif i[0] == 'BatchNormalization':
				batch_normalization = BatchNormalization(i[1], i[2], i[3], i[4])
				self.layers.append(batch_normalization)
			elif i[0] == 'Sigmoid':
				sigmoid = Sigmoid()
				self.layers.append(sigmoid)
			elif i[0] == 'Tanh':
				tanh = Tanh()
				self.layers.append(tanh)
			elif i[0] == 'MSELoss':
				self.lastLayer = MSELoss()
			"""elif i[0] == 'CrossEntropy':
				self.lastLayer = CrossEntropy()"""		 

	def train(self, x, t, isTrain):
		for layer in self.layers:
			layer.trainable = self.trainable
		vals = [0]*2
		for layer in self.layers:
			x = layer.forward(x, isTrain)
		vals[0] = self.lastLayer.forward(x, t)
		if not isTrain:
			return vals
		dout = 1
		dout = self.lastLayer.backward(dout)
		self.layers.reverse()
		for layer in self.layers:
			dout = layer.backward(dout)
		vals[1] = dout
		self.layers.reverse()
		return vals