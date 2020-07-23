# coding: utf-8
import numpy as np
from gan_layers import *

class SequentialGen:
	def __init__(self):
		self.layersPre = None
		self.lastLayer = None
		self.layers = []
		self.trainable = True

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

	def predict(self, x, isTrain):
		for layer in self.layers:
			x = layer.forward(x, isTrain)
		return x

	def backProp(self, x):
		for layer in self.layers:
			layer.trainable = self.trainable
		self.layers.reverse()
		for layer in self.layers:
			x = layer.backward(x)
		self.layers.reverse()
		return