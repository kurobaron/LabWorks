# coding: utf-8
import numpy as np
from gan_layers import *

class Sequential:
	def __init__(self, inpDim, isDiscriminator):
		self.inpDim = inpDim
		self.layersPre = None
		self.lastLayer = None
		self.layers = []
		self.trainable = True
		self.isTrain = True
		self.isReverse = False
		self.isDiscriminator = isDiscriminator
		if self.isDiscriminator:
			self.lastLayer = 

	def compile(self):
		for i in layersPre:
			if i[0] == 'Dense':
				affine = Affine(i[1], i[2], i[3], self.trainable)
				self.layers.append(affine)
			elif i[0] == 'Relu':
				relu = Relu()
				self.layers.append(relu)
			elif i[0] == 'LeakyRelu':
				leaky_relu = LeakyRelu(i[1])
				self.layers.append(leaky_relu)
			elif i[0] == 'BatchNormalization':
				batch_normalization = BatchNormalization(i[1], i[2], i[3], i[4], self.trainable)
				self.layers.append(batch_normalization)
			elif i[0] == 'Sigmoid':
				sigmoid = Sigmoid()
				self.layers.append(sigmoid)
			elif i[0] == 'Tanh':
				tanh = Tanh()
				self.layers.append(tanh)

	def predict(self, x):
		if self.isReverse:
			self.layers.reverse()
			self.isReverse = False
		for layer in self.layers:
			x = layer.forward(x, self.isTrain)
		return x

	def loss(self, x, t):
		y = self.predict(x)
		return self.lastLayer.forward(y, t)

	def gradient(self, x, t):
		self.loss(x, t)
		dout = 1
		dout = self.lastLayer.backward(dout)
		if not self.isReverse:
			self.layers.reverse()
			self.isReverse = True
		for layer in self.layers:
			dout = layer.backward(dout)
		return dout