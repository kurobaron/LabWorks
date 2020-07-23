# coding: utf-8
import numpy as np

class CombinedModel:
	def __init__(self, modelGen, modelDis):
		self.modelGen = modelGen
		self.modelDis = modelDis

	def predict(self, x):
		if self.modelGen.isReverse:
			self.modelGen.layers.reverse()
			self.modelGen.isReverse = False
		if self.modelDis.isReverse:
			self.modelDis.layers.reverse()
			self.modelDis.isReverse = False
		for layer in self.modelGen.layers:
			x = layer.forward(x, self.modelGen.isTrain)
		for layer in self.modelDis.layers:
			x = layer.forward(x, self.modelDis.isTrain)
		return x

	def loss(self, x, t):
		y = self.predict(x)
		return self.modelDis.lastLayer.forward(y, t)

	def gradient(self, x, t):
		self.loss(x, t)
		dout = 1
		dout = self.modelDis.lastLayer.backward(dout)
		if not self.modelDis.isReverse:
			self.modelDis.layers.reverse()
			self.modelDis.isReverse = True
		if not self.modelGen.isReverse:
			self.modelGen.layers.reverse()
			self.modelGen.isReverse = True
		for layer in self.modelDis.layers:
			dout = layer.backward(dout)
		for layer in self.modelGen.layers:
			dout = layer.backward(dout)
		return dout