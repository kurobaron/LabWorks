# coding: utf-8
import numpy as np
from gan_optimizer import *

class Affine:
    def __init__(self, inputDim, outputDim, learningRate = 0.001, weight = 0.01):
        self.W = weight*np.random.randn(inputDim, outputDim)
        self.b = np.zeros(outputDim)
        self.lr = learningRate
        #self.optimizer = Adam(self.W, self.b, self.lr)
        self.optimizer = SGD(self.W, self.b, self.lr) #変更点
        self.trainable = None
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x, train_flg = True):
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        if self.trainable:
            self.dW = np.dot(self.x.T, dout)
            self.db = np.sum(dout, axis = 0)
            self.optimizer.update(self.dW, self.db)
        return dx

class Relu:
    def __init__(self):
        self.mask = None
        self.trainable = None

    def forward(self, x, train_flg = True):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class LeakyRelu:
    def __init__(self, alpha = 0.01):
        self.alpha = alpha
        self.mask = None
        self.trainable = None

    def forward(self, x, train_flg = True):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = x[self.mask]*self.alpha
        return out

    def backward(self, dout):
        dout[self.mask] = dout[self.mask]*self.alpha
        dx = dout
        return dx

class Sigmoid:
    def __init__(self):
        self.out = None
        self.trainable = None

    def forward(self, x, train_flg = True):
        out = 1/(1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout*(1.0 - self.out)*self.out
        return dx

class Tanh:
    def __init__(self):
        self.out = None
        self.trainable = None

    def forward(self, x, train_flg = True):
        out = (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout*(1.0 - self.out*self.out)
        return dx

class Dropout:
    def __init__(self, dropout_ratio = 0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.trainable = None

    def forward(self, x, train_flg = True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x*self.mask
        else:
            return x*(1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout*self.mask

class BatchNormalization:
    def __init__(self, gamma, beta, momentum = 0.9, learningRate = 0.001, running_mean = None, running_var = None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.lr = learningRate
        self.trainable = None
        self.optimizer = Adam(self.gamma, self.beta, self.lr)
        self.running_mean = running_mean
        self.running_var = running_var          
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg = True):
        out = self.__forward(x, train_flg)        
        return out
            
    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)                        
        if train_flg:
            mu = x.mean(axis = 0)
            xc = x - mu
            var = np.mean(xc**2, axis = 0)
            std = np.sqrt(var + 10e-7)
            xn = xc/std            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum*self.running_mean + (1 - self.momentum)*mu
            self.running_var = self.momentum*self.running_var + (1 - self.momentum)*var            
        else:
            xc = x - self.running_mean
            xn = xc/((np.sqrt(self.running_var + 10e-7)))            
        out = self.gamma*xn + self.beta 
        return out

    def backward(self, dout):
        dx = self.__backward(dout)
        if self.trainable:
            self.optimizer.update(self.dgamma, self.dbeta)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis = 0)
        dgamma = np.sum(self.xn*dout, axis = 0)
        dxn = self.gamma*dout
        dxc = dxn/self.std
        dstd = -np.sum((dxn*self.xc)/(self.std*self.std), axis = 0)
        dvar = 0.5*dstd/self.std
        dxc += (2.0/self.batch_size)*self.xc*dvar
        dmu = np.sum(dxc, axis = 0)
        dx = dxc - dmu/self.batch_size        
        self.dgamma = dgamma
        self.dbeta = dbeta        
        return dx

class MSELoss:
    def __init__(self):
        self.loss = None
        self.y = None 
        self.t = None 
        self.trainable = None

    def forward(self, y, t):
        self.t = t
        self.y = y
        batch_size = self.y.shape[0]
        self.loss = 0.5*np.sum((y - t)**2)/batch_size        
        return self.loss

    def backward(self, dout = 1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t)/batch_size
        return dx
