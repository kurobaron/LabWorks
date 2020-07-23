# coding: utf-8
import numpy as np

class SGD:
    def __init__(self, param1 = None, param2 = None, lr=0.01):
        self.lr = lr
        self.params = {'param1':param1, 'param2':param2}
        self.grads = {}
               
    def update(self, dparam1 = None, dparam2 = None):
        self.grads['param1'] = dparam1
        self.grads['param2'] = dparam2       
        for key in self.params.keys():
            if self.params[key] is None:
                continue
            self.params[key] -= self.lr*self.grads[key]
        
class Momentum:
    def __init__(self, param1 = None, param2 = None, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.params = {'param1':param1, 'param2':param2}
        self.grads = {}
        self.v = {}
        for key in self.params.keys():
            if self.params[key] is None:
                continue
            self.v[key] = np.zeros_like(self.params[key])
               
    def update(self, dparam1 = None, dparam2 = None):
        self.grads['param1'] = dparam1
        self.grads['param2'] = dparam2       
        for key in self.params.keys():
            if self.params[key] is None:
                continue
            self.v[key] = self.momentum*self.v[key] - self.lr*self.grads[key]
            self.params[key] += self.v[key]

class AdaGrad:
    def __init__(self, param1 = None, param2 = None, lr=0.01):
        self.lr = lr
        self.params = {'param1':param1, 'param2':param2}
        self.grads = {}
        self.h = {}
        for key in self.params.keys():
            if self.params[key] is None:
                continue
            self.h[key] = np.zeros_like(self.params[key])
               
    def update(self, dparam1 = None, dparam2 = None):
        self.grads['param1'] = dparam1
        self.grads['param2'] = dparam2       
        for key in self.params.keys():
            if self.params[key] is None:
                continue
            self.h[key] += self.grads[key]*self.grads[key]
            self.params[key] -= self.lr*self.grads[key]/(np.sqrt(self.h[key]) + 1e-7)

class RMSprop:
    def __init__(self, param1 = None, param2 = None, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.params = {'param1':param1, 'param2':param2}
        self.grads = {}
        self.h = {}
        for key in self.params.keys():
            if self.params[key] is None:
                continue
            self.h[key] = np.zeros_like(self.params[key])
               
    def update(self, dparam1 = None, dparam2 = None):
        self.grads['param1'] = dparam1
        self.grads['param2'] = dparam2       
        for key in self.params.keys():
            if self.params[key] is None:
                continue
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate)*self.grads[key]*self.grads[key]
            self.params[key] -= self.lr*self.grads[key]/(np.sqrt(self.h[key]) + 1e-7)

class Adam:
    def __init__(self, param1 = None, param2 = None, lr = 0.001, beta1 = 0.9, beta2 = 0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.params = {'param1':param1, 'param2':param2}
        self.m = {}
        self.v = {}
        self.grads = {}
        for key in self.params.keys():
            if self.params[key] is None:
                continue
            self.m[key] = np.zeros_like(self.params[key])
            self.v[key] = np.zeros_like(self.params[key])
               
    def update(self, dparam1 = None, dparam2 = None):
        self.grads['param1'] = dparam1
        self.grads['param2'] = dparam2
        self.iter += 1
        lr_t  = self.lr*np.sqrt(1.0 - self.beta2**self.iter)/(1.0 - self.beta1**self.iter)        
        for key in self.params.keys():
            if self.params[key] is None:
                continue
            self.m[key] += (1 - self.beta1)*(self.grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2)*(self.grads[key]**2 - self.v[key])            
            self.params[key] -= lr_t*self.m[key]/(np.sqrt(self.v[key]) + 1e-7)