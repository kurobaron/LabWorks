from gan_layers import *
from collections import OrderedDict
import numpy as np
class Sumple:
	def __init__(self):
		self.params = [1, 2, 3]
		self.isSum = True

class Sumple2:
	def __init__(self):
		self.params = np.array([1, 2, 3])
		self.isSum = True
		self.opt = Sum(self.params, self.isSum)

class Comb:
	def __init__(self, aa, bb):
		self.a = aa
		self.b = bb
		self.a.params[0] = 100
		self.a.isSum = False
		self.b.params.reverse()

class Sum:
	def __init__(self, cc, dd):
		self.c = cc
		self.d = dd
		self.c *= 2

A = Sumple2()
print(A.params)