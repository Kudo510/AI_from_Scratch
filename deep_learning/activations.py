import numpy as np
import math


# Collection of activation functions
# Reference: https://en.wikipedia.org/wiki/Activation_function

class Sigmoid():
	def activate(self,x):
		return 1/(1+np.exp(-x))
	def derivative(self,x):
		return self.activate(x)*(1-self.activate(x))

class Tanh():
	def activate(self,x):
		return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
	def derivative(self,x):
		return (1- self.activate(x)**2)

class Linear():
	def activate(self,x):
		return x 
	def derivative(self,x):
		return 1

class Relu():
	def activate(self,x):
		return (x+abs(x))/2
	def derivative(self,x):
		return 1.*(x>0)
		
class LeakyRelu():
	def __init__(self,alpha=0.01):
		self.alpha=alpha
	def activate(self,x):
		res=np.where(x<0,x*self.alpha,x)
		return res
	def derivative(self,x):
		res=np.where(x<0,self.alpha,1)
		return res

class Softmax():
	def activate(self, x):
		e_x = np.exp(x - np.max(x))
		return e_x / e_x.sum()

	def derivative(self, x):
		return self.activate(x) * (1 - self.activate(x))


		