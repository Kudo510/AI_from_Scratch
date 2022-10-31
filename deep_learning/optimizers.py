import numpy as np

class GradientDescent():
	def __init__(self,learning_rate=0.01):
		self.learning_rate=learning_rate
		self.weight_update=None
		
	def update(self,weight,gradient):
		if self.weight_update is None:
			self.weight_update=np.zeros(np.shape(weight)) ## initialize the weight_update having the same shape as weight
		self.weight_update=gradient
		return weight- self.learning_rate*weight_update


class StochasticGradientDescent():
	def __init__(self,learning_rate=0.01,momentum=0.09):
		self.learning_rate=learning_rate
		self.momentum=momentum
		self.weight_update=None

	def update(self,weight,gradient):
		if self.weight_update is None:
			self.weight_update=np.zeros(np.shape(weight))
		self.weight_update= self.momentum*self.weight_update+(1-self.momentum) *gradient
		return weight- self.learning_rate*self.weight_update

class Adagrad():
	def __int__(self,learning_rate=0.01,epsilon=1e-6):
		self.learning_rate=learning_rate
		self.epsilon=epsilon
		self.G=None

	def update(self,weight,gradient):
		if self.G is None:
			self.G=np.zeros(np.shape(weight))
		self.G +=np.square(gradient)
		return weight- self.learning_rate*gradient/np.sqrt(self.G+self.epsilon)


class RMSProp():
	def __init__(self,learning_rate=0.01,epsilon=1e-6, decay_rate=0.9):
		self.learning_rate=learning_rate
		self.epsilon=epsilon
		self.grad_squared=None
		self.decay_rate=decay_rate

	def update(self,weight,gradient):
		if self.grad_squared is None:
			self.grad_squared=np.zeros(np.shape(weight))
		self.grad_squared=self.decay_rate*self.grad_squared+(1- self.decay_rate)*np.square(gradient)
		return weight- self.learning_rate*gradient/np.sqrt(self.grad_squared+self.epsilon)

class Adam():
	#Adam with beta1 = 0.9, beta2 = 0.999, and learning_rate = 1e-3, 5e-4, 1e-4 is a great starting point for many models!
	def __init__(self,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-6):
		self.learning_rate=learning_rate
		self.beta1=beta1
		self.beta2=beta2
		self.moment1=None
		self.moment2=None
		self.time_step=0
		self.epsilon=epsilon

	def update(self,weight,gradient):
		if self.moment1 is None:
			self.moment1=np.zeros(np.shape(weight))
		if self.moment2 is None:
			self.moment2=np.zeros(np.shape(weight))

		#at t (time_step)= 0 gradient = positive limitness -so need to avoid that by start at t=1
		self.time_step +=1 		

		self.moment1=self.beta1*self.moment1 + (1- self.beta1)*gradient
		self.moment2=self.beta2*self.moment2+(1- self.beta2)*np.power(gradient,2)

		return weight- self.learning_rate*self.moment1/(np.sqrt(self.moment2)+self.epsilon)



