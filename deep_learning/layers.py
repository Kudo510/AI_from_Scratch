import numpy as np 
import math
from deep_learning.initializations import constant_value,xavier_uniform
from deep_learning.activations import Linear, Sigmoid, Tanh, LeakyRelu, Softmax, Relu
import copy

class Dense():
	"""A fully-connected NN layer.
    Parameters:
    -----------
    n_units: int
        The number of neurons in the layer.
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying
        the number of features of the input. Must be specified if it is the first layer in
        the network.
    """
	def __init__(self,n_units,input_shape=None):
		self.n_units=n_units
		self.input_shape=input_shape
		self.weight=None
		self.bias=None
		self.weight_optimizer=None 
		self.bias_optimizer=None
		self.layer_input=None

	def set_input_shape(self,shape):
		self.input_shape=shape

	def initialize(self):
		self.weight=xavier_uniform(shape=(self.input_shape[0],self.n_units)) ##
		self.bias=constant_value(0,shape=(self.n_units,))

	def set_optimizers(self, optimizer):
		self.weight_optimizer = copy.copy(optimizer)
		self.bias_optimizer = copy.copy(optimizer)

	def forward(self,X):
		self.layer_input=X
		return X@self.weight+self.bias

	def backward(self,gradient):
		# Calculate gradient w.r.t layer weights
		grad_weight=self.layer_input.T@gradient
		grad_bias=np.sum(gradient,axis=0)
		# Update the layer weights
		self.weight=self.weight_optimizer.update(self.weight, grad_weight)
		self.bias=self.bias_optimizer.update(self.bias,grad_bias)

		# Update gradient
        # Calculated based on the weights used during the forward pass
		gradient=gradient@self.weight.T
		return gradient
		
	def output_shape(self):
		return(self.n_units,)

activation_functions = {
	'relu': Relu,
	'sigmoid': Sigmoid,
	'leaky_relu': LeakyRelu,
	'tanh': Tanh,
	'linear': Linear,
	'softmax': Softmax
}

class Activation():
	def __init__(self, activation_name):
		self.activation_class = activation_functions[activation_name]()
		self.input_shape=None
		self.layer_input=None

	def set_input_shape(self, shape):
		self.input_shape = shape

	def forward(self, _input):
		self.layer_input = _input
		return self.activation_class.activate(_input)

	def backward(self, gradient):
		return gradient * self.activation_class.derivative(self.layer_input)

	def output_shape(self):
		return self.input_shape