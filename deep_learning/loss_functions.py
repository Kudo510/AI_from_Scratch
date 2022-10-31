import numpy as np
import math

class SquareLoss():
	def loss(self, y, predicted_y):
		length=len(y)		
		return 1/(2*length)*np.power((y- predicted_y),2)

	def derivative(self,y,predicted_y):
		length=len(y)
		return 1/length*(predicted_y-y)

	def RMSE(self,y,predicted_y):
		return np.sqrt(np.mean(np.power((y- predicted_y),2)))

class CrossEntropy():
	def loss(self,y,p):
		#avoid devision by zero
		p=np.clip(p,1e-15,1-1e-15)
		return -y*np.log(p) - (1-y)*np.log(1-p)
		
	def derivative(self,y,p):
		#avoid devision by zero
		p=np.clip(p,1e-15,1-1e-15)
		return -y/p + (1-y)/(1-p)

