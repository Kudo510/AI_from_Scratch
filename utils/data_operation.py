import numpy as np

def mean_squared_error(y_true, y_pred):
	"""Calculate the mean squared error between y_true and y_pred"""
	return np.mean(np.power(y_true - y_pred, 2))

def accuracy_score(y_true, y_pred):
	""" Compare y_true to y_pred and return the accuracy """
	count  = 0
	for i in range(len(y_true)):
		if y_true[i] == y_pred[i]:
			count += 1
	return count/len(y_true)

def R_square(y_pred, y_actual):
	'''calculate the coefficient of determination R2'''
	y_bar = np.mean(y_actual)
	SS_tot = np.sum(np.power(y_actual - y_bar,2))
	SS_res = np.sum(np.power(y_pred - y_actual,2))
	return 1 - SS_res/SS_tot