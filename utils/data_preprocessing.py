import numpy as np
import math as m

def min_max_scaler(X):
	#scaler data to range from 0 to 1
	assert isinstance(X, np.ndarray)
	mi = np.min(X, axis = 0)
	ma = np.max(X, axis = 0)
	return (X - mi)/(ma - mi)
	#return (X - X.min(axis = 0))/(X.max(axis = 0) - X.min(axis = 0))

def normalize_scaler(X):
    """ Scaling dataset X using mean normalization, divide by (max-min)"""
    assert isinstance(X, np.ndarray)
    return (X - X.mean(axis = 0))/(X.max(axis = 0) - X.min(axis = 0))

def standardize_scaler(X):
	'''Scaling dataset X using standard, divide by variance.
	The data will have 0 mean and variance of 1'''
	assert isinstance(X, np.ndarray)
	X_std = X
	mean = X.mean(axis=0)
	std = X.std(axis=0)
	X_std = (X - X.mean(axis=0)) / X.std(axis=0)
	return X_std

