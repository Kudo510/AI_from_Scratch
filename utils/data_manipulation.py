import numpy as np
import math as m

## all the functions work with index of X - X[idx]
def shuffle_data(X, y, seed = None):
	''' random samples of dataset X and y'''
	if seed:
		np.random.seed(seed)
	idx=np.arange(X.shape[0])
	np.random.shuffle(idx)
	return X[idx],y[idx]


def train_test_split(X, y, test_size = 0.2, shuffle = True, seed = None):
	'''split data into train, test set'''
	if shuffle:
		X,y=shuffle_data(X,y,seed)
	split=len(y) - int(len(y)//(1/test_size))
	X_train,X_test=X[0:split],X[split:]
	y_train,y_test=y[0:split],y[split:]

	return X_train,X_test,y_train,y_test

def batch_iterator(X, y=None, batch_size=64):
	""" Simple batch generator """
	n_samples=X.shape[0]
	for i in range(0, n_samples,batch_size):
		begin,end= i,min(i+batch_size,n_samples)
		if y is not None:
			yield X[begin:end], y[begin:end]
		else:
			yield X[begin:end]
