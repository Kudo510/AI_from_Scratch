import numpy as np
import math

class KNN():
	def __init__(self, k):
		self.k = 3

	def vote(self, labels):
		dict_ = {"1": 0, "0": 0}

		for i in labels:
			if i == 1:
				dict_["1"] += 1
			else:
				dict_["0"] += 1
		if dict_['1'] > dict_['0']:
			return 1
		else:
			return 0
	def predict(self, X_test, X_train, y_train):
		y_pred = np.empty(X_test.shape[0])
		for i, test in enumerate(X_test):
			idx = np.argsort([np.linalg.norm(test - x) for x in X_train])[:self.k]
			k_nearest_neighbors = np.array([y_train[i] for i in idx])
			y_pred[i] = self.vote(k_nearest_neighbors)

		return y_pred

