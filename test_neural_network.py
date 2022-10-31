import numpy as np 
import os
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_operation import R_square,mean_squared_error,accuracy_score
from utils.data_preprocessing import min_max_scaler
from utils.data_manipulation import train_test_split
from deep_learning.optimizers import (GradientDescent,StochasticGradientDescent,Adagrad,RMSProp,Adam)
from deep_learning.models import NeuralNetwork
from deep_learning.loss_functions import SquareLoss
from deep_learning.layers import Dense,Activation

def get_data(file_name):
	dir_name="data"
	path=os.path.join(dir_name,file_name)
	data=pd.read_csv(path)
	return data

def preprocessing_data(data):
	features=["ON_STREAM_HRS", "AVG_DOWNHOLE_PRESSURE", "AVG_DOWNHOLE_TEMPERATURE","AVG_CHOKE_SIZE_P", "AVG_WHP_P", "AVG_WHT_P"]
	target=["BORE_OIL_VOL"]
	X=data[features].to_numpy()
	y=data[target].to_numpy()
	return X,y

def main():
	#load dataset
	X=np.loadtxt("data/input.txt")
	y=np.loadtxt("data/output.txt")
	# data=get_data("data.csv")
	# X,y=preprocessing_data(data)
	n_samples,n_features=X.shape

	X=min_max_scaler(X)

	## split two times one for testset another for validation set
	X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.1, shuffle = True, seed = 5)
	X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,test_size = 0.1, shuffle = True, seed = 5)

	validation_data=(X_val,y_val)
	print(n_features)

	GD = GradientDescent(0.001)
	SGD = StochasticGradientDescent(learning_rate = 0.001, momentum = 0.9)
	RMS = RMSProp(learning_rate=0.01,epsilon=1e-6, decay_rate=0.9)
	Adam_opt = Adam(learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-6)

	model=NeuralNetwork(optimizer=SGD,loss=SquareLoss,validation_data=validation_data)

	model.add(Dense(200, input_shape=(n_features,)))
	model.add(Activation('sigmoid'))
	model.add(Dense(100))
	model.add(Activation('tanh'))
	model.add(Dense(100))
	model.add(Activation('sigmoid'))
	model.add(Dense(2))
	model.add(Activation('linear'))


	train_err, val_err = model.fit(X_train,y_train, n_epochs = 50, batch_size=8)
	print(len(train_err))
	print(len(val_err))

	print("Training and validate errors", train_err[-1], val_err[-1])
	SGD_err = np.concatenate((np.array(train_err).reshape(-1,1), np.array(val_err).reshape(-1,1)), axis = 1)

	np.savetxt("errors/SGD_err.txt", SGD_err, delimiter = ',')

	y_train_pred = model.predict(X_train)
	print("===Training results===")
	print("R-square on y1",R_square(y_train_pred[0], y_train[0]))
	print("R-square on y2", R_square(y_train_pred[1], y_train[1]))
	print("Overal error on traning set",(mean_squared_error(y_train_pred[0], y_train[0]) + \
		mean_squared_error(y_train_pred[1], y_test[1]))/2)
	y_val_pred = model.predict(X_val)
	print("===Validation results===")
	print("R-square on y1", R_square(y_val_pred[0], y_val[0]))
	print("R-square on y2", R_square(y_val_pred[1], y_val[1]))
	print("Overal error on valiation set",(mean_squared_error(y_val_pred[0], y_val[0]) + \
		mean_squared_error(y_val_pred[1], y_val[1]))/2)

	y_pred = model.predict(X_test)
	print("===Testing results===")
	print("The portions of training is %0.2f , validation is %0.2f and testing data is %0.2f"\
	 %(len(y_train)/len(y)*100, len(y_val)/len(y)*100, len(y_test)/len(y)*100))
	print("Result on blind test samples")
	print("R2 value on the y1",R_square(y_pred[0], y_test[0]))
	print("R2 value on the y2", R_square(y_pred[1], y_test[1]))
	print("Overal blind test error", (mean_squared_error(y_pred[0], y_test[0]) + \
		mean_squared_error(y_pred[1], y_test[1]))/2 )


	# plt.plot(train_err, 'r', label = "training")
	# plt.plot(val_err, 'b', label = 'validation')
	# plt.xlabel("Iterations")
	# plt.ylabel("Error")
	# plt.legend()
	# plt.show()
	
	# plt.plot(np.arange(len(y_pred)), y_pred[:,0], 'r', label = 'y1 predict')
	# plt.plot(np.arange(len(y_pred)), y_test[:,0], 'b', label = 'y1 actual')
	# # plt.plot(np.arange(len(y_pred)), y_pred[:,1], 'g', label = 'y2 predict')
	# # plt.plot(np.arange(len(y_pred)), y_test[:,1], 'k', label = 'y2 actual')
	# plt.title("Result of blind test")
	# plt.legend()
	# plt.show()

	# plt.plot(np.arange(len(y_train_pred)), y_train_pred[:,0], 'r', label = 'y1 training predict')
	# plt.plot(np.arange(len(y_train)), y_train[:,0], 'b', label = 'y1 training actual')
	# # plt.plot(np.arange(len(y_train)), y_train_pred[:,1], 'g', label = 'y2 training predict')
	# # plt.plot(np.arange(len(y_train)), y_train[:,1], 'k', label = 'y2 training actual')
	# plt.title("Result of traing set")
	# plt.legend()
	# plt.show()

	# plt.plot(np.arange(len(y_val)), y_val_pred[:,0], 'r', label = 'y1 val predict')
	# plt.plot(np.arange(len(y_val)), y_val[:,0], 'b', label = 'y1 val actual')
	# # plt.plot(np.arange(len(y_val)), y_val_pred[:,1], 'g', label = 'y2 val predict')
	# # plt.plot(np.arange(len(y_val)), y_val[:,1], 'k', label = 'y2  val actual')
	# plt.title("Result of validation set")
	# plt.legend()
	# plt.show()


if __name__ == "__main__":
    main()