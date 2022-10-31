import numpy as np
from utils.data_manipulation import batch_iterator


class NeuralNetwork():
    """Neural Network. Deep Learning base model.
    Parameters:
    -----------
    optimizer: class
        The weight optimizer that will be used to tune the weights in order of minimizing
        the loss.
    loss: class
        Loss function used to measure the model's performance. SquareLoss or CrossEntropy.
    validation: tuple
        A tuple containing validation data and labels (X, y)
    """
    def __init__(self, optimizer, loss, validation_data = None):
        self.optimizer=optimizer
        self.loss=loss
        self.errors={"training":[],"validation":[]}
        self.loss_function=loss()
        self.validation_set=None
        self.iterations=0
        self.layers=[]
        if validation_data is not None:
        	X,y=validation_data
        	self.validation_set={"X":X,"y":y}

    def add(self,layer):
        '''This method will add a layer to current neural network architechture'''
        # If this is not the first layer added then set the input shape
        # to the output shape of the last added layer
        if self.layers:
        	layer.set_input_shape(shape=self.layers[-1].output_shape())
        # If the layer has weights that needs to be initialized 
        if hasattr(layer,"initialize"):
        	layer.initialize()
        if hasattr(layer,"set_optimizers"):        ###dm set_optimizers with s
        	layer.set_optimizers(self.optimizer)   ###dm set_optimizers with s 	
        	layer.iterations=self.iterations

        self.layers.append(layer)

    def _forward_pass(self,X):
    	layer_output=X
    	for layer in self.layers:
    		layer_output=layer.forward(layer_output)
    	return layer_output

    def _backward_pass(self,gradient):
    	for layer in reversed(self.layers):
    		gradient=layer.backward(gradient)

    def train(self,X,y):
    	#Training the model on single batch size
    	y_predict=self._forward_pass(X)
    	loss=np.mean(self.loss_function.loss(y,y_predict))
    	accuracy=self.loss_function.RMSE(y,y_predict)
    	gradient=self.loss_function.derivative(y,y_predict)

    	self._backward_pass(gradient)
    	return loss, accuracy

    def test(self,X,y):
    	#Testing model in the single batch of training process
    	y_predict=self._forward_pass(X)
    	loss=np.mean(self.loss_function.loss(y,y_predict))
    	accuracy=self.loss_function.RMSE(y,y_predict)

    	return loss,accuracy

    def fit(self,X,y, n_epochs, batch_size):
    	#Method to fit neural networks with the data samples
        for i in range(n_epochs):
            self.iterations +=1
            batch_errors=[]
            for X_batch,y_batch in batch_iterator(X,y,batch_size):
            	loss,accuracy=self.train(X_batch,y_batch)
            	batch_errors.append(loss)
            self.errors["training"].append(np.mean(batch_errors))
            if self.validation_set is not None:
            	val_loss,accuracy=self.test(self.validation_set["X"],self.validation_set["y"])
            	self.errors["validation"].append(val_loss)
        return self.errors["training"], self.errors["validation"]

    def predict(self,X):
    	return self._forward_pass(X)



