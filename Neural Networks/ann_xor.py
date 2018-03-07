# -*- coding: utf-8 -*-
"""
Simple XOR gate implementation with a neural net
@author: jason-ioffe
"""
import numpy as np

class XORNeuralNetwork(object):
    def __init__(self, input_size): 
        np.random.seed(10)
        #Add an extra input for bias
        self.layer_0 = np.random.randn(input_size + 1, 2)
        self.layer_1 = np.random.randn(2 + 1, 1)

    def predict(self, X):
        #add a column of 1s to simulate bias
        X0 = self.__add_bias(X)
        
        # The activation of input to the first layer
        # is the input for the second layer
        X1 = self.__add_bias(self.__activation(np.dot(X0, self.layer_0)))
        return self.__activation(np.dot(X1, self.layer_1))
        
    def train(self, X, y, epochs=100, learning_rate=1):
        for i in range(epochs):
            X0 = self.__add_bias(X)
            X1 = self.__add_bias(self.__activation(np.dot(X0, self.layer_0)))
    
            prediction = self.__activation(np.dot(X1, self.layer_1))
    
            if((i+1) % 100 == 0):
                #using squared cost function
                total_error = 0.5 * np.square(y - prediction)
                average_total_error = np.mean(np.abs(total_error))
                print('Epoch {}: Avg. Error = {}'.format(i+1, average_total_error))
                
            #Compute gradients for adjustment
            output_error = prediction - y
            layer_1_delta = output_error * self.__derive_activation(prediction)
            layer_1_adjustment = np.dot(X1.T, layer_1_delta)
            
            #layer 0s impact is based on the weights of layer1 - except for the bias!
            layer_0_error = np.dot(layer_1_delta, self.layer_1[:-1].T)
            layer_0_delta = layer_0_error * self.__derive_activation(X1[:,:-1])
            layer_0_adjustment = np.dot(X0.T, layer_0_delta)
            
#            # Weights need to be adjusted all at once
#            # Although tempting, do not update weights while backpropagating!
            self.layer_0 -= learning_rate * layer_0_adjustment
            self.layer_1 -= learning_rate * layer_1_adjustment
    def __add_bias(self, X):
        X_biased = np.ones((X.shape[0], X.shape[1] + 1))
        X_biased[:, :-1] = X
        return X_biased
    
    def __activation(self, x):
        # Sigmoid
        return 1 / (1 + np.exp(-x))
    
    def __derive_activation(self, x):
        return x * (1 - x)
        

# the training set assumes an XOR relationship
X = np.array(([0,0],[1,1],[1,0],[0,1]), dtype=float)
y = np.array(([0],[0],[1],[1]), dtype=float)

neural_net = XORNeuralNetwork(2)
neural_net.train(X, y, epochs = 1000, learning_rate = 2)
pred = neural_net.predict(X)