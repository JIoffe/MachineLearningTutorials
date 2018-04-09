# -*- coding: utf-8 -*-
"""
Simple Neural Network implementation
Created on Fri Jan 12 12:39:08 2018

@author: jason
"""

import numpy as np

# X is our feature se
# four rows of features
X = np.array(([0,0],[1,1],[1,0],[0,1]), dtype=float)

# our known y for these given features
y = np.array(([0],[1],[1],[1]), dtype=float)

class NeuralNetwork(object):
    #Constructor
    def __init__(self):
        np.random.seed(2018)
        self.layer0 = np.random.randn(2 + 1, 1)
        return
    
    def predict(self, X):
        X0 = self.__add_bias(X)
        
        return self.__activation(np.dot(X0, self.layer0))
     
    def train(self, X, y, epochs = 100, learning_rate = 1):
        X0 = self.__add_bias(X)
        
        for epoch in range(epochs):             
            output = self.__activation(np.dot(X0, self.layer0))
            
            if((epoch+1) % 100 == 0):
                #using squared cost function
                total_error = 0.5 * np.square(y - output)
                average_total_error = np.mean(np.abs(total_error))
                print('Epoch {}: Avg. Error = {}'.format(epoch+1, average_total_error))
                
            """
                Follow the backpropagation:
                     output error
                     derivative of activation
                   X input
                   -------------------
                     contribution of weight to error
            """
            
            output_error = output - y
            layer_0_delta = self.__derive_activation(output) * output_error
            layer_0_adjustment = learning_rate * np.dot(X0.T, layer_0_delta)
            
            self.layer0 -= layer_0_adjustment
        return
    
    def __add_bias(self, X):
        X_biased = np.ones((X.shape[0], X.shape[1] + 1))
        X_biased[:, :-1] = X
        return X_biased
    
    def __derive_activation(self, x):
        return x * (1 - x)                   # Sigmoid
    def __activation(self, x):
        return 1 / (1 + np.exp(-x))          # Sigmoid

neural_net = NeuralNetwork()
neural_net.train(X, y, epochs = 1000, learning_rate = 1)

prediction = neural_net.predict(X)