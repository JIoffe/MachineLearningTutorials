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
y = np.array(([0],[1],[0],[0]), dtype=float)

class NeuralNetwork(object):
    #Constructor
    def __init__(self):
        np.random.seed(2018)
        self.w1 = np.random.randn(2,1)
        return
    
    def predict(self, X):
        return self.__activation(np.dot(X, self.w1))
    
    def train(self, X, y, epochs = 100, learning_rate = 1):
        for epoch in range(epochs):
            output = self.__activation(np.dot(X, self.w1))
            
            output_error = y - output
            print(np.mean(np.abs(output_error)))
            activation_error = output_error * self.__derive_activation(output)
            weight_error = np.dot(X.T, activation_error)
            
            self.w1 += learning_rate * weight_error
        return
    
    def __derive_activation(self, x):
        return x * (1 - x)                   # Sigmoid
    def __activation(self, x):
        return 1 / (1 + np.exp(-x))          # Sigmoid

neural_net = NeuralNetwork()
neural_net.train(X, y, epochs = 1000, learning_rate = 1)