# -*- coding: utf-8 -*-
"""
All purpose
Created on Wed Jan 10 12:42:41 2018
@author: jason
"""
import numpy as np

class NeuralNetwork(object):
    def __init__(self):
        self.neuron_layers = []
        np.random.seed(1)
        
    def add_layer(self, input_size, output_size):
        self.neuron_layers.append(np.random.randn(input_size, output_size))
    
    def predict(self, X):
        #imagine that X is the activation of the input layer
        activation = X
        for layer in self.neuron_layers:
            activation = self.__activation(np.dot(activation, layer))
        return activation
    
    def train(self, X, y, epochs = 100, learning_rate = 1):
        for ep in range(epochs):
            #store all the levels of activations in the forwards pass
            activations = []
            for i in range(len(self.neuron_layers)):
                previous_activation = X if i == 0 else activations[i - 1]
                activations.append(self.__activation(np.dot(previous_activation, self.neuron_layers[i])))
                
            #Go backwards, step by step
            #First get error of the entire system
            prediction = activations[-1]
            output_error = y - prediction
            
            average_batch_error = np.mean(np.abs(output_error))
            print('Average Output Error: {}', average_batch_error)

            # Initialize arrays with 0s so we can traverse backwards
            # Need to apply all weight changes at once
            layer_deltas = [0] * len(self.neuron_layers)
            weight_deltas = [0] * len(self.neuron_layers)

            #once we have the outer layer, traverse backwards computing the contribution
            #but update the weights all at once
            for i in reversed(range(len(self.neuron_layers))):
                if(i == len(self.neuron_layers) - 1):
                    layer_error = output_error
                else:
                    layer_error = np.dot(layer_deltas[i+1], self.neuron_layers[i+1].T)
                    
                layer_deltas[i] = layer_error * self.__derive_activation(activations[i])
                previous_activation = X if i == 0 else activations[i - 1]
                weight_deltas[i] = np.dot(previous_activation.T, layer_deltas[i])
#                
#            #Now we can adjust!
            for i in range(len(weight_deltas)):
                self.neuron_layers[i] += learning_rate * weight_deltas[i]
            
    def __derive_activation(self, x):
        return x * (1 - x)
    def __activation(self, x):
        # Sigmoid
        return 1 / (1 + np.exp(-x))
    
X = np.array(([0,0],[1,1],[1,0],[0,1]), dtype=float)
y = np.array(([0],[1],[0],[0]), dtype=float)
net = NeuralNetwork()
net.add_layer(2, 3)
net.add_layer(3,1)

net.train(X, y, epochs = 10000)

p = net.predict(X)

test = [0] * 5
test[2] = [4124]
test[-1] = np.random.randn(2, 3)
for i in reversed(range(5)):
    print(i)