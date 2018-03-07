# -*- coding: utf-8 -*-
"""
Machine Learning - Intent Recognition
    Basic example of using a neural net as a classifier
    to determine intent behind user speech. The dataset
    includes utterances for four "intents"
    
@author: Jason Ioffe
"""

import numpy as np
import pandas as pd

# Text Preprocessing Utilities
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Utilities for training
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Neural Network Utilities: Keras
import keras
from keras.models import Sequential
from keras.layers import Dense

class IntentRecognizer(object):
        def __init__(self):
            # Need to hold a list of pruned phrases
            self.corpus = []
            self.key_words = []
            
            # Use porter stemmer to simplify words to common root
            self.ps = PorterStemmer()
            
            # Remove 'stop words' 
            # - Except for a couple of words that are important to our case :)
            self.stopword_set = set(stopwords.words('english'))
            self.stopword_set.remove('on')
            self.stopword_set.remove('off')
            
        def predict(self, text):
            # First prune the phrase and count vectorize based on known key words
            text = self.__clean_phrase(text)
            cv = CountVectorizer(vocabulary=self.key_words)
            X = cv.fit_transform([text]).toarray()
            
            y_pred = self.classifier.predict(X)[0, :]
            y_pred = (y_pred > 0.5)

            try:
                return self.output_classes[list(y_pred).index(True)]
            except:
                return 'Unknown'
        
        def fit(self, dataset):
            n = dataset.shape[0]
            
            
            print('Pre-processing {} utterances...'.format(n))
            for i in range(n):
                utterance = self.__clean_phrase(dataset['utterance'][i])
                self.corpus.append(utterance)
                
            # Create the "Bag of Words" - Matrix of words
            cv = CountVectorizer()
            X = cv.fit_transform(self.corpus).toarray()
            
            # Extract the column names so we can test against them later
            feature_df = pd.DataFrame(X, columns=cv.get_feature_names())
            self.key_words = list(feature_df)
            print('Isolated {} key words:'.format(len(self.key_words)))
            print(self.key_words)
            
            # y represents the labeled output in our dataset
            # however, these are just strings. We need to encode them as categories
            y = dataset.iloc[:, 1].values
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(y)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

            # Save class mappings to refer to later when making predictions
            self.output_classes = list(label_encoder.classes_)
            
            one_hot_encoder = OneHotEncoder(sparse=False)
            y = one_hot_encoder.fit_transform(integer_encoded)

            
            # Split into training and test set - 3/4 training, 1/4 test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
            
            # The number of inputs to this network is going to be the number of key words we found
            # For simplicity, hidden layer size = input - output size
            n_inputs = len(self.key_words)
            n_outputs = len(set(dataset.iloc[:, 1].values))
            hidden_units = abs(n_inputs - n_outputs)
            
            self.classifier = Sequential()
            self.classifier.add(Dense(activation="relu", input_dim=n_inputs, units=hidden_units, kernel_initializer="uniform"))
            self.classifier.add(Dense(activation="relu", units=hidden_units, kernel_initializer="uniform"))

            self.classifier.add(Dense(activation="softmax", units=n_outputs, kernel_initializer="uniform"))
            
            # Compile the classifier so it can be trained and used!
            self.classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#
#            # Fitting the ANN to the Training set
            self.classifier.fit(X_train, y_train, batch_size=10, epochs=20)
            
            # How well did we do?
            score = self.classifier.evaluate(X_test, y_test, batch_size=10)
            print("Baseline Accuracy: %.2f%%" % (score[1] * 100))

        def __clean_phrase(self, s):
                raw_s = s
                # Only take in lower-case, alphanumeric characters
                s = re.sub('[^a-zA-Z]', ' ', s)
                s = s.lower()
                
                # Remove stop words and recombine
                s = s.split()
                s = [self.ps.stem(word) for word in s if not word in self.stopword_set]
                #s = [self.ps.stem(word) for word in s]
                s = ' '.join(s)  
                
                print('Cleaned: {} -> {}'.format(raw_s, s))
                return s
                
# First, we need data:
dataset = pd.read_csv('intent_dataset.tsv', header=None, names=['utterance', 'intent'], encoding = 'ISO-8859-1', delimiter = '\t', quoting = 3)

recognizer = IntentRecognizer()
recognizer.fit(dataset)