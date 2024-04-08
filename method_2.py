import sys 
import numpy as np
import time
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.datasets import fashion_mnist
from sklearn import preprocessing
from sklearn import metrics



from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from collections import Counter 
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import InputLayer


start = time.time()
(Xtrain,ytrain),(Xtest,ytest) = fashion_mnist.load_data()

Xtrain = Xtrain.astype('float32')
Xtest = Xtest.astype('float32')
Xtrain=Xtrain/255.0
Xtest=Xtest/255.0


#Reshaping data 
Xtest=Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
Xtrain =Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])

get_hidden_layer = sys.argv[1]
get_k = sys.argv[2]
hidden_layer = int(get_hidden_layer)
k = int(get_k)

if (hidden_layer == 1):
	print("Hidden layer:",hidden_layer)
	NN = MLPClassifier(hidden_layer_sizes=(500,), solver='sgd')
	
	NN.fit(Xtrain,ytrain)
	
	y_pred = NN.predict(Xtest)
	
	NN_f1 = metrics.f1_score(ytest,y_pred,average="weighted")
	accuracy_NN = metrics.accuracy_score(ytest,y_pred)


	
	nn = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(10)])
	nn.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
	nn.fit(Xtrain,ytrain, epochs=10)
	probability_nn = tf.keras.Sequential([nn,  tf.keras.layers.Softmax()]) 
	prediction = probability_nn.predict(Xtest)

	print("F1 score :",(NN_f1)*100,"%")
	print("Accuracy :",(accuracy_NN)*100,"%")
	print("The probability for each label: ",prediction[k])
	end = time.time()
	print("Execution time:",end-start)
	
elif (hidden_layer == 2):
   	
	print("Hidden layer:",hidden_layer)
	NN = MLPClassifier(hidden_layer_sizes=(500,200,), solver='sgd')
	
	NN.fit(Xtrain,ytrain)
	
	y_pred = NN.predict(Xtest)
	
	NN_f1 = metrics.f1_score(ytest,y_pred,average="weighted")
	accuracy_NN = metrics.accuracy_score(ytest,y_pred)


	
	nn = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='sigmoid'),
    tf.keras.layers.Dense(10)])
	nn.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
	nn.fit(Xtrain,ytrain, epochs=10)
	probability_nn = tf.keras.Sequential([nn,  tf.keras.layers.Softmax()]) 
	prediction = probability_nn.predict(Xtest)

	print("F1 score :",(NN_f1)*100,"%")
	print("Accuracy :",(accuracy_NN)*100,"%")
	print("The probability for each label: ",prediction[k])
	end = time.time()
	print("Execution time:",end-start)
else:
	print("Error in input you have to specify the number of hidden layers!")