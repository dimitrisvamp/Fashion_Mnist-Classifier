import sys 
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.datasets import fashion_mnist
from sklearn import preprocessing
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from collections import Counter 
from sklearn.metrics import accuracy_score


def cosine_kernel(dist1,dist2):
	return cosine_similarity(dist1,dist2)

def gaussian_kernel(X1,X2,sigma=0.1):
	gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
	for i, x1 in enumerate(X1):
		for j, x2 in enumerate(X2):
			x1 = x1.flatten()
			x2 = x2.flatten()
			gram_matrix[i, j] = np.exp(- np.sum( np.power((x1 - x2),2) ) / float( 2*(sigma**2) ) )
	return gram_matrix

start = time.time()
(Xtrain,ytrain),(Xtest,ytest) = fashion_mnist.load_data()

Xtrain = Xtrain.astype('float32')
Xtest = Xtest.astype('float32')
Xtrain=Xtrain/255.0
Xtest=Xtest/255.0


#Reshaping data 
Xtest=Xtest.reshape(Xtest.shape[0], Xtest.shape[1] * Xtest.shape[2])
Xtrain =Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2])

kernel = sys.argv[1]
if kernel == 'linear':
	print("SVM - kernel:linear")
	svm_classifier = SVC(C=1,kernel='linear',gamma="auto")
	svm_classifier.fit(Xtrain,ytrain)
	y_pred = svm_classifier.predict(Xtest)
	f1_score_svm = metrics.f1_score(ytest,y_pred,average="weighted")
	accuracy_svm = metrics.accuracy_score(ytest,y_pred)
	
	print("F1 score:",f1_score_svm)
	print("Accuracy :",accuracy_svm)
elif kernel == 'gaussian':
	print("SVM - kernel:gaussian")
	svm_classifier = SVC(C=1,kernel=gaussian_kernel)
	#svm_classifier = SVC(C=1,kernel='rbf',gamma="auto")
	svm_classifier.fit(Xtrain,ytrain)
	y_pred = svm_classifier.predict(Xtest)
	f1_score_svm = metrics.f1_score(ytest,y_pred,average="weighted")
	accuracy_svm = metrics.accuracy_score(ytest,y_pred)
	
	print("F1 score:",f1_score_svm)
	print("Accuracy :",accuracy_svm)
elif kernel == 'cosine':
	print("SVM - kernel:cosine")
	svm_classifier = SVC(C=1,kernel=cosine_kernel,gamma="auto")
	svm_classifier.fit(Xtrain,ytrain)
	y_pred = svm_classifier.predict(Xtest)
	f1_score_svm = metrics.f1_score(ytest,y_pred,average="weighted")
	accuracy_svm = metrics.accuracy_score(ytest,y_pred)
	
	print("F1 score:",f1_score_svm)
	print("Accuracy :",accuracy_svm)
else:
	print("Error , you have to specify which kernel operation you want to use!")