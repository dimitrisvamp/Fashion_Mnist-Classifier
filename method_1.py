import sys 
import numpy as np
import math 
import struct
import operator
import tensorflow as tf 
from keras.datasets import fashion_mnist

predictions = []


def EuclideanDistance(input1,input2,length):
  dist = 0
  for i in range(length-1):
    dist = dist + (int(input1[i])-int(input2[i]))**2
    euclidean_distance = math.sqrt(dist)

   return euclidean_distance


def read_file(name):
  with open(name,'rb') as file :
    zero,data_type,dims = struct.unpack('>HBB',file.read(4))
    shape = tuple(struct.unpack('>I',file.read(4))[0] for d in range(dims))

  return np.frombuffer(file.read(),dtype=np.uint8).reshape(shape)


def search_neighbours(train_matrix,test_matrix,k):
  dist_of_all = [] # distance of all neigbhours
  for i in range(len(train_matrix)):
    calc_dist = EuclideanDistance(test_matrix,train_matrix[i,1:785],len(train_matrix))
    dist_of_all.append((train_matrix[i],calc_dist))

  # Sorting the distance array 
  dist_of_all.sort(key=operator.itemgetter(1))

  final_neigbhours = []
  for i in range(k):
    final_neigbhours.append(dist_of_all[i][0])

  return final_neigbhours

def pick_best_neigbhour(best):
  check_count = {}
  for i in range(len(best)):
    occ = best[i][0]
    if occ in check_count:
      check_count[occ] = check_count[occ] + 1
    else:
      check_count[occ] = 1
    
  best_neigbhour = sorted(check_count.items(),key=operator.itemgetter(1),reverse=True)
  return best_neigbhour[0][0]
print("Loading train Data \n")
get_train = read_file("train-images-idx3-ubyte")
train_data = np.reshape(get_train,(60000,28*28))
train_label = read_file("train-labels-idx1-ubyte")
print("Loading test data \n")
get_test = read_file("t10k-images-idx3-ubyte")
test_data = np.reshape(get_test,(10000,28*28))
test_label = read_file("t10k-labels-idx1-ubyte")
print("taking k values")
get_k=sys.argv[1]
k= int(get_k)
print(k)
print("Searching for Neighbors")
for i in range(len(test_data)):
  find_neigbhours = search_neighbours(train_data,test_data[i],k)

  res = pick_best_neigbhour(find_neigbhours)

TP = 0
print("Calculating accuracy")
for i in range(len(test_data)):
  if test_data[i][0] == predictions[i]:
    TP = TP + 1
accurancy=(TP/float(len(test_matrix)))*100.0
print('Accuracy: ' + repr(accuracy) + '%')

"""
(X_train_data,y_train_data) , (X_test_data,y_test_data) = fashion_mnist.load_data() 
print("Shape of x_train: {}".format(X_train_data.shape))
print("Shape of y_train: {}".format(y_train_data.shape))
print()
print("Shape of x_test: {}".format(X_test_data.shape))
print("Shape of y_test: {}".format(y_test_data.shape))
"""