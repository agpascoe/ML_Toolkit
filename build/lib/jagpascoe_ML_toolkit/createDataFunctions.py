# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import copy

import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')


def createQwerties(clusters=1, nPerClust=100, blur=0.5, centroids=[[0,0]],  draw=True):
  '''
  This function builds a set of data in up to 4 qwerties or clusters, each one with nPerclust points.
  The centroids of the qwerties is given by a np array of point (elements). Blur is a parameter that close up or not the qwerties,
  playing as a noice to each point.

   for example, to create a set of 200 points in two qwerties, with some intersection:
  createQwerties( clusters=2,
                   nPerClust = 100,
                  blur = 1.5,
                  centroids = np.array([[3,3],[5,3]]),
                  draw = True)
  
  The draw parameter indicates if the functions displays a graph of the querties distribution ot not.
  This function returns two datasets of points, and labels.
  
  '''
  # create data
  data_np = np.reshape(np.zeros(clusters*2*nPerClust),(nPerClust*clusters,2))
  labels_np =np.zeros(clusters*nPerClust) 

  for i in range(clusters):
    for j in range(nPerClust):
      data_np[i*nPerClust + j,0] = centroids[i,0] + (np.random.randn(1)-0.5)*blur
      data_np[i*nPerClust + j,1] = centroids[i,1] + (np.random.randn(1)-0.5)*blur
      labels_np[i*nPerClust + j] = i 

 
  # convert to a pytorch tensor
  data = torch.tensor(data_np).float()
  labels = torch.tensor(labels_np).long() # note: "long" format for CCE
 
  if(draw):
    # show the data
    cType =['bs', 'ko', 'rs', 'go', 'co', 'ms']
    fig = plt.figure(figsize=(5,5))
    for i in range(clusters):
      plt.plot(data[np.where(labels==i)[0],0],data[np.where(labels==i)[0],1],cType[i],alpha=.5)

    plt.title('The qwerties!')
    plt.xlabel('qwerty dimension 1')
    plt.ylabel('qwerty dimension 2')
    plt.show()

  return(data, labels)


def splitData(partitions, batch_size, data, labels, verbose=False):
  '''
  This function splits data and labels that come in np.array or tensor, and return three sets of data in
  pytorch dataloaders with batches of batch_size. For the splitting, uses skilearn train_test_split function.

  partitions is an array of [%trainning, %valid], where %test set is the remaining to get 100%
  '''
  
  # split the data
  
  from torch.utils.data import DataLoader,TensorDataset
  from sklearn.model_selection import train_test_split
  
  #data = torch.tensor(data).float()
  #labels = torch.tensor(labels).float()

  train_data, devtest_data, train_labels,devtest_labels = train_test_split(data, labels, train_size=partitions[0],shuffle=True)#randomized

  # now split the devtest data
  dev_data,test_data, dev_labels,test_labels = train_test_split(devtest_data, devtest_labels, train_size=partitions[1]/(1-partitions[0]),shuffle=True)
  
  # print out the sizes
  if verbose:
    print('   Total data size: ' + str(data.shape) + '\n')
    print('Training data size: ' + str(train_data.shape))
    print('Trainning Label data size: ' + str(train_labels.shape))
    print('Dev data size: ' + str(dev_data.shape))
    print('Dev Label data size: ' + str(dev_labels.shape))
    print('Dev test size: ' + str(test_data.shape))
    print('Dev test data size: ' + str(test_labels.shape))

  # then convert them into PyTorch Datasets (note: already converted to tensors)
  train_data = TensorDataset(train_data, train_labels)
  dev_data   = TensorDataset(dev_data,dev_labels)
  test_data  = TensorDataset(test_data,test_labels)
 
 
  # finally, translate into dataloader objects
  train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=True,drop_last=True)
  dev_loader   = DataLoader(dev_data,batch_size=dev_data.tensors[0].shape[0])
  test_loader  = DataLoader(test_data,batch_size=test_data.tensors[0].shape[0])

  return(train_loader, dev_loader, test_loader)