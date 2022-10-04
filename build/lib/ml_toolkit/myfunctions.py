# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
import copy

import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')


def createQwerties(clusters=1, nPerClust=100, blur=0.5, centroids=[(0,0)],  draw=True):
 
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
    cType =['bs', 'ko', 'rs', 'go']
    fig = plt.figure(figsize=(5,5))
    for i in range(clusters):
      plt.plot(data[np.where(labels==i)[0],0],data[np.where(labels==i)[0],1],cType[i],alpha=.5)

    plt.title('The qwerties!')
    plt.xlabel('qwerty dimension 1')
    plt.ylabel('qwerty dimension 2')
    plt.show()

  return(data, labels)