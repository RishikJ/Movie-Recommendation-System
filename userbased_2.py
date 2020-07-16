# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 23:33:23 2020

@author: rishi
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sortedcontainers import SortedList

# load in the data
import os
if not os.path.exists('user2movie.pickle') or \
   not os.path.exists('movie2user.pickle') or \
   not os.path.exists('usermovie2rating.pickle') or \
   not os.path.exists('usermovie2rating_test.pickle'):
   import preprocess2dict


with open('user2movie.pickle', 'rb') as f:
  user2movie = pickle.load(f)

with open('movie2user.pickle', 'rb') as f:
  movie2user = pickle.load(f)

with open('usermovie2rating.pickle', 'rb') as f:
  usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.pickle', 'rb') as f:
  usermovie2rating_test = pickle.load(f)


N = np.max(list(user2movie.keys())) + 1
# the test set may contain movies the train set doesn't have data on
m1 = np.max(list(movie2user.keys()))
m2 = np.max([m for (u, m), r in usermovie2rating_test.items()])
M = max(m1, m2) + 1
print("N:", N, "M:", M)

if N > 10000:
  print("N =", N, "are you sure you want to continue?")
  print("Comment out these lines if so...")
  exit()


# to find the user similarities, you have to do O(N^2 * M) calculations!
# in the "real-world" you'd want to parallelize this
# note: we really only have to do half the calculations, since w_ij is symmetric
K = 25 # number of neighbors we'd like to consider
limit = 5 # number of common movies users must have in common in order to consider
neighbors = [] # store neighbors in this list
averages = [] # each user's average rating for later use
deviations = [] # each user's deviation for later use
for i in range(N):
  # find the 25 closest users to user i
  movies_i = user2movie[i]
  movies_i_set = set(movies_i)

  # calculate avg and deviation
  ratings_i = { movie:usermovie2rating[(i, movie)] for movie in movies_i }
  avg_i = np.mean(list(ratings_i.values()))
  dev_i = { movie:(rating - avg_i) for movie, rating in ratings_i.items() }
  dev_i_values = np.array(list(dev_i.values()))
  sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

  # save these for later use
  averages.append(avg_i)
  deviations.append(dev_i)

  sl = SortedList()
  for j in range(N):
    # don't include yourself
    if j != i:
      movies_j = user2movie[j]
      movies_j_set = set(movies_j)
      common_movies = (movies_i_set & movies_j_set) # intersection
      if len(common_movies) > limit:
        # calculate avg and deviation
        ratings_j = { movie:usermovie2rating[(j, movie)] for movie in movies_j }
        avg_j = np.mean(list(ratings_j.values()))
        dev_j = { movie:(rating - avg_j) for movie, rating in ratings_j.items() }
        dev_j_values = np.array(list(dev_j.values()))
        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

        # calculate correlation coefficient
        numerator = sum(dev_i[m]*dev_j[m] for m in common_movies)
        w_ij = numerator / (sigma_i * sigma_j)

        # insert into sorted list and truncate
        # negate weight, because list is sorted ascending
        # maximum value (1) is "closest"
        sl.add((-w_ij, j))
        if len(sl) > K:
          del sl[-1]

  # store the neighbors
  neighbors.append(sl)

  # print out useful things
  if i % 1 == 0:
    print(i)

def predict(i,m):
    numerator =0
    denominator=0
    
    for neg_w,j in neighbors[i]:
        try:
            numerator += -neg_w*deviations[i][j]
            denominator += abs(neg_w)
        except KeyError:
            pass
        if denominator ==0:
            prediction = averages[i]
        else:
            prediction = averages[i]+ numerator/denominator
        prediction = max(5,prediction)
        prediction - min(0.5,prediction)
        
        return prediction
        
train_predictions=[]
train_targets=[]

for (i,m),target in usermovie2rating.items():
    prediction = predict(i,m)
    
    train_predictions.append(i)
    train_targets.append(target)
    
test_prediction=[]
test_targets=[]

for (i,m),target in usermovie2rating_test.items():
    prediction = predict(i,m)
    
    test_prediction.append(prediction)
    test_targets.append(target)
    
def mse(pred,target):
    pred = np.array(pred)
    target = np.array(target)

    return np.mean((pred-target)**2)

print("trian mse:", mse(train_predictions,train_targets))

print("test mse", mse(test_prediction,test_targets))