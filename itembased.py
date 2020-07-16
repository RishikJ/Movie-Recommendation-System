# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:26:04 2020

@author: rishi
"""

import pickle
from sortedcontainers import SortedList
import numpy as np


with open('user2movie.pickle', 'rb') as f:
  user2movie = pickle.load(f)

with open('movie2user.pickle', 'rb') as f:
  movie2user = pickle.load(f)

with open('usermovie2rating.pickle', 'rb') as f:
  usermovie2rating = pickle.load(f)

with open('usermovie2rating_test.pickle', 'rb') as f:
  usermovie2rating_test = pickle.load(f)



N = np.max(list(user2movie.keys()))+1

m1 = np.max(list(movie2user.keys()))
m2 = np.max(list(m for (u,m) in usermovie2rating_test.keys()))

M = max(m1,m2)+1

if M >2000:
    print("are you sure you want to continue")
    exit()
    

K = 20 #number of neighbours that we want
limit = 5
neighbours=[]
averages= []
deviations = []


for i in range(M): #iterating through the movies
    users_i = movie2user[i]
    users_i_set = set(users_i)
    
    ratings_i = {user :usermovie2rating[(user,i)] for user in users_i}
    avg_i = np.mean(list(ratings_i.values()))
    dev_i = {user:(rating_i-avg_i) for (user,rating_i) in ratings_i.items()}
    dev_i_values = np.array(list(dev_i.values()))
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))
    
    averages.append(avg_i)
    deviations.append(dev_i)
    sl = SortedList()
    
    for j in range(M):
        
        if j!=i:            
            users_j = movie2user[j]
            users_j_set = set(users_j)
            
            common_users = users_i_set.intersection(users_j_set)
            
            if common_users>limit:
                #calculating the average and deviation
                ratings_j = {user:usermovie2rating[(user,j)] for user in users_j}
                avg_j = np.mean(list(ratings_j.values()))
                dev_j = {user:(rating_j-avg_j) for (user,rating_j) in ratings_j.items()}
                dev_j_values = np.array(list(dev_j.values()))
                sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))
                
                #calculating the correlation coefficient
                numerator = sum(dev_i[u]*dev_j[u] for u in common_users)
                wij = numerator/(sigma_i*sigma_j)\
                
                sl.append(-wij,j)
                if len(sl)>K:
                    del sl[-1]
        
    neighbours.append(sl)
    print(i)
    
def predict(i,u):
    numerator = 0
    denominator = 0
    
    for neg_w,j in neighbours[i]:
        try:
            numerator += -neg_w*deviations[j][u]
            denominator += abs(neg_w)
        except KeyError:
            pass
        
        if denominator ==0:
            prediction = averages[i]
        else:
            prediction = numerator/denominator +averages[i]
            
        prediction=max(prediction,5)
        prediction = min(0.5,prediction)
    return prediction

train_predictions = []
train_target=[]

for (u,m),target in usermovie2rating.items():
    prediction = predict(m,u)
    train_predictions.append(prediction)
    train_target.append(target)
 
test_predictions = []
test_target=[]

for (u,m),target in usermovie2rating_test.items():
    prediction = predict(m,u)
    test_predictions.append(prediction)
    test_target.append(target)

def mse(pred,target):
    pred = np.array(pred)
    target = np.array(target)

    return np.mean((pred-target)**2)

print("trian mse:", mse(train_predictions,train_target))

print("test mse", mse(test_predictions,test_target))
    
    
    
        
    
    
                
                
                
            
            