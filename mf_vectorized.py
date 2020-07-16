# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 01:50:11 2020

@author: rishi
"""

import pickle 
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.utils import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    
N = np.max(list(user2movie.keys()))+1
m1 = np.max(list(movie2user.keys()))
m2 = max([movie for (user,movie),r in usermovie2rating_test.items()])
M=  max(m1,m2)+1

print("converting")

user2movierating={}
for i,movies in user2movie.items():
    r = np.array([usermovie2rating[(i,j)]] for j in movies)
    user2movierating[i]=(movies,r)
movie2userrating={}
for j,users in movie2user.items():
    r = np.array([usermovie2rating[(i,j)]] for i in users)
    movie2userrating[j]=(users,r)
    
movie2userrating_test={}
for (i,j),r in usermovie2rating_test.items():
    if j not in movie2userrating_test:
        movie2userrating[j]=[[i],[r]]
    else:
        movie2userrating[j][0].append(i)
        movie2userrating[j][1].append(r)
for j,(users,r) in movie2userrating_test.items():
    movie2userrating[j][1]=np.array(r)
print("conversion done")

K=10
W=np.random.randn(N,K)
b= np.zeros(N)
U=np.random.randn(M,K)
c = np.zeros(M)
mu = np.mean(list(usermovie2rating.values()))

def get_loss(m2u):
#    d:movie_id->(user_ids,ratings)
     N=0
     sse=0
     for j,(user_ids,r) in m2u.items():
         p = W[user_ids].dot(U[j])+b[user_ids]+c[j]+mu
         delta = r-p
         sse += delta.dot(delta)
         N+=len(r)
     return sse/N

epochs = 25
reg = 0.1
train_losses = []
test_losses = []

for epoch in tqdm(range(epochs)):
    print("epoch:",epoch)
    epoch_start=datetime.now()
    
    #update W and b
    t0 = datetime.now()
    for i in tqdm(range(N)):
        m_ids, r = user2movierating[i]
        matrix = U[m_ids].T.dot(U[m_ids]) + np.eye(K) * reg
        vector = (r - b[i] - c[m_ids] - mu).dot(U[m_ids])
        bi = (r - U[m_ids].dot(W[i]) - c[m_ids] - mu).sum()
        
        #set updates
        W[i] = np.linalg.solve(matrix, vector)
        b[i]=bi/(len(user2movie[i])+reg)
    print("Updated W and b:",datetime.now()-t0) 
    
    t0 = datetime.now()    
    for j in range(M):
        try:
            u_ids,r = user2movierating[j]
            matrix = W[u_ids].T.dot(W[u_ids])+np.eye(K)*reg
            vector = (r-b[u_ids]-c[j]-mu).dot(W[u_ids])
            cj = (r-W[u_ids].dot(U[j])-b[u_ids]-mu).sum()
            
            U[j] = np.linalg.solve(matrix,vector)
            c[j] = cj/(len(movie2user[j])+reg)
            
        except KeyError:
            pass
    print("Updated U and c:",datetime.now()-t0)
    print("Epoch duration :",datetime.now()-epoch_start)
    
    #store train loss
    t0=datetime.now()
    #store train loss
    train_losses.append(get_loss(movie2userrating))
    #store test loss
    test_losses.append(get_loss(movie2userrating_test))
    
    print("Cost:", datetime.now()-t0)
    print("train_loss:", train_losses[-1])
    print("test_loss:",test_losses[-1])
    
print("train losses:",train_losses)
print("test losses:",test_losses)

    
plt.plot("train_losses",label="train loss")
plt.plot("test losses", label="test loss")
plt.legend()
plt.show            
        
    
            
                
    