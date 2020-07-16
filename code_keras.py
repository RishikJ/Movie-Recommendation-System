# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 01:30:05 2020

@author: rishi
"""
import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


import keras
from keras.models import Model
from keras.layers import Input,Embedding,Dot,Add,Flatten
from keras.regularizers import l2
from keras.optimizers import Adam,SGD

df = pd.read_csv("edited_rating.csv")
N = df.userId.max()+1
M = df.movie_idx.max()+1

df = shuffle(df)
cutoff = int(0.8*len(df))
df_train = df.iloc[:cutoff]
df_test = df.iloc[cutoff:]

K=10
mu = df_train.rating.mean()
epochs = 15
reg=0. #regularization penalty

u = Input(shape=(1,))
m = Input(shape=(1,))

u_embedding = Embedding(N,K, embeddings_regularizer=l2(reg))(u) #(N,1,K) the 1 is the sequence length
m_embedding = Embedding(M,K, embeddings_regularizer = l2(reg))(m)  #(N,1,K) the 1 is the sequence length

#subsubmodel = Model([u, m], [u_embedding, m_embedding])
#user_ids = df_train.userId.values[0:5]
#movie_ids = df_train.movie_idx.values[0:5]
#print("user_ids.shape", user_ids.shape)
#p = subsubmodel.predict([user_ids, movie_ids])
#print("p[0].shape:", p[0].shape)
#print("p[1].shape:", p[1].shape)
#exit()


u_bias = Embedding(N,1, embeddings_regularizer=l2(reg))(u) #(N,1,1) the 1 is the sequence length
m_bias = Embedding(M,1, embeddings_regularizer = l2(reg))(m) #(N,1,1)

#submodel = Model([u,m],[u_bias,m_bias])
#user_ids = df_train.userId.values[0:5]
#movie_ids = df_train.movie_idx.values[0:5]
#p = submodel.predict([user_ids,movie_ids])
#print("p[0].shape:",p[0].shape)
#print("p[1].shape:",p[1].shape)
#exit()


x = Dot(axes=2)([u_embedding,m_embedding]) #(N,1,1)

x = Add()([x, u_bias, m_bias])
x = Flatten()(x) #(N,1)

model = Model(inputs=[u,m],outputs=x)
model.compile(loss="mse",
              optimizer=SGD(lr=0.08, momentum=0.9),
              metrics=["mse"])

r = model.fit(
        x=[df_train.userId.values,df_train.movie_idx.values],
        y=df_train.rating.values-mu,
        epochs=epochs,
        batch_size=128,
        validation_data=(
                [df_test.userId.values,df_test.movie_idx.values],
                df_test.rating.values-mu),
        verbose=1) 
        
 #plot losses       
plt.plot(r.history["loss"],label = "train loss")
plt.plot(r.history["val_loss"], label="test loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()


#plot mse
plt.plot(r.history["mean_squared_error"],label ="train_mse")
plt.plot(r.history["val_mean_squared_error"],label="test_mse")
plt.xlabel("epoch")
plt.ylabel("error")
plt.legend()
plt.show()             
                

plt.plot(r.history["loss"],label = "train loss")
plt.plot(r.history["val_loss"], label="test loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()


#plot mse
plt.plot(r.history["mean_squared_error"],label ="train_mse")
plt.plot(r.history["val_mean_squared_error"],label="test_mse")
plt.xlabel("epoch")
plt.ylabel("error")
plt.legend()
plt.show()  

 