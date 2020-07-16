# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 15:44:24 2020

@author: rishi
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle 

import keras
from keras.models import Model
from keras.layers import Input,Embedding,Dense,Concatenate,Flatten,Add,Dot
from keras.regularizers import l2
from keras.optimizers import Adam,SGD
from keras.layers import BatchNormalization,Activation,Dropout
from keras.callbacks import EarlyStopping,ReduceLROnPlateau


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

#############MAIN BRANCH###############
u = Input(shape=(1,))
m = Input(shape=(1,))

u_embedding = Embedding(N,K)(u) #(N,1,K) the 1 is the sequence length note that N is the batch_size and not the number of users
m_embedding = Embedding(M,K)(m)  #(N,1,K) the 1 is the sequence length
u_bias = Embedding(N,1)(u) #(N,1,1) the 1 is the sequence length
m_bias = Embedding(M,1)(m) #(N,1,1)

x= Dot(axes=2)([u_embedding,m_embedding]) #(N,1,1)
x = Add()([x,u_bias,m_bias])
x= Flatten()(x)#(N,1)


##########SIDE BRANCH###################
u_embedding = Flatten()(u_embedding)    #(N,K)
m_embedding = Flatten()(m_embedding)    #(N,K)
y = Concatenate()([u_embedding,m_embedding])
y=Dense(400)(y)
y= Activation("elu")(y)
y= Dropout(0.5)(y)
y=Dense(1)(y)


###MERGE
x = Add()([x,y])

model = Model(inputs=[u,m],outputs=x)
model.compile(loss="mse",
              optimizer=SGD(lr=0.08,momentum=0.9),
              metrics=["mse"])
r = model.fit(
  x=[df_train.userId.values, df_train.movie_idx.values],
  y=df_train.rating.values - mu,
  epochs=epochs,
  batch_size=128,
  validation_data=(
    [df_test.userId.values, df_test.movie_idx.values],
    df_test.rating.values - mu
  )
)


# plot losses
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.legend()
plt.show()

# plot mse
plt.plot(r.history['mean_squared_error'], label="train mse")
plt.plot(r.history['val_mean_squared_error'], label="test mse")
plt.legend()
plt.show()



