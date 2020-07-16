# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:00:48 2020

@author: rishi
"""

import pickle
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


import keras
from keras.models import Model
from keras.layers import Input,Embedding,Dense,Concatenate,Flatten
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
mu=df_train.rating.mean()
epochs=15
reg=0.0001


u = Input(shape=(1,))
m = Input(shape=(1,))

u_embedding = Embedding(N,K, embeddings_regularizer=l2(reg))(u) #(N,1,K) the 1 is the sequence length
m_embedding = Embedding(M,K, embeddings_regularizer = l2(reg))(m)  #(N,1,K) the 1 is the sequence length
u_embedding = Flatten()(u_embedding)# (N,K)
m_embedding =Flatten()(m_embedding)#  (M,K)
X = Concatenate()([u_embedding,m_embedding])

X = Dense(512)(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = Dropout(0.2)(X)
X = Dense(256)(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = Dropout(0.3)(X)
X = Dense(128)(X)
X = BatchNormalization()(X)
X = Activation("relu")(X)
X = Dropout(0.4)(X)
X= Dense(1)(X)

model = Model(inputs=[u,m],outputs=X)

model.compile(loss="mse",
              optimizer="adam",
              metrics=["mse"])
reduce_lr=ReduceLROnPlateau(monitor="val_loss",patience=2,min_lr=0.0001,mode="auto")
early = EarlyStopping(patience=3)

r = model.fit(
        x=[df_train.userId.values,df_train.movie_idx.values],
        y=df_train.rating.values-mu,
        epochs=epochs,
        batch_size=128,
        validation_data=(
                [df_test.userId.values,df_test.movie_idx.values],
                df_test.rating.values-mu),
        verbose=1,
        callbacks=[reduce_lr,early])