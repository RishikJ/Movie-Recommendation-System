# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:14:20 2020

@author: rishi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from scipy.sparse import lil_matrix,csr_matrix,save_npz,load_npz

import keras.backend as K
from keras.layers import Input,Dropout,Dense
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD,Adam
from keras.callbacks import ReduceLROnPlateau

batch_size=128
epochs=20
reg=0.0001

A= load_npz("Atrain.npz")
A_test =load_npz("Atest.npz")

mask = (A>0)*1.0
mask_test = (A_test>0)*1.0

A_copy = A.copy()
mask_copy = mask.copy()
A_test_copy = A_test.copy()
mask_test_copy = mask_test.copy()

N,M = A.shape
print("N:",N,"M:",M)

print("N//batch_size:",N//batch_size)

mu=A.sum()/mask.sum()

print("mu:",mu)

i= Input(shape=(M,))
x= Dropout(0.7)(i)
x= Dense(700,activation="relu",kernel_regularizer=l2(reg))(x)
x = Dense(M,kernel_regularizer=l2(reg))(x)


def custom_loss(y_true,y_pred):
    mask = K.cast(K.not_equal(y_true,0),dtype="float32")
    diff = y_true-y_pred
    sqdiff = diff*diff*mask
    sse = K.sum(K.sum(sqdiff))
    n= K.sum(K.sum(mask))
    return sse/n

def generator(A,Mask):
    while True:
        A,Mask = shuffle(A,Mask)
        for i in range(A.shape[0]//batch_size+1):
            upper = min((i+1)*batch_size,A.shape[0])
            a = A[i*batch_size:upper].toarray()
            m = Mask[i*batch_size:upper].toarray()
            a= a-mu*m
#            m2 = (np.random.random(a.shape)>0.5)
#            noisy = a*m2
            noise = a#no noise
            yield noise,a   #input,target

def test_generator(A,Mask,A_test,Mask_test):
    while True:
        for i in range(A.shape[0]//batch_size+1):
            upper = min((i+1)*batch_size,A.shape[0])
            a = A[i*batch_size:upper].toarray()
            m = Mask[i*batch_size:upper].toarray()
            at = A_test[i*batch_size:upper].toarray()
            mt = Mask_test[i*batch_size:upper].toarray()
            a= a-mu*m
            at-mu*mt
            yield a,at
            
model =Model(i,x)
model.compile(
        loss = custom_loss,
        optimizer = "adam",
        metrics = [custom_loss])

reduce_lr=ReduceLROnPlateau(monitor="val_custom_loss",patience=2,min_lr=0.0001,mode="auto")

r=model.fit_generator(
        generator(A,mask),
        validation_data = test_generator(A_copy,mask_copy,A_test_copy,mask_test_copy),
        epochs = epochs,
        steps_per_epoch = A.shape[0]//batch_size+1,
        validation_steps = A_test.shape[0]//batch_size+1,
        callbacks = [reduce_lr])
print(r.history.keys())

# plot losses
plt.plot(r.history['loss'], label="train loss")
plt.plot(r.history['val_loss'], label="test loss")
plt.legend()
plt.show()

# plot mse
plt.plot(r.history['custom_loss'], label="train mse")
plt.plot(r.history['val_custom_loss'], label="test mse")
plt.legend()
plt.show()


            
    
    
    
