#!/usr/bin/env pyhton
# coding: utf-8

from keras.layers import LSTM,Masking,Dropout,Concatenate,Permute,Dot,Input,Multiply,TimeDistributed
from keras.layers import RepeatVector,Activation,Lambda,Dense,Reshape
from keras.optimizers import *
from keras.models import load_model,Model,Sequential
from keras.engine.topology import Layer,InputSpec
from keras import regularizers,constraints,initializers,activations
from Attention import Encoder

import keras.backend as K
import numpy as np 
import pandas as pd 
import time 
import random
import copy
import seaborn as sb 
import matplotlib.pyplot as plt

"""
In this section,we create models as list:
basic model | Vanilla LSTM
stack model | stacked LSTM
multi model | Encoder-Decoder LSTM
atten model | Attention ED LSTM
mulat model | MultiAttention ED LSTM
extra model | above model with extra features(similarity,截止目前的正确率...此处最好加入全局变量)
"""

def Vanilla_lstm(n_units,n_step,n_feature,n_out,activate='relu',opt='adam',loss='mse'):
    model = Sequential()
    model.add(LSTM(n_units,activation=activate,input_shape=(n_step,n_feature)))
    model.add(Dense(n_out))
    model.compile(optimizer=opt,loss=loss)
    return model

def stacked_lstm(n_units,n_step,n_feature,n_out,activate='relu',opt='adam',loss='mse'):
    if len(n_units) == 1:
        return Vanilla_lstm(n_units[0],n_step,n_feature,n_out,activate,opt,loss)
    model = Sequential()
    model.add(LSTM(n_units[0],activation=activate,return_sequences=True,input_shape=(n_step,n_feature)))
    for i in range(1,len(n_units)-1):
        mode.add(LSTM(n_units[i],activation=activate,return_sequences=True))
    model.add(LSTM(n_units[-1],activation=activate))
    model.add(Dense(n_out))
    model.compile(optimizer=opt,loss=loss)
    return model

def en_de_lstm(n_input_units,n_input_step,n_feature,n_output_units,n_output_step,opt='adam',loss='mse'):
    model = Sequential()
    model.add(LSTM(n_input_units,activation='relu',input_shape=(n_input_step,n_feature)))
    model.add(RepeatVector(n_output_step))
    model.add(LSTM(n_output_units,activation='relu',return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer=opt,loss=loss)
    return model

def attention_lstm(units,steps,input_dims):
    #Functional
    #define layer
    inputs = Input(shape=(steps,input_dims),name='inputs')
    h = Input(shape=(units,),name='h0')
    c = Input(shape=(units,),name='c0')

    mask = Masking(mask_value=-1.)
    lstm = LSTM(units,activation='relu',return_sequences=True,return_state=True)

    #build model
    hidden = []
    x = mask(inputs)
    for index in range(steps):
        attention,x1 = Encoder(units,index)(x,states=[h,c])
        _,h1,c1 = lstm(x1,initial_state=[h,c])
        hidden.append(h1)

    hidden = Concatenate(axis=-1)(hidden)    
    model = Model(inputs=[inputs,h,c],outputs=hidden)
    #model = Model(inputs=[inputs,h,c],outputs=[attention,x1])
    return model

if __name__ == "__main__":
    model = attention_lstm(5,4,3)
    inputs = [
        [[1,2,3],[3,4,5],[6,7,8],[-1,-1,-1]],
        [[3,5,7],[2,4,6],[6,8,0],[2,3,4]],
        [[1,1,1],[2,2,2],[-1,-1,-1],[-1,-1,-1]]
    ]

    inputs = np.array(inputs)
    print(inputs.shape)
    h = np.zeros(shape=(inputs.shape[0],5))
    c = np.zeros(shape=(inputs.shape[0],5))

    hidden = model.predict([inputs,h,c])
    #print(attention,attention.shape)
    #print(x1,x1.shape)
    print(hidden,hidden.shape)

    