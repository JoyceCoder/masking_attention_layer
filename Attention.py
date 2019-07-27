import copy
import random
import time

import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import activations, constraints, initializers, regularizers
from keras.engine.topology import InputSpec, Layer
from keras.optimizers import *

class Encoder(Layer):
    """
    customize attention layer.
    build(input_shape) : define weights.This method must set `self.build = True` at the end,do as "super([Layer],self).bulid()"
    call(x) : layer`s logic lives.如果需要设置支持屏蔽，需要定义compute_mask_value函数，如果不需要，只关注参数x的逻辑
    compute_output_shape(input_shape) : if this layer modifies the shape of its input,specify here the shape transformation logic.This allow keras to do automatic shape inference.

    # Example

    ```python
    from keras import backend as K
    from keras.layers import Layer

    class MyLayer(Layer):
        def __init__(self,output_dim,**kwargs):
            self.output_dim = output_dim
            super(MyLayer,self).__init__(**kwargs)
        
        def build(self,input_shape):
            self.kernel = self.add_weight(name='kernel',shape=(input_shape[1],self.output_dim),initializer='uniform',trainable=True)
            super(MyLayer,self).bulid(input_shape)

        def call(self,x):
            return K.dot(x,self.kernel)

        def compute_output_shape(self,input_shape):
            return (input_shape[0],self.output_dim)
    ```
    # Function

    e = V* tanh(W*[X,state] + b)
    a = softmax(e)
    x = (a*x) [n,1] # typed feature

    # Parameter
    对于不同的层应该建立不同的attention层哦
    """

    def __init__(self,units,
                index,
                activation='tanh',
                attention_activation='softmax',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                **kwargs):
        super(Encoder,self).__init__(**kwargs)
        self.units = units
        self.index = index

        self.activation = activations.get(activation)
        self.attention_activation = activations.get(attention_activation)
        self.use_bias = use_bias
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.supports_masking = True
        self.state_size = (self.units,self.units)

    def build(self,input_shape):
        
        self.batch_size,self.steps,self.input_dim = input_shape

        self.kernel_W = self.add_weight(shape=(self.units * 2,self.steps),
                                        name='kernel',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.kernel_U = self.add_weight(shape=(self.steps,self.steps),
                                        name='kernel',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        self.kernel_V = self.add_weight(shape=(self.steps,),
                                        name='kernel',
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint)
        if self.use_bias:

            self.bias = self.add_weight(shape=(self.input_dim,self.steps),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.build = True

    def get_initial_state(self,inputs,mask=True):
        initial_state = K.zeros_like(inputs) #(sample,steps,input_dims)
        initial_state = K.sum(initial_state,axis=(1,2)) #(sample,)
        initial_state = K.expand_dims(initial_state) #(sample,1)
        if mask:
            return K.tile(initial_state,[1,inputs.shape[-1]]) #(sample,input_dims)
        else:
            return [K.tile(initial_state, [1, dim]) for dim in self.state_size]

    def call(self,inputs,states=None,mask=None,training=None):
        if states is None:
            state = self.get_initial_state(inputs,mask=False)

        if mask is not None:
            if isinstance(mask,list):
                mask = mask[0]
            if mask.dtype != tf.bool:
                mask = K.cast(mask,tf.bool)
            mask = K.permute_dimensions(mask,[1,0])

            def get_match_mask(mask,tensor):
                ndim = K.ndim(tensor)
                for _ in range(ndim-1):
                    mask = K.expand_dims(mask)
                add_shape = K.shape(tensor)[1:]
                multiple = K.concatenate([[1],add_shape],0)
                return K.tile(mask,multiple)

        h = states[0]
        c = states[1]

        state = K.concatenate([h,c]) #(samples,units*2)
        state = K.expand_dims(state,axis=1) #(samples,1,units*2)
        state = K.tile(state,[1,self.input_dim,1]) #(samples,input_dim,units*2)
        
        axes = [0,2,1]
        step_inputs = K.permute_dimensions(inputs,axes) #(samples,input_dim,steps)

        """
        K.transpose == tf.transpose(x)
        K.permuter_dimensions == tf.transpose(x,axes)
        """

        e = K.dot(step_inputs,self.kernel_U) #(samples,input_dim,steps)
        if self.use_bias:
            e = K.bias_add(e,self.bias)
        
        e = self.activation(e + K.dot(state,self.kernel_W)) #(samples,input_dim,steps)

        e = K.dot(e,K.expand_dims(self.kernel_V)) #(sample,input_dim,1)
        attention = K.squeeze(self.attention_activation(e),axis=-1) #(sample,input_dim)
        #使用tensorflow的slice函数，由于keras在切片操作上，会丢失一些信息，需要使用内置函数来操作。
        x = K.squeeze(K.slice(inputs,[0,self.index,0],[-1,1,-1]),axis=1)
        update_x = attention * x #(sample,input_dim)

        if mask is not None:
            initial_state = self.get_initial_state(inputs)
            mask_t = K.squeeze(K.slice(mask,[self.index,0],[1,-1]),axis=0)
            update_x = tf.where(get_match_mask(mask_t,update_x),
                                        update_x,
                                        initial_state)  
        update_x = K.expand_dims(update_x,axis=1)             
        return [attention,update_x]

    def compute_output_shape(self,input_shape):
        attention_shape = (input_shape[0],input_shape[2])
        output_shape = (input_shape[0],1,input_shape[2])
        return [attention_shape,output_shape]
        # output返回每个tensor的shape，以便让keras进行自行推算。

    def compute_mask(self,inputs,mask):
        if isinstance(mask,list):
            mask = mask[0]
        output_mask = K.slice(mask,[0,self.index],[-1,1])
        return output_mask

    def get_config(self):
        config = {
            'units': self.units,
            'index': self.index,
            'activation': activations.serialize(self.activation),
            'attention_activation': activations.serialize(self.attention_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Encoder,self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
