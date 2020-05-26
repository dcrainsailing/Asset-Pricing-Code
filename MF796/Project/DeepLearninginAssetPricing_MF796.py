#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import time
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import concatenate
from keras.optimizers import SGD
from keras.layers.core import Lambda
import keras.backend as K
from keras.optimizers import Adam
from keras.constraints import MaxNorm
from keras.models import model_from_json


# # data preparation

# In[21]:


file = open('a4','rb')
dataset = pickle.load(file, encoding='utf-8')
file.close()

macro_data, micro_data, return_data = dataset
print(macro_data.shape, micro_data.shape, return_data.shape)
print(list(map(type, dataset)))


# In[22]:


# dataset split
train_micro = micro_data[:1800, :, :]
train_macro = macro_data[:1800, :, :]
train_return = return_data[1:1801, :]
train_micro = (train_micro - train_micro.mean(axis=1).reshape(-1, 1, train_micro.shape[2]))/train_micro.std(axis=1).reshape(-1, 1, train_micro.shape[2])


validation_micro = micro_data[1801:2400, :, :]
validation_macro = macro_data[1801:2400, :, :]
validation_return = return_data[1802:2401, :]
validation_micro = (validation_micro - validation_micro.mean(axis=1).reshape(-1, 1, validation_micro.shape[2]))/validation_micro.std(axis=1).reshape(-1, 1, validation_micro.shape[2])


test_micro = micro_data[2401:-1, :, :]
test_macro = macro_data[2401:-1, :, :]
test_return = return_data[2402:, :]
test_micro = (test_micro - test_micro.mean(axis=1).reshape(-1, 1, test_micro.shape[2]))/test_micro.std(axis=1).reshape(-1, 1, test_micro.shape[2])


# # model parameters

# In[23]:


LSTM_units, sdf_units, LSTM_units2, n_g =   [4, 32, 8, 8]


# In[24]:


LSTM_units = 4
sdf_units = 32

LSTM_units2 = 8
n_g = 8

dropout = 0.2

epoch = 400
batch_size = 200
n_iteration = 1


# In[25]:


# model parameter
ndays = train_micro.shape[0]
nstocks = train_micro.shape[1]
nmacro = train_macro.shape[2]
nmicro = train_micro.shape[2]
LSTM_delay = train_macro.shape[1]


# # SDF Network

# In[26]:



# SDF networks

# data loading
ma1 = Input(shape=(LSTM_delay, nmacro), name='macro')

mi1 = Input(shape=(nstocks,nmicro), name='micro')

ret1 = Input(shape=(nstocks,), name='return')

g_loaded = Input(shape=(nstocks, n_g), name='function_G')

# LSTM
lstm1_1 = LSTM(LSTM_units, name='lstm')(ma1) # (1000, 4)
lstm1 = Lambda(lambda x:K.repeat(x, nstocks), name='lstm_reshape')(lstm1_1)


# SDF weights DNN
w1 = concatenate([mi1, lstm1], name='micro_macro_combined') # (1000, 50, 104)  
w2 = Lambda(lambda x:K.reshape(x, shape=(-1,nstocks*(nmicro+LSTM_units),)), name='ffn_input_reshape')(w1)
w3 = Dense(SGD_units, activation='relu', name='ffn_layer1')(w2)
w4 = Dropout(dropout, name='ffn_dropout1')(w3)
w5 = Dense(SGD_units, activation='relu', name='ffn_layer2')(w4)
w6 = Dropout(dropout, name='ffn_dropout2')(w5)
w = Dense(nstocks, name='ffn_output_weights')(w6)


# SDF construction
def construction(x):
    tmp = 1 - x[0] * x[1]
    tmp = K.sum(tmp, axis=1)
    tmp = K.reshape(tmp, shape=(-1,1)) # (1000, 1)
    tmp = K.repeat(tmp, nstocks) # (1000, 50, 1)
    tmp = K.reshape(tmp, shape=(-1,nstocks)) # (1000, 50)
    tmp = tmp * x[1] # (1000, 50)
    tmp = K.reshape(tmp, shape=(-1, nstocks, 1))
    return tmp # the M_{t+1}R_{t+1}

sdf = Lambda(construction, name='SDF')([w,ret1])


# combine those two and calculate loss
loss_function_w1 = Lambda(lambda x:x[0]*x[1], name='loss')([sdf, g_loaded]) 
loss_function_w = Lambda(lambda x:K.reshape(x, shape=(-1, nstocks*n_g)), name='loss_reshape')(loss_function_w1)


def mean_squared_error1(y_true, y_pred):
    return K.mean(K.square(K.mean(y_pred-y_true,axis=0)))

# with weights output for validation and sdf output for condition network training
model_output_w = Model(inputs=[ma1, mi1], outputs=w) # acquires weights given info
model_output_sdf = Model(inputs=[ma1, mi1, ret1], outputs=sdf) # acquires MR for condition networks

model_output_sdf.compile(optimizer='adam', loss=mean_squared_error1)

# sdf model compile
model_w = Model(inputs=[ma1, mi1, ret1, g_loaded], outputs=loss_function_w)
model_w.compile(optimizer='adam', loss=mean_squared_error1)


# # Conditional Networks

# In[27]:


# conditional networks

ma2 = Input(shape=(LSTM_delay, nmacro), name='macro')

mi2 = Input(shape=(nstocks,nmicro), name='micro')

sdf_loaded = Input(shape=(nstocks,1), name='sdf')

lstm2_1 = LSTM(LSTM_units2, name='lstm')(ma2) 
lstm2 = Lambda(lambda x:K.repeat(x, nstocks), name='lstm_reshape')(lstm2_1)

g0 = concatenate([mi2, lstm2], name='g0') 
g1 = Lambda(lambda x:K.reshape(x, shape=(-1,nstocks*(nmicro+LSTM_units2),)), name='g1')(g0)
g3 = Dropout(dropout, name='g3')(g1)
g4 = Dense(nstocks*n_g, name='g4')(g3) 
g5 = Lambda(lambda x:K.reshape(x, shape=(-1,nstocks,n_g)), name='g5')(g4)
g = Lambda(lambda x:(x-K.reshape(K.mean(x, axis=-1), (-1,nstocks,1)))/K.reshape(K.std(x, axis=-1), (-1,nstocks,1)), name='g')(g5)

loss_function_g1 = Lambda(lambda x:x[0]*x[1], name='loss')([sdf_loaded, g]) 
loss_function_g = Lambda(lambda x:K.reshape(x, shape=(-1, nstocks*n_g)), name='loss_reshape')(loss_function_g1)

model_output_g = Model(inputs=[ma2, mi2], outputs=g) # acquires MR for condition networks

def mean_squared_error2(y_true, y_pred):
    return -K.mean(K.square(K.mean(y_pred-y_true,axis=0)))

model_g = Model(inputs=[ma2, mi2,sdf_loaded], outputs=loss_function_g)
model_g.compile(optimizer='adam', loss=mean_squared_error2)


# In[28]:



def moving_average(x, n):
    xx = np.cumsum(x)
    xxx = xx.copy()
    xxx[n:] = (xx[n:] - xx[:-n]) / n
    for i in range(n):
        xxx[i] = xxx[i] / (i+1)
    return xxx


# # traning process

# In[29]:


# data process
y_train = np.zeros((ndays, n_g*nstocks))
y_train_unconditional = np.zeros((ndays, nstocks, 1))

ma = train_macro
mi = train_micro
ret = train_return
sdf_loss = []



# first use unconditional methods to give a initial guess of sdf
t1 = time.time()
# SDF nets
history_w = model_output_sdf.fit([ma, mi, ret], y_train_unconditional, epochs=epoch, batch_size=batch_size, verbose=0)
function_sdf = model_output_sdf.predict([ma, mi, ret])
sdf_loss = sdf_loss + history_w.history['loss'].copy() # save loss
print('done with pre-training, the train loss is', round(sdf_loss[-1],6) ,', using', round(time.time()-t1, 2), 'seconds')
# plt.plot(moving_average(history_w.history['loss'], 20))
# plt.show()
print('============================')

for i in range(n_iteration):
    t1 = time.time()
    # conditional nets
    history_w = model_g.fit([ma, mi, function_sdf], y_train, epochs=epoch,  batch_size=batch_size, verbose=0)
    function_g = model_output_g.predict([ma, mi])
    print('done with conditional', i, 'loss is',round(history_w.history['loss'][-1],6),'using', round(time.time()-t1, 2), 'seconds')
#     plt.plot(moving_average(history_w.history['loss'], 20))
#     plt.show()

    t1 = time.time()
    # SDF nets
    history_w = model_w.fit([ma, mi, ret, function_g], y_train, epochs=epoch, batch_size=batch_size,  verbose=0)
    function_sdf = model_output_sdf.predict([ma, mi, ret])
    sdf_loss = sdf_loss + history_w.history['loss'].copy() # save loss
    print('done with sdf network', i,',train loss is', round(sdf_loss[-1],6),  ', using', round(time.time()-t1, 2), 'seconds')
#     plt.plot(moving_average(history_w.history['loss'], 20))
#     plt.show()
    
    print('============================')

plt.figure(figsize=(20,6))
plt.plot(moving_average(sdf_loss[:], 20))
plt.title('loss function of sdf nets')
plt.grid()
plt.show()


# In[30]:



result = pd.DataFrame([[0,0,0,0, 0, 0, 0] for i in range(3)], columns = ['mean','std','shape','SR','bench_mean','bench_std','bench_SR'], index=['train','validation','test'])

# training set condition
train_weights = model_output_w.predict([train_macro[:,:], train_micro[:, :, :]])
train_weights = 1/(1+np.exp(-train_weights)) 
train_daily_return = (train_weights/train_weights.sum(axis=1).reshape(-1,1) * train_return).sum(axis=1)
result.iloc[0, 0] = train_daily_return.mean()*252
result.iloc[0, 1] = train_daily_return.std() * np.sqrt(252)
result.iloc[0, 2] = train_daily_return.shape

fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(311)
benchmark = train_return.mean(axis=1)
ax1.plot(np.cumprod(1+train_daily_return), color='red',label='sdf')
ax1.plot(np.cumprod(1+benchmark), color='blue',label='benchmark')
plt.legend()
plt.title('training set backtest')
plt.grid()

result.iloc[0, 3] = train_daily_return.mean()/train_daily_return.std()*np.sqrt(252)
result.iloc[0, 4] = benchmark.mean()*252
result.iloc[0, 5] = benchmark.std()*np.sqrt(252)
result.iloc[0, 6] = benchmark.mean()/benchmark.std()*np.sqrt(252)


# training set condition
validation_weights = model_output_w.predict([validation_macro[:,:], validation_micro[:, :, :]])
validation_weights = 1/(1+np.exp(-validation_weights))
validation_daily_return = (validation_weights/validation_weights.sum(axis=1).reshape(-1,1) * validation_return).sum(axis=1)
result.iloc[1, 0] = validation_daily_return.mean()*252
result.iloc[1, 1] = validation_daily_return.std() * np.sqrt(252)
result.iloc[1, 2] = validation_daily_return.shape

benchmark = validation_return.mean(axis=1)
ax2 = fig.add_subplot(312)
ax2.plot(np.cumprod(1+validation_daily_return), color='red',label='sdf')
ax2.plot(np.cumprod(1+benchmark), color='blue',label='benchmark')
plt.legend()
plt.grid()
plt.title('validation set backtest')

result.iloc[1, 3] = validation_daily_return.mean()/validation_daily_return.std()*np.sqrt(252)
result.iloc[1, 4] = benchmark.mean()*252
result.iloc[1, 5] = benchmark.std()*np.sqrt(252)
result.iloc[1, 6] = benchmark.mean()/benchmark.std()*np.sqrt(252)


# # training set condition
test_weights = model_output_w.predict([test_macro[:,:], test_micro[:, :, :]])
test_weights = 1/(1+np.exp(-test_weights))
test_daily_return = (test_weights/test_weights.sum(axis=1).reshape(-1,1) * test_return).sum(axis=1)
result.iloc[2, 0] = test_daily_return.mean()*252
result.iloc[2, 1] = test_daily_return.std() * np.sqrt(252)
result.iloc[2, 2] = test_daily_return.shape

benchmark = test_return.mean(axis=1)
ax2 = fig.add_subplot(313)
plt.plot(np.cumprod(1+test_daily_return), color='red',label='sdf')
plt.plot(np.cumprod(1+benchmark), color='blue',label='benchmark')
plt.legend()
plt.grid()
plt.title('test set backtest')

result.iloc[2, 3] = test_daily_return.mean()/test_daily_return.std()*np.sqrt(252)
result.iloc[2, 4] = benchmark.mean()*252
result.iloc[2, 5] = benchmark.std()*np.sqrt(252)
result.iloc[2, 6] = benchmark.mean()/benchmark.std()*np.sqrt(252)

plt.show()
print(LSTM_units, sdf_units, LSTM_units2, n_g)
print(result)
print('====================================')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




