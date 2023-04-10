#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import datetime
from cv import *
from utils import plot_cv_indices, IC, mse_corr_loss
from mlp import create_ae_mlp
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import os, gc
import keras.backend as K

path='../data/'#please change to your directory

data=pd.read_csv(path+"panel_zscore.csv")

data.date=pd.to_datetime(data.date)
data.symbol=data.symbol.astype('str').apply(lambda x:x.zfill(6))
data.describe()


data = data.drop('x_90', axis=1)

train_data = data[(data['date'] >= '2018-04-01') & (data['date'] <= '2021-01-01')]
eval_data = data[data['date'] >= '2021-01-01']
print(train_data.isnull().sum())
def fillna_group(group):
    return group.fillna(method='ffill', axis=0).fillna(method='bfill', axis=0)

train_data = train_data.groupby('symbol').apply(fillna_group)
train_data = train_data.dropna()

X = train_data.drop(['y', 'date', 'symbol'], axis=1).values
y = train_data['y'].values

eval_data = data[data['date'] >= '2021-01-01']
print(eval_data.isnull().sum())

eval_data = eval_data.groupby('symbol').apply(fillna_group)
eval_data = eval_data.dropna()

eval_dates = eval_data.date
eval_X = eval_data.drop(['y', 'date', 'symbol'], axis=1).values
eval_y = eval_data['y'].values

print(train_data.isnull().sum().sum())


os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

params = {'in_dim': X.shape[1], 
          'out_dim': 1, 
          'hidden_units': [96, 96, 896, 448, 448, 256], 
          'dropout_rates': [0.03527936123679956, 0.038424974585075086, 0.42409238408801436, 0.10431484318345882, 0.49230389137187497, 0.32024444956111164, 0.2716856145683449, 0.4379233941604448], 
          'lr':1e-3, 
         }


res_fold = '../results/mlp/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(res_fold):
    os.makedirs(res_fold)
log_fold = '../logs/mlp/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if not os.path.exists(log_fold):
    os.makedirs(log_fold)
    
batch_size = 4096
prediction_length = 10
pred_results = pd.DataFrame(columns=['date', 'symbol', 'y', 'pred_y'])
train_X = train_data.drop(['y', 'symbol'], axis=1)
train_Y = train_data[['date', 'y']]
train_x = train_X.drop(['date'], axis=1).values
train_y = train_Y.drop(['date'], axis=1).values

eval_dates = np.sort(np.unique(eval_dates)) # sort dates 
train_dates = np.sort(np.unique(train_data['date'].values))

# initialize the prediction X
pred_dates = eval_dates[:prediction_length]
pred_data = eval_data[eval_data['date'].isin(pred_dates)]
pred_X = pred_data.drop(['y', 'symbol'], axis=1)
pred_Y = pred_data[['date','y']]

for k in range(len(eval_dates)//prediction_length):
    # add the next predicted time interval data into and delete the oldest data 
    # use the pre_data as validation for avoiding overfitting
    if k > 0:
        del_train_dates = train_dates[k * prediction_length : (k+1) * prediction_length]
        # train_X = train_X[~train_X['date'].isin(del_train_dates)]
        # train_Y = train_Y[~train_Y['date'].isin(del_train_dates)]
        
        train_X = pd.concat((train_X, pred_X))
        train_Y = pd.concat((train_Y, true_Y))
        
        train_x = train_X.drop(['date'], axis=1).values
        train_y = train_Y.drop(['date'], axis=1).values
        
    pred_dates = eval_dates[k * prediction_length : (k+1) * prediction_length]
    pred_data = eval_data[eval_data['date'].isin(pred_dates)]
    pred_X = pred_data.drop(['y', 'symbol'], axis=1)
    true_Y = pred_data[['date', 'y']]
    pred_x = pred_X.drop(['date'], axis=1).values
    true_y = true_Y.drop(['date'], axis=1).values
    
    ckp_path = res_fold + f'/Model_{k}.hdf5'
    model = create_ae_mlp(**params)
    ckp = keras.callbacks.ModelCheckpoint(ckp_path, monitor = 'val_pred_loss', verbose = 0, 
                          save_best_only = True, save_weights_only = True, mode = 'min')
    es = keras.callbacks.EarlyStopping(monitor = 'val_pred_loss', min_delta = 1e-4, patience = 10, mode = 'min', 
                       baseline = None, restore_best_weights = True, verbose = 0)
    tb = keras.callbacks.TensorBoard(log_dir=log_fold+f'/Model_{k}',
                             histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch',
                             embeddings_freq=0, embeddings_metadata=None) 
    history = model.fit(train_x, [train_x,train_y], validation_data = (pred_x, [pred_x, true_y]),
                        epochs = 100, batch_size = batch_size, callbacks = [ckp, es, tb], verbose = 0)
    hist = pd.DataFrame(history.history)
    score = hist['val_pred_IC'].max()
    print(f'val_correlation:\t', score)
    print(f'====== Prediction {k} =======')

    _, pred_y = model.predict(pred_x, batch_size)
    prediction = pd.concat((pred_data[['date', 'symbol', 'y']].reset_index(), pd.DataFrame(pred_y, columns=['pred_y'])), axis=1)
    pred_results = pd.concat((pred_results, prediction), axis=0).reset_index(drop=True)
    print(f"Correlation {IC(tf.convert_to_tensor(true_y, tf.float32), pred_y).numpy()}")

# save the prediction results
pred_results.to_csv(res_fold + '/mlp_pred_results.csv')

