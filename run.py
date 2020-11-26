import time
import argparse
import random

import numpy as np
import tensorflow as tf

from model import *
import dataset
import mapmaker

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DATAFILE = 'G:/OneDrive/TEU/Tensor/VIN_TensorFlow-master/data/gridworld_8x8.npz'
IMSIZE = 8
WALLNUMBER = 30
GOALNUMBER = 1
LEARNING_RATE = 0.003
EPOCHS = 30
VINUM = 36
CH_I = 2
CH_H = 150
CH_Q = 10
BATCH_SIZE = 128
TRAINROUND=600
TESTROUND = 1
USE_LOG = False
LOG_DIR = '.log/'


def train(my_model, dataset):
    num_batches = dataset.num_examples//BATCH_SIZE
    total_examples = num_batches*BATCH_SIZE
    #total_examples = TRAINROUND*BATCH_SIZE

    total_err = 0.0
    total_loss = 0.0

    for batch in range(TRAINROUND):
        #X_batch, S1_batch, S2_batch, y_batch = dataset.rollNEWMULTI(width=IMSIZE,height=IMSIZE,wallnum=WALLNUMBER,goalnum=GOALNUMBER,mapnum=BATCH_SIZE)
        X_batch, S1_batch, S2_batch, y_batch = dataset.next_batch(BATCH_SIZE)
        
        y_batch = tf.constant(y_batch)
        y_batch = tf.dtypes.cast(y_batch, dtype=tf.int64)
        '''print('------------------')
        #X_map = tf.constant(X_batch[1])
        #X_map = tf.slice(X_map,[0,0,0],[IMSIZE,IMSIZE,1])
        #X_map = tf.reshape(X_map,[-1,IMSIZE])
        V_map = tf.constant(X_batch[1])
        V_map = tf.slice(V_map,[0,0,1],[IMSIZE,IMSIZE,1])
        V_map = tf.reshape(V_map,[-1,IMSIZE])
        #print('----------------------')
        #print(X_map)
        print(V_map)
        #print(X_batch)
        #print('------------------')
        print(S1_batch[1])
        #print('------------------')
        print(S2_batch[1])
        #print('------------------')
        print(y_batch[1])
        #print('------------------')'''
        with tf.GradientTape() as tape:
            logits, prob_actions,value = my_model.call(
                X_batch, S1_batch, S2_batch, VInum=VINUM)
            value = tf.slice(value,[1,0,0,1],[1,IMSIZE,IMSIZE,1])
            value = tf.reshape(value,[-1,IMSIZE])
            value = tf.dtypes.cast(value,tf.int32)
            
            '''print(value)
            print('----------------------')'''
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_batch, logits=logits, name='cross_entropy')
            loss = tf.reduce_sum(cross_entropy, name='cross_entropy_mean')
            actions = tf.argmax(prob_actions, 1)
            num_err = tf.reduce_sum(tf.dtypes.cast(tf.not_equal(actions, y_batch),dtype=tf.float64))
        grads = tape.gradient(loss,my_model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads,my_model.variables))
        total_err += num_err
        total_loss += loss
    
    tf.saved_model.save(my_model,'./SavedModel/')
    return total_err/total_examples, total_loss/total_examples

def eval(my_model, dataset):
    num_batches = dataset.num_examples//BATCH_SIZE
    total_examples = num_batches*BATCH_SIZE
    #total_examples = TESTROUND*BATCH_SIZE

    total_err = 0.0
    total_loss = 0.0
    for batch in range(TESTROUND):
        #X_batch, S1_batch, S2_batch, y_batch = dataset.rollNEWMULTI(width=8,height=8,wallnum=10,goalnum=1,mapnum=BATCH_SIZE)
        X_batch, S1_batch, S2_batch, y_batch = dataset.next_batch(BATCH_SIZE)
        X_batch = tf.dtypes.cast(X_batch,tf.float32)
        S1_batch = tf.dtypes.cast(S1_batch,tf.float32)
        S2_batch = tf.dtypes.cast(S2_batch,tf.float32)
        y_batch = tf.dtypes.cast(y_batch,tf.float32)
        
        y_batch = tf.constant(y_batch)
        y_batch = tf.dtypes.cast(y_batch, dtype=tf.int64)
        
        logits, prob_actions,value = my_model.call(
                X_batch, S1_batch, S2_batch, VInum=VINUM)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_batch, logits=logits, name='cross_entropy')
        loss = tf.reduce_sum(cross_entropy, name='cross_entropy_mean')
        actions = tf.argmax(prob_actions, 1)
        num_err = tf.reduce_sum(tf.dtypes.cast(tf.not_equal(actions, y_batch),dtype=tf.float64))

        total_err += num_err
        total_loss += loss

    return total_err/total_examples, total_loss/total_examples


#trainset = mapmaker.map(width=IMSIZE,height=IMSIZE,wallnum=WALLNUMBER,goalnum=GOALNUMBER)
#testset = mapmaker.map(width=IMSIZE,height=IMSIZE,wallnum=WALLNUMBER,goalnum=GOALNUMBER)
trainset = dataset.Dataset(filepath=DATAFILE, mode='train', imsize=IMSIZE)
testset = dataset.Dataset(filepath=DATAFILE, mode='test', imsize=IMSIZE)

RunVIN = VIN(chi=CH_I, chh=CH_H, chq=CH_Q)
mean_err = 0.0
mean_loss = 0.0

optimizer = tf.compat.v1.train.RMSPropOptimizer(
            LEARNING_RATE, epsilon=1e-6, centered=True)

for epoch in range(EPOCHS):

    start_time = time.time()

    mean_err, mean_loss = train(RunVIN, trainset)
    time_duration = time.time() - start_time
    out_str = 'Epoch: {:3d} ({:.1f} s): \n\t Train Loss: {:.5f} \t Train Err: {:.5f}'
    print(out_str.format(epoch, time_duration, mean_loss, mean_err))
print('\n Finished training...\n ')

mean_err, mean_loss = eval(RunVIN, testset)
print('Test Accuracy: {:.2f}%'.format(100*(1 - mean_err)))
