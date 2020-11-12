import time
import argparse

import numpy as np
import tensorflow as tf

from model import *
import dataset

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DATAFILE = 'G:/OneDrive/TEU/Tensor/VIN_TensorFlow-master/data/gridworld_8x8.npz'
IMSIZE = 8
LEARNING_RATE = 0.002
EPOCHS = 30
VINUM = 10
CH_I = 2
CH_H = 150
CH_Q = 10
BATCH_SIZE = 128
USE_LOG = False
LOG_DIR = '.log/'


def train(my_model, dataset):
    num_batches = dataset.num_examples//BATCH_SIZE
    total_examples = num_batches*BATCH_SIZE

    total_err = 0.0
    total_loss = 0.0

    for batch in range(num_batches):
        X_batch, S1_batch, S2_batch, y_batch = dataset.next_batch(BATCH_SIZE)
        
        y_batch = tf.constant(y_batch)
        y_batch = tf.dtypes.cast(y_batch, dtype=tf.int64)

        with tf.GradientTape() as tape:
            logits, prob_actions,q,q_out = my_model.call(
                X_batch, S1_batch, S2_batch, VInum=VINUM)
            print(tf.shape(q))
            #print(tf.shape(q_out))
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

    total_err = 0.0
    total_loss = 0.0
    for batch in range(num_batches):
        X_batch, S1_batch, S2_batch, y_batch = dataset.next_batch(BATCH_SIZE)
        
        y_batch = tf.constant(y_batch)
        y_batch = tf.dtypes.cast(y_batch, dtype=tf.int64)
        
        logits, prob_actions = my_model.call(
            X_batch, S1_batch, S2_batch, VInum=VINUM)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_batch, logits=logits, name='cross_entropy')
        loss = tf.reduce_sum(cross_entropy, name='cross_entropy_mean')
        actions = tf.argmax(prob_actions, 1)
        num_err = tf.reduce_sum(tf.dtypes.cast(tf.not_equal(actions, y_batch),dtype=tf.float64))

        total_err += num_err
        total_loss += loss

    return total_err/total_examples, total_loss/total_examples


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
