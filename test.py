import time
import argparse
import random

import numpy as np
import tensorflow as tf

from model import *
import dataset
import mapmaker
import gamesystem

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

NOMAL = 0
ME = 2
WALL = 1
GOAL = 10

def makeMulti(X_map, S1, S2, goal, agentNum):
    '''在空地上生成除了自己和终点以外的Agent'''
    agentList = []
    S_map = X_map
    mask = [[0 for i in range(IMSIZE)]for j in range(IMSIZE)]
    mask[goal[0]][goal[1]] = GOAL
    mask[S2][S1] = GOAL #虽然是自己但是用GOAL代替···
    mask = tf.constant(mask)
    S_map = tf.where(mask == GOAL,GOAL,S_map)
    new_Multi = tf.where(X_map == NOMAL) # 获取可以生成Agent的位置
    agentNumMax = new_Multi.get_shape().as_list()[0]
    if(agentNumMax<=agentNum-1):
        print('ERROR:empty space is not enough!')
    else:
        agentIndex = random.sample(range(agentNumMax),agentNum-1)
        agentList.append([S2,S1])
        for i in range(agentNum-1):
            agentList.append(new_Multi[agentIndex[i]])
        agentList = np.array(agentList)
        return agentList
    
    
def eval(my_model, dataset):
    num_batches = dataset.num_examples//BATCH_SIZE
    total_examples = num_batches*BATCH_SIZE
    #total_examples = TESTROUND*BATCH_SIZE

    total_err = 0.0
    total_loss = 0.0
    
    X_mapList = []
    V_mapList = []
    goalList = []
    game = gamesystem.gamesystem(IMSIZE,BATCH_SIZE)
    for batch in range(TESTROUND):
        #X_batch, S1_batch, S2_batch, y_batch = dataset.rollNEWMULTI(width=8,height=8,wallnum=10,goalnum=1,mapnum=BATCH_SIZE)
        X_batch, S1_batch, S2_batch, y_batch = dataset.next_batch(BATCH_SIZE)
        
        y_batch = tf.constant(y_batch)
        y_batch = tf.dtypes.cast(y_batch, dtype=tf.int64)
        
        logits, prob_actions,value = my_model.call(
                X_batch, S1_batch, S2_batch, VInum=VINUM)
        for i in range(BATCH_SIZE):
            now_batch = tf.constant(X_batch[i])
            # 障碍物地图
            X_map = tf.slice(now_batch,[0,0,0],[IMSIZE,IMSIZE,1])
            X_map = tf.reshape(X_map,[-1,IMSIZE])
            # 原始价值地图
            origin_V_map = tf.slice(now_batch,[0,0,1],[IMSIZE,IMSIZE,1])
            origin_V_map = tf.reshape(origin_V_map,[-1,IMSIZE])
            goal = tf.where(origin_V_map == GOAL)[0]
            #处理后价值地图
            V_map = tf.constant(value[i])
            V_map = tf.slice(V_map,[0,0,1],[IMSIZE,IMSIZE,1])
            V_map = tf.reshape(V_map,[-1,IMSIZE])
            print(i)
            print(X_map)
            print(V_map)
            X_mapList.append(X_map)
            V_mapList.append(V_map)
            goalList.append(goal)
        roadList, stepList = game.rungame(X_mapList,V_mapList,S1_batch,S2_batch,goalList)
    return roadList,stepList


#trainset = mapmaker.map(width=IMSIZE,height=IMSIZE,wallnum=WALLNUMBER,goalnum=GOALNUMBER)
#testset = mapmaker.map(width=IMSIZE,height=IMSIZE,wallnum=WALLNUMBER,goalnum=GOALNUMBER)
#trainset = dataset.Dataset(filepath=DATAFILE, mode='train', imsize=IMSIZE)
testset = dataset.Dataset(filepath=DATAFILE, mode='train', imsize=IMSIZE)

RunVIN = tf.saved_model.load('./SavedModel/')
mean_err = 0.0
mean_loss = 0.0

roadList, stepList = eval(RunVIN, testset)
roadList = np.array(roadList)
stepList = np.array(stepList)
print(roadList)
print(stepList)
