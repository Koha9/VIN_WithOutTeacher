import time
import argparse
import random

import numpy as np
import tensorflow as tf

from model import *
import dataset
import mapmaker
import gamesystem
import multiagent

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

DATAFILE = 'G:/OneDrive/TEU/Tensor/VIN_TensorFlow-master/data/gridworld_28x28.npz'
IMSIZE = 28
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

CONGESTIONINF = 1.5 # 拥挤影响常数
CONGESTION_GATE = 3 # 拥挤判断阈值
MULTIAGENTNUM = 20 # 多agent数量

def makeMulti(X_map, S1, S2, goal, agentNum):
    '''在空地上生成除了自己和终点以外的Agent
        返回值为[agentNum,2]大小的numpy'''
    goal = np.array(goal).tolist()
    agentList = []
    S_map = X_map
    mask = [[0 for i in range(IMSIZE)]for j in range(IMSIZE)]
    mask[goal[0]][goal[1]] = WALL
    mask[S2][S1] = WALL #虽然是自己但是用WALL代替···
    mask = tf.constant(mask)
    S_map = tf.where(mask == WALL,WALL,S_map)
    new_Multi = tf.where(S_map == NOMAL) # 获取可以生成Agent的位置
    agentNumMax = new_Multi.get_shape().as_list()[0]
    if(agentNumMax<=agentNum-1):
        print('ERROR:empty space is not enough!')
    else:
        agentIndex = random.sample(range(agentNumMax),agentNum-1)
        agentList.append([S2,S1])
        for i in range(agentNum-1):
            agentList.append(new_Multi[agentIndex[i]])
        agentList = np.array(agentList)# 转为numpy，由于存在[S1][S2]，将[S1][S2]转为[S1,S2]
        agentList = tf.constant(agentList) #再转为tensor
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
    
    agentStep = []
    agentRoad = []
    
    game = gamesystem.gamesystem(IMSIZE,BATCH_SIZE)
    
    #X_batch, S1_batch, S2_batch, y_batch = dataset.rollNEWMULTI(width=8,height=8,wallnum=10,goalnum=1,mapnum=BATCH_SIZE)
    X_batch, S1_batch, S2_batch, y_batch = dataset.next_batch(BATCH_SIZE)
    
    y_batch = tf.constant(y_batch)
    y_batch = tf.dtypes.cast(y_batch, dtype=tf.int64)
    
    logits, prob_actions,value = my_model.call(
            X_batch, S1_batch, S2_batch, VInum=VINUM)
    for i in range(BATCH_SIZE):
        startTime = time.time()
        print('/===========================BATCH',i,'START==========================\\')
        now_batch = tf.constant(X_batch[i])
        # 障碍物地图
        X_map = tf.slice(now_batch,[0,0,0],[IMSIZE,IMSIZE,1])
        X_map = tf.reshape(X_map,[-1,IMSIZE])
        # 原始价值地图
        origin_V_map = tf.slice(now_batch,[0,0,1],[IMSIZE,IMSIZE,1])
        origin_V_map = tf.reshape(origin_V_map,[-1,IMSIZE])
        goal = tf.where(origin_V_map == GOAL)[0]
        # 处理后价值地图
        V_map = tf.constant(value[i])
        V_map = tf.slice(V_map,[0,0,1],[IMSIZE,IMSIZE,1])
        V_map = tf.reshape(V_map,[-1,IMSIZE])
        # agent位置
        S1 = tf.dtypes.cast(S1_batch[i], dtype=tf.int64)
        S2 = tf.dtypes.cast(S2_batch[i], dtype=tf.int64)
        #print(i)
        #print(X_map)
        #print(V_map)
        #print(goal)
        #print(S1_batch[i],S2_batch[i])
        multiAgentList = makeMulti(X_map,S1,S2,goal,MULTIAGENTNUM)
        multiGame = multiagent.multiAgent()
        thisAgentStep,thisAgentRoad = multiGame.runMulti(X_map,V_map,multiAgentList,goal)
        
        X_mapList.append(X_map)
        V_mapList.append(V_map)
        goalList.append(goal)
        agentStep.append(thisAgentStep)
        agentRoad.append(thisAgentRoad)
        endTime = time.time()
        print('RUN_TIME:%s'%(endTime-startTime))
        print('\\===========================BATCH',i,'OVER===========================/')
    return agentStep,agentRoad


#trainset = mapmaker.map(width=IMSIZE,height=IMSIZE,wallnum=WALLNUMBER,goalnum=GOALNUMBER)
#testset = mapmaker.map(width=IMSIZE,height=IMSIZE,wallnum=WALLNUMBER,goalnum=GOALNUMBER)
#trainset = dataset.Dataset(filepath=DATAFILE, mode='train', imsize=IMSIZE)
testset = dataset.Dataset(filepath=DATAFILE, mode='train', imsize=IMSIZE)

RunVIN = tf.saved_model.load('./SavedModel/')
mean_err = 0.0
mean_loss = 0.0

agentStep,agentRoad = eval(RunVIN, testset)
#roadList = np.array(roadList)
#agentRoad = np.array(agentRoad)
print(agentStep)
print(agentRoad)
