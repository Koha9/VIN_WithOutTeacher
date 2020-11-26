import tensorflow as tf
import numpy as np
import random
import sys
import time

NOMAL = 0
ME = 2
WALL = 1
GOAL = 10

def up(myX, myY):
    nowX = myX
    nowY = myY - 1
    return nowX, nowY


def up_right(myX, myY):
    nowX = myX + 1
    nowY = myY - 1
    return nowX, nowY


def right(myX, myY):
    nowX = myX + 1
    nowY = myY
    return nowX, nowY


def down_right(myX, myY):
    nowX = myX + 1
    nowY = myY + 1
    return nowX, nowY


def down(myX, myY):
    nowX = myX
    nowY = myY + 1
    return nowX, nowY


def down_left(myX, myY):
    nowX = myX - 1
    nowY = myY + 1
    return nowX, nowY


def left(myX, myY):
    nowX = myX - 1
    nowY = myY
    return nowX, nowY


def up_left(myX, myY):
    nowX = myX - 1
    nowY = myY - 1
    return nowX, nowY


def move(singlemap, myX, myY, step=0, action=0):
    step += 1
    # 用dic模仿switch
    action_dic = {
        0: up,
        1: up_right,
        2: right,
        3: down_right,
        4: down,
        5: down_left,
        6: left,
        7: up_left
    }
    movement = action_dic.get(action)
    if movement:
        nowX, nowY = movement(myX, myY)
    
    if singlemap[0][nowY][nowX] == 1:# 行进方向为墙，返回原来位置并Feedback = 1
        Feedback = 0
        nowX = myX
        nowY = myY
    else:
        if singlemap[1][nowY][nowX] == 0:
            Feedback = 1
        else:
            Feedback = 2
    return nowX, nowY, step, Feedback

def focusSize(X,Y,imSIZE):
    '''切割x,y周围3x3所需的起始点与size，并保证不出图边界'''
    beginX = X - 1
    beginY = Y - 1
    sizeX = 3
    sizeY = 3
    if beginX < 0:
        beginX = 0
        sizeX -=1
    if beginY < 0:
        beginY = 0
        sizeY -=1
    if X >= imSIZE-1:
        sizeX -=1
    if Y >= imSIZE-1:
        sizeY -=1
    return beginX,beginY,sizeX,sizeY
def valueSlice(singleMap,valueMap,myX,myY,imSIZE):
    '''切片myX与my坐标周围的价值'''
    inputMap = valueMap
    wallMap = singleMap
    myX = int(myX)
    myY = int(myY)
    print('MYX,MYY',myX,myY)
    outputmap = tf.dtypes.cast(inputMap,tf.int32)
    print(outputmap)
        
    # 修改价值地图墙的位置为最小值+10
    wallsize=-999999
    inputMap = tf.where(wallMap == WALL,wallsize,inputMap)
    print('MYX,MYY',myX,myY)
    outputmap = tf.dtypes.cast(inputMap,tf.int32)
    print(outputmap)
    
    # 修改价值地图我的位置为最小值
    minsize = -sys.maxsize
    myZero = [[0 for i in range(imSIZE)]for j in range(imSIZE)]
    myZero[myY][myX] = minsize
    myZero = tf.constant(myZero)
    inputMap = tf.where(myZero == minsize,minsize,inputMap)
    outputmap = tf.dtypes.cast(inputMap,tf.int32)
    print(outputmap)

    
    #获取切片起始位置与size
    sliceBeginX,sliceBeginY,sliceSizeX,sliceSizeY = focusSize(myX,myY,imSIZE)
    focusValue = tf.slice(inputMap,[sliceBeginY,sliceBeginX],[sliceSizeY,sliceSizeX])
    print(focusValue)
    return focusValue

def getNewMe(singleMap, valueMap, nowMe,imSIZE):
    '''移动到新的最大value的地方'''
    myX = nowMe[1]
    myY = nowMe[0]
    focusValue = valueSlice(singleMap,valueMap,myX,myY,imSIZE)
    minsize = -sys.maxsize
    meInFocusValue = tf.where(focusValue == minsize) # 现在的位置 in sliceValuemap
    maxValueIndex = tf.where(focusValue == tf.reduce_max(focusValue)) # 最大value in slicevaluemap
    print('MAXVALUE',tf.reduce_max(focusValue))
    print('MAXINDEX',maxValueIndex[0])
    print('MY_INDEX',meInFocusValue[0])
    if tf.size(maxValueIndex) > 2:
        maxValueIndex = maxValueIndex[random.randint(0,tf.shape(maxValueIndex))]
    
    dx = maxValueIndex[0][1]-meInFocusValue[0][1]
    dy = maxValueIndex[0][0]-meInFocusValue[0][0]
    
    newNowMyX = myX + dx
    newNowMyY = myY + dy
    newNowMe = [newNowMyY,newNowMyX]
    return newNowMe

def rungame(mapList, valueMapList, myXList, myYList,goalList,imSIZE,BATCHSIZE):
    roadList = []
    stepList = []
    err = 0
    for i in range(BATCHSIZE):
        step = 0
        validStep = True
        nowMe = [myYList[i],myXList[i]]
        goal = goalList[i]
        singleMap = mapList[i]
        valueMap = valueMapList[i]
        road = [nowMe]
        while((nowMe[0] != goal[0] or nowMe[1] != goal[1]) and validStep == True):
            print('-------------ROUND START--------------')
            print('GOAL',goal)
            nowMe = getNewMe(singleMap,valueMap,nowMe,imSIZE)
            road.append(nowMe)
            step +=1
            
            if step>=imSIZE*4:
                validStep = False
            if (singleMap[nowMe[0]][nowMe[1]]==WALL or validStep == False):
                err+=1
            print('NOW STEP')
            print(step)
            print('NOW ERR')
            print(err)
            print('-----------------------------------')
        roadList.append(road)
        stepList.append(step)
        print("ROUND OVER ")
        print(i)
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    return roadList,stepList
            