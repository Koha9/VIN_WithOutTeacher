import tensorflow as tf
import numpy as np
import random
import sys
import time

NOMAL = 0
ME = 2
WALL = 1
GOAL = 10
class gamesystem():
    def __init__(self, imSIZE = 8,BatchSize = 128):
        self.imSIZE = imSIZE
        self.BatchSize = BatchSize


    def up(self, myX, myY):
        nowX = myX
        nowY = myY - 1
        return nowX, nowY


    def up_right(self, myX, myY):
        nowX = myX + 1
        nowY = myY - 1
        return nowX, nowY


    def right(self, myX, myY):
        nowX = myX + 1
        nowY = myY
        return nowX, nowY


    def down_right(self, myX, myY):
        nowX = myX + 1
        nowY = myY + 1
        return nowX, nowY


    def down(self, myX, myY):
        nowX = myX
        nowY = myY + 1
        return nowX, nowY


    def down_left(self, myX, myY):
        nowX = myX - 1
        nowY = myY + 1
        return nowX, nowY


    def left(self, myX, myY):
        nowX = myX - 1
        nowY = myY
        return nowX, nowY


    def up_left(self, myX, myY):
        nowX = myX - 1
        nowY = myY - 1
        return nowX, nowY


    def move(self, singlemap, myX, myY, step=0, action=0):
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

    def focusSize(self, X,Y):
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
        if X >= self.imSIZE-1:
            sizeX -=1
        if Y >= self.imSIZE-1:
            sizeY -=1
        return beginX,beginY,sizeX,sizeY
    def valueSlice(self, singleMap,valueMap,myX,myY):
        '''切片myX与my坐标周围的价值'''
        inputMap = valueMap
        wallMap = singleMap
        myX = int(myX)
        myY = int(myY)
        #print('MYX,MYY',myX,myY)
        outputmap = tf.dtypes.cast(inputMap,tf.int32)
        #print(outputmap)
            
        # 修改价值地图墙的位置为最小值+10
        wallsize=-999999
        inputMap = tf.where(wallMap == WALL,wallsize,inputMap)
        #print('MYX,MYY',myX,myY)
        outputmap = tf.dtypes.cast(inputMap,tf.int32)
        #print(outputmap)
        
        # 修改价值地图我的位置为最小值
        minsize = -sys.maxsize
        mask = [[0 for i in range(self.imSIZE)]for j in range(self.imSIZE)]
        mask[myY][myX] = minsize
        mask = tf.constant(mask)
        inputMap = tf.where(mask == minsize,minsize,inputMap)
        outputmap = tf.dtypes.cast(inputMap,tf.int32)
        #print(outputmap)

        
        #获取切片起始位置与size
        sliceBeginX,sliceBeginY,sliceSizeX,sliceSizeY = self.focusSize(myX,myY)
        focusValue = tf.slice(inputMap,[sliceBeginY,sliceBeginX],[sliceSizeY,sliceSizeX])
        #print(focusValue)
        return focusValue

    def getNewMe(self, singleMap, valueMap, nowMe):
        '''移动到新的最大value的地方'''
        myX = nowMe[1]
        myY = nowMe[0]
        focusValue = self.valueSlice(singleMap,valueMap,myX,myY)
        minsize = -sys.maxsize
        meInFocusValue = tf.where(focusValue == minsize) # 现在的位置 in sliceValuemap
        maxValueIndex = tf.where(focusValue == tf.reduce_max(focusValue)) # 最大value in slicevaluemap
        #print('MAXVALUE',tf.reduce_max(focusValue))
        #print('MAXINDEX',maxValueIndex[0])
        #print('MY_INDEX',meInFocusValue[0])
        if tf.size(maxValueIndex) > 2:
            maxValueIndex = maxValueIndex[random.randint(0,np.array(tf.shape(maxValueIndex)))]
        
        dx = maxValueIndex[0][1]-meInFocusValue[0][1]
        dy = maxValueIndex[0][0]-meInFocusValue[0][0]
        
        newNowMyX = myX + dx
        newNowMyY = myY + dy
        newNowMe = [newNowMyY,newNowMyX]
        return newNowMe

    def rungame(self, mapList, valueMapList, myXList, myYList,goalList):
        roadList = []
        stepList = []
        err = 0
        for i in range(self.BatchSize):
            step = 0
            validStep = True
            nowMe = [myYList[i],myXList[i]]
            goal = goalList[i]
            singleMap = mapList[i]
            valueMap = valueMapList[i]
            road = [np.array(tf.dtypes.cast(nowMe,dtype=tf.int32)).tolist()]
            #print('Search i',i)
            while((nowMe[0] != goal[0] or nowMe[1] != goal[1]) and validStep == True):
                #print('GOAL',goal)
                nowMe = self.getNewMe(singleMap,valueMap,nowMe)
                road.append(np.array(nowMe).tolist())
                step +=1
                
                if step>=self.imSIZE*4:
                    validStep = False
                if (singleMap[nowMe[0]][nowMe[1]]==WALL or validStep == False):
                    err+=1
            roadList.append(road)
            stepList.append(step)
        #roadList = np.array(roadList)
        return roadList,stepList
    
    def runSingleGame(self, singleMap, valueMap, nowMe,goal):
        '''获取单场game的路径与step，返回值为list与int'''
        err = 0
        step = 0
        validStep = True
        road = []
        while((nowMe[0] != goal[0] or nowMe[1] != goal[1]) and validStep == True):
            nowMe = self.getNewMe(singleMap,valueMap,nowMe)
            road.append(np.array(nowMe).tolist())
            step +=1
            
            if step>=self.imSIZE*4:
                validStep = False
            if (singleMap[nowMe[0]][nowMe[1]]==WALL or validStep == False):
                err+=1
        return road,step