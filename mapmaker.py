import random
import numpy as np
import json
import math
'''
0 = white
1 = me
2 = wall
3 = goal
'''
WIDTH = 25
HEIGHT = 25
WALLNUMBER = 15
GOALNUMBER = 1

NOMAL = 0
ME = 2
WALL = 1
GOAL = 50


class map():
    def __init__(self, width=8, height=8, wallnum=10, goalnum=1):
        # default
        if goalnum > 4:
            print('傻逼你的Goal大于4！请设定一个小于4的整数')
            return
        else:
            self.numofGOAL = goalnum  # 终点数量
        if wallnum >= (width*height - 5):
            print('全是墙！')
            return
        else:
            self.numofWALL = wallnum  # 墙的数量

        self.width = width  # 宽
        self.height = height  # 高
        self.wallnum = wallnum
        self.goalnum = goalnum

    def rollME(self, singlemap, singleWidth=8, singleHeight=8):
        '''Roll一个不在墙或者终点内的坐标，并返回int坐标'''
        myselfX = 0
        myselfY = 0
        inGoal = True
        while(inGoal):
            myselfX = random.randint(1, singleWidth-2)
            myselfY = random.randint(1, singleHeight-2)
            if (singlemap[myselfY][myselfX] == NOMAL):
                inGoal = False
        return myselfX, myselfY

    def rollGOAL(self, singlemap, goalmap, myselfX, myselfY, num=1):
        '''Roll num个终点坐标，并应返回'''
        goalX = 0
        goalY = 0
        inGoal = True
        # 随机一个不在墙上且不和agent重合的goal坐标
        while(inGoal):
            goalX = random.randint(1, self.width-2)
            goalY = random.randint(1, self.height-2)
            if (singlemap[goalX][goalY] == NOMAL and (goalX != myselfX or goalY != myselfY)):
                inGoal = False
        goalmap[goalY][goalX] = GOAL  # 应用到goalmap中
        return goalmap

    def rollWALL(self, singlemap, num, width=8, height=8):
        '''Roll num个不在墙或者终点内的坐标，并返回singlemap'''
        listofWALL = []
        sgmap = singlemap

        for i in range(1, width-1):  # 宽
            for j in range(1, self.height-1):  # 高
                # [[1,1],[1,2],[1,3]....] 可以生成墙的所有坐标铺平
                listofWALL.append([i, j])
        # 墙坐标的索引，对应listofWALL
        WALLnum = random.sample(range((height - 2) * (width - 2)), num)

        for i in range(num):
            sgmap[listofWALL[WALLnum[i]][1]
                  ][listofWALL[WALLnum[i]][0]] = WALL  # 应用于sgmap
        return sgmap

    def azimuthAngle(self, x1,  y1,  x2,  y2):
        angle = 0.0;
        dx = x2 - x1
        dy = y2 - y1
        if  x2 == x1:
            angle = math.pi / 2.0 # 真上
            if  y2 == y1 :
                angle = 0.0 
            elif y2 < y1 :
                angle = 3.0 * math.pi / 2.0 # 直下
        elif x2 > x1 and y2 > y1: # 1 象限
            angle = math.atan(dy / dx)
        elif  x2 > x1 and  y2 < y1 : # 4 象限
            angle = 3* math.pi / 2 + math.atan(-dx / dy)
        elif  x2 < x1 and y2 < y1 : # 3 象限
            angle = math.pi + math.atan(dy / dx)
        elif  x2 < x1 and y2 > y1 : # 2 象限
            angle = math.pi / 2.0 + math.atan(dx / -dy)
        return (angle/math.pi)
    
    def getLable(self, myX, myY, goalX, goalY):
        angle = self.azimuthAngle(myX,myY,goalX,goalY)
        
        action = 0
        if(angle<1/8 or angle>=15/8):
            action = 0
        elif(angle>1/8 and angle<=3/8):
            action = 1
        elif(angle>3/8 and angle<=5/8):
            action = 2
        elif(angle>5/8 and angle<=7/8):
            action = 3
        elif(angle>7/8 and angle<=9/8):
            action = 4
        elif(angle>9/8 and angle<=11/8):
            action = 5
        elif(angle>11/8 and angle<=13/8):
            action = 6
        elif(angle>13/8 and angle<=15/8):
            action = 7
        
        return action

    def rollNEWSINGLE(self, singleWidth=8, singleHeight=8, wallnum=20, goalnum=1):
        '''ROLL一个新地图并返回'''
        wallmap = [[0 for col in range(singleWidth)] for row in range(
            singleHeight)]  # 单张地图，默认width x length大小的list，数据为0
        for i in range(singleHeight):
            for j in range(singleWidth):
                if(i == 0 or i == singleHeight-1 or j == 0 or j == singleWidth-1):
                    wallmap[i][j] = WALL  # 外围一圈墙
        goalmap = [[0 for col in range(singleWidth)] for row in range(
            singleHeight)]  # 目标点地图，默认width x length大小的list，数据为0

        wallmap = self.rollWALL(
            wallmap, wallnum, singleWidth, singleHeight)  # 随机墙
        myX, myY = self.rollME(wallmap, singleWidth,
                               singleHeight)  # 随机agent出生位置
        goalmap = self.rollGOAL(wallmap, goalmap, myX,
                                myY, goalnum)  # 随机goal点

        # 转为numpy，并reshape为[height,width,2]
        wallmap = np.array(wallmap)
        goalmap = np.array(goalmap)
        goalX = np.where(goalmap == 10)[1]
        goalY = np.where(goalmap == 10)[0]
        lable = self.getLable(myX,myY,goalX,goalY)
        wallmap = wallmap.reshape(-1)
        goalmap = goalmap.reshape(-1)
        singlemap = np.stack((wallmap, goalmap), 1)
        singlemap = singlemap.reshape((singleWidth, singleWidth, 2))
        
        
        
        return singlemap.tolist(), myX, myY, lable

    def rollNEWMULTI(self, width=8, height=8, wallnum=20, goalnum=1, mapnum=2):
        mapset = []
        myX = []
        myY = []
        lable = []

        for i in range(mapnum):
            singlemap, singleX, singleY, singleLable = self.rollNEWSINGLE(
                width, height, wallnum, goalnum)
            mapset.append(singlemap)
            myX.append(singleX)
            myY.append(singleY)
            lable.append(singleLable)

        lable = np.array(lable,dtype = np.float32)
        mapset = np.array(mapset, dtype = np.float32)
        myX = np.array(myX, dtype = np.float32)
        myY = np.array(myY, dtype = np.float32)
        return mapset, myX, myY, lable

    def save(self, maplist=[], myXlist=[], myYlist=[], filename='number.json'):
        file_name = 'number.json'
        list = [['name', 'length', 3, 2],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0], ]
        with open(file_name, 'w') as file_object:
            json.dump(list, file_object)
        print('SAVED')

    def load(self):
        return
