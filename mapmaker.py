import random
import numpy as np
import json
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
GOAL = 10


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
        self.maplist = []  # 地图库 n x width x height x 2
        self.singlemap = []  # 单一地图 width x height

    def rollME(self, singlemap, singleWidth=8,singleHeight=8):
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

    def rollNEWSINGLE(self, singleWidth=8, singleHeight=8, wallnum=25, goalnum=1):
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
        myX, myY = self.rollME(wallmap, singleWidth,singleHeight)  # 随机agent出生位置
        goalmap = self.rollGOAL(wallmap, goalmap, myX,
                                myY, goalnum)  # 随机goal点

        # 转为numpy，并reshape为[height,width,2]
        wallmap = np.array(wallmap)
        goalmap = np.array(goalmap)
        print(wallmap)
        print(goalmap)
        wallmap = wallmap.reshape(-1)
        goalmap = goalmap.reshape(-1)
        singlemap = np.stack((wallmap,goalmap),1)
        singlemap = singlemap.reshape((8,8,2))

        return singlemap
        print('RANDOM SINGLEMAP COMPLETE')

    def addtoMAPLIST(self):
        '''添加singlemap地图到maplist中'''
        self.maplist.append(self.singlemap)
        print('MAP ADDED TO LIST')

    def save(self, filename):
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

    def setasNOMAL(self, x, y):
        '''将x,y坐标在地图中设置为0(道路白),setasNOMAL(int,int)'''
        self.singlemap[x][y] = NOMAL

    def setasME(self, x, y):
        '''将x,y坐标在地图中设置为1(我),setasME(int,int)'''
        self.singlemap[x][y] = ME

    def setasGOAL(self, x, y):
        '''将x,y坐标在地图中设置为3(终点绿),setasGOAL(int,int)'''
        self.singlemap[x][y] = GOAL

    def getheight(self):
        return self.height

    def getwidth(self):
        return self.width

    def getCORNER(self, num):
        if num == 0:
            ray1 = [0, 0]
            return ray1
        elif num == 1:
            ray1 = [0, self.width-1]
            return ray1
        elif num == 2:
            ray1 = [self.height-1, self.width-1]
            return ray1
        elif num == 3:
            ray1 = [self.height-1, 0]
            return ray1


map1 = map(8,8,20,1)
print(map1.rollNEWSINGLE())
