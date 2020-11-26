import sys
import time

import numpy as np
import Astar_point

NOMAL = 0
ME = 2
WALL = 1
GOAL = 50

class AStar:
    def __init__(self, map, myX, myY, goalX, goalY, width,height):
        self.map = map
        self.open_set = []
        self.close_set = []
        self.myX = myX
        self.myY = myY
        self.goalX = goalX
        self.goalY = goalY
        print(self.map)
        print(self.myX)
        print(self.myY)
        print(self.goalX)
        print(self.goalY)
        time.sleep(1)
        
        self.width = width
        self.height = height

    def baseCost(self, nowNODE):  
        '''now to my'''
        x_dis = abs(nowNODE.x-self.myX)
        y_dis = abs(nowNODE.y-self.myY)
        # Distance to start point
        return x_dis + y_dis + (np.sqrt(2) - 1) * min(x_dis, y_dis)

    def heuristicCost(self, nowNODE):  
        '''now to goal'''
        x_dis = abs(nowNODE.x-self.goalX)
        y_dis = abs(nowNODE.y-self.goalY)
        # Distance to end point
        return x_dis + y_dis + (np.sqrt(2) - 1) * min(x_dis, y_dis)

    def totalCost(self, nowNODE): 
        '''与终点和起点的距离和 f(n)'''
        return self.baseCost(nowNODE) + self.heuristicCost(nowNODE)
    
    def checkValid(self,x,y): 
        '''检查node是否于地图内'''
        if x < 0 or y < 0:
            return False
        elif x >= self.width or y >= self.height:
            return False
        elif self.map[y][x] == WALL:
            return False
        else:
            return True
        
    def checkInlist(self, nowNODE, nodeLIST):
        '''检查nowNODE是否于nodeLIST内'''
        for node in nodeLIST:
            if node.x == nowNODE.x and node.y == nowNODE.y:
                return True
            else:
                return False
    def checkOpen(self, nowNODE):
        '''检查nowNODE是否open'''
        return self.checkInlist(nowNODE, self.open_set)
    def checkClose(self, nowNODE):
        '''检查nowNODE是否close'''
        return self.checkInlist(nowNODE, self.close_set)
    def checkStart(self, nowNODE):
        '''检查nowNODE是否于start'''
        return nowNODE.x == 0 and nowNODE.y ==0
    def checkEnd(self, nowNODE):
        '''检查nowNODE是否于end'''
        return nowNODE.x == self.goalX and nowNODE.y == self.goalY
    
    def ProcessPoint(self, x, y, parent):
        if not self.checkValid(x, y):
            return # Do nothing for invalid point
        p = Astar_point.point(x, y)
        if self.checkClose(p):
            return # Do nothing for visited point
        print('Process Point [', p.x, ',', p.y, ']', ', cost: ', p.cost)
        if not self.checkOpen(p):
            p.parent = parent
            p.cost = self.totalCost(p)
            self.open_set.append(p)

    def selectHightNode(self):
        '''返回现在最高value的node地址'''
        index = 0
        selected_index = -1
        min_cost = sys.maxsize
        for p in self.open_set:
            cost = self.totalCost(p)
            if cost < min_cost:
                min_cost = cost
                selected_index = index
            index += 1
        return selected_index

    def BuildPath(self, p, start_time):
        '''以parent向上回溯找到path'''
        path = []
        r = 0
        while True:
            path.insert(0, p) # Insert first
            if self.checkStart(p):
                break
            else:
                p = p.parent
        for p in path:
            r+=1
            print(p)
            print(p.x)
            print(p.y)
            if r == 2 :
                return p.x, p.y
        end_time = time.time()
        print('===== Algorithm finish in', int(end_time-start_time), ' seconds')

    def runAstar(self):
        start_time = time.time()
        start_point = Astar_point.point(self.myX, self.myY)
        start_point.cost = 0

        self.open_set.append(start_point)
        while True:
            index = self.selectHightNode()
            if index < 0:
                print('No path found, algorithm failed!!!')
                return
            p = self.open_set[index]
            
            if self.checkEnd(p):
                return self.BuildPath(p,start_time)

            del self.open_set[index]
            self.close_set.append(p)

            # Process all neighbors
            x = p.x
            y = p.y
            self.ProcessPoint(x-1, y+1, p)
            self.ProcessPoint(x-1, y, p)
            self.ProcessPoint(x-1, y-1, p)
            self.ProcessPoint(x, y-1, p)
            self.ProcessPoint(x+1, y-1, p)
            self.ProcessPoint(x+1, y, p)
            self.ProcessPoint(x+1, y+1, p)
            self.ProcessPoint(x, y+1, p)
            print('ROUND OVER')
            