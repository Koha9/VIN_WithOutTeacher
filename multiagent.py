import numpy as np
import tensorflow as tf
import gamesystem
import os
import random
import gamesystem

NOMAL = 0
ME = 2
WALL = 1
GOAL = 10
CONGESTIONINF = 1.5 # 拥挤影响常数
CONGESTION_GATE = 3 # 拥挤判断阈值

class multiAgent():
    def __init__(self, imSIZE=8, batchSize=128):
        self.imSIZE = imSIZE
        self.batchSize = batchSize
        self.ONCongeCount = 0

    def getCongestion(self, singleMap, agentNowList):
        '''获取混雑区域坐标，返回一个Dic [拥挤区坐标]:坐标区域所拥有最大agent数//
        singlemap = tensor, agentNowList = list'''
        congestion = {}  # 混雑区域坐标
        isFirst = True
        agentMap = [[0 for i in range(self.imSIZE)]for j in range(self.imSIZE)]
        for i in range(len(agentNowList)):
            agentMap[agentNowList[i][0]][agentNowList[i][1]] = ME
        agentMap = tf.constant(agentMap)
        for j in range(1, self.imSIZE-1):
            for i in range(1, self.imSIZE-1):
                sliceMap = tf.slice(agentMap, [j-1, i-1], [3, 3])
                agentNumInSlice = tf.where(sliceMap == ME)
                numOfAgent = np.array(tf.shape(agentNumInSlice)[0]).tolist()  # 此区域agent数量
                if numOfAgent >= CONGESTION_GATE:  # 判定为拥挤区域
                    sliceMapIndex = [[j-1, i-1], [j-1, i], [j-1, i+1], [j, i-1],
                                     [j, i], [j, i+1], [j+1, i-1], [j+1, i], [j+1, i+1]]
                    for i in range(len(sliceMapIndex)):
                        if (sliceMapIndex[i][0], sliceMapIndex[i][1]) in congestion:
                            congestion[sliceMapIndex[i][0], sliceMapIndex[i][1]] = max(
                                numOfAgent, congestion[sliceMapIndex[i][0], sliceMapIndex[i][1]])
                        else:
                            congestion[sliceMapIndex[i][0],
                                       sliceMapIndex[i][1]] = numOfAgent
        return congestion

    def getThisSingleMap(self, singleMap, agentNowList, thisAgent):
        '''获取对应该agent的障碍物地图，将其他agent视为障碍物追加于其中,返回值为tensor//
        singleMap = tensor , agentNowList = list , thisAgent = list
        '''
        thisSingleMap = singleMap
        agentNowList.remove(thisAgent)
        chacheMap = np.array(thisSingleMap).tolist()
        for i in range(len(agentNowList)):
            chacheMap[agentNowList[i][0]][agentNowList[i][1]] = WALL
        thisSingleMap = tf.constant(thisSingleMap)
        return thisSingleMap

    def getDistance(self, aCDNT, bCDNT):
        '''计算两点间距离,返回一个float32
        aCDNT = list,bCDNT = list'''
        xDis = float(abs(aCDNT[1] - bCDNT[1]))
        yDis = float(abs(aCDNT[0] - bCDNT[0]))
        if xDis == yDis:
            return xDis*np.sqrt(2)
        else:
            return xDis + yDis + (np.sqrt(2) - 1.0) * min(xDis, yDis)

    def getRoad(self, singleMap, valueMap, singleAgent, goal):
        '''获取除了自己以外的仮経路的坐标list,返回一个list型'''
        gamesys = gamesystem.gamesystem(self.imSIZE, self.batchSize)
        road, step = gamesys.runSingleGame(
            singleMap, valueMap, singleAgent, goal)
        return road

    def checkIsCongestion(self, road, congestion):
        '''检查是否与Congestion相同，并返回相同处的坐标，返回值为np
        road = list, congestion = Dic'''
        road = np.array(road)
        uniqueRoad = np.unique(road, axis=0)
        if len(road) != len(uniqueRoad): # 检查road是否存在重复并去重
            road = uniqueRoad
        if congestion:
            congestionCoordTuple = list(congestion)  # 取出key转化为list,内部保存数据仍然为tuple
            congestionCoord = [list(congestionCoordTuple[0])]
            for i in range(1,len(congestionCoordTuple)): # 将tuple转换为list
                congestionCoord.append(list(congestionCoordTuple[i]))
            totalCoord = np.append(congestionCoord, road, axis=0)  # 混雑和路径坐标合并为同一np
            only, counts = np.unique(totalCoord, return_counts=True, axis=0)
            index = np.where(counts > 1)
            ISCongestionCoord = only[index]
        else:
            ISCongestionCoord = []
        return ISCongestionCoord

    def congestionInf(self, valueMap, agentNow, onCongestion, congestion, actionCoord):
        '''根据障碍物坐标更新价值地图，仅更新周围8格
        valuemap = tensor, agentNow = list,onCongestion = np, targetCoord = list'''
        newValueMap = np.array(valueMap).tolist()
        Inf = 0.0
        for i in range(len(onCongestion)):
            agnetNumInCong = congestion[onCongestion[i][0], onCongestion[i][1]]
            distance = self.getDistance(onCongestion[i], agentNow)
            if distance == 0:
                Inf += 0
            else:
                Inf += agnetNumInCong*CONGESTIONINF/distance
        newValueMap[actionCoord[0]][actionCoord[1]] -= Inf
        newValueMap = tf.constant(newValueMap)
        return newValueMap

    def checkSearchOver(self, lastAction, nowAction, checkTimes, goal):
        '''检查是否满足结束条件，返回bool型
        lastAction = list, nowAction = list, checkTimes = int'''
        if lastAction == nowAction or checkTimes >= 16 or nowAction == goal:
            return False
        else:
            return True

    def searchAction(self, singleMap, valueMap, agentSearch, agentRoad, goal):
        '''搜寻该轮agents最佳行进方向,并返回一个dict[agent原坐标]:动作方向坐标.
        singlemap = tensor, valuemap = tensor, agentSearch = list, agentRoad = dict, goal = lsit'''
        agentAction = {}  # agent原坐标:动作
        agentDisDic = {}  # agent距离:原坐标
        agentNowDic = {}  # agent原坐标:现坐标
        agentNowList = []  # agent的现坐标list

        for i in range(tf.shape(agentSearch)[0]):  # 初始化
            agentNowList.append(
                agentRoad[agentSearch[i][0], agentSearch[i][1]][-1])
            agentNowDic[agentSearch[i][0], agentSearch[i][1]
                        ] = agentRoad[agentSearch[i][0], agentSearch[i][1]][-1]
            distance = self.getDistance(agentSearch[i], goal)
            if distance in agentDisDic: # 若存在距离相同agent
                temp = agentDisDic[distance]
                temp.append(agentSearch[i])
                agentDisDic[distance] = temp # 于同key值内添加新坐标
            else:
                agentDisDic[distance] = [agentSearch[i]]
        '''------------------此处或需要修改论文-----------------
        -------考虑将获取拥挤区list频率从每轮一次到每agent一次--------
        --------------------------------------------------------'''
        congestion = self.getCongestion(singleMap, agentSearch)  # 获取拥挤区域np
        for dis in sorted(agentDisDic):  # 按距离顺序执行搜寻
            numOfAgentInDis = len(agentDisDic[dis])
            indexOfDis = random.sample(range(numOfAgentInDis),numOfAgentInDis)
            for index in range(numOfAgentInDis):
                #print('/===========================ACTION SEARCH==========================\\')
                isSearchOver = False
                thisAgentOrigin = agentDisDic[dis][indexOfDis[index]] # 处理当agent距离相同时随机选择agent进行寻路
                thisAgentNow = agentNowDic[thisAgentOrigin[0], thisAgentOrigin[1]] # agent现在位置
                thisSingleMap = self.getThisSingleMap(
                    singleMap, agentNowList, thisAgentNow)  # 初始化即将使用的障碍物map
                thisValueMap = valueMap  # 初始化即将使用的valuemap
                #print('/===================state==================\\')
                #print('|thisAgentOrigin:',thisAgentOrigin)
                #print('|thisAgentNow   :',thisAgentNow,type(thisAgentNow))
                #print('|agentNowDic    :',agentNowDic)
                #print('|agentNowList   :',agentNowList)

                lastTemRoad = []  # 初始化前一回合模拟路线
                checkTimes = 0
                if not isSearchOver:
                    checkTimes += 1
                    tempRoad = self.getRoad(
                        thisSingleMap, thisValueMap, thisAgentNow, goal)
                    # 以获取路径是否存在3个以上的重复来判断死循环并跳出
                    uniqueTempRoad = np.unique(tempRoad,axis=0)
                    if len(tempRoad) - len(uniqueTempRoad) > 3:
                        agentAction = False
                        return agentAction
                    onCongestion = self.checkIsCongestion(tempRoad, congestion)
                    if len(onCongestion) > 0:  # 若不存在与拥挤区相交之坐标
                        #print('存在拥挤区!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        thisValueMap = self.congestionInf(
                            thisValueMap, thisAgentNow, onCongestion, congestion,tempRoad[0])  # 更新valuemap(自身周边的value)
                        '''------------------此处或需要修改论文-----------------
                        -------考虑将结束条件改为搜索次数达到上限而不是全部检查一次--------
                        --------------------------------------------------------'''
                        if checkTimes <= 1:
                            isSearchOver = False
                        else:
                            isSearchOver = self.checkSearchOver(
                                lastTemRoad[0], tempRoad[0], checkTimes, goal)
                        lastTemRoad = tempRoad
                    else:
                        #print('不存在拥挤区~')
                        isSearchOver = True
                else:
                    # 更新该agent现在坐标位置，使其不挡别的agent的道
                    agentNowList.remove(thisAgentNow)
                    agentNowList.append(tempRoad[0])
                # 将动作添加至动作dict
                agentAction[thisAgentOrigin[0], thisAgentOrigin[1]] = tempRoad[0]
                #print('\\========================ACTION SEARCH OVER========================/')
        return agentAction

    def getAgentSearch(self, agentList, agentRoad, goal):
        '''获取本轮需要更新的agent，即还未抵达goal的agent，返回一个list
        agentRoad = dict, goal = list'''
        agentSearch = []
        for i in range(len(agentList)):
            if agentRoad[agentList[i][0], agentList[i][1]][-1] != goal:
                agentSearch.append(agentList[i])
        return agentSearch

    def runMulti(self, singleMap, valueMap, agentList, goal):
        '''执行多Agent寻路，返回dict[agent原坐标]:[[路径]]  和  dict[agent原坐标]:[步数]
        singleMap = tensor, valueMap = tensor, agentList = tensor, goal = tensor'''
        searchTimes = 0
        goal = np.array(goal).tolist()
        agentList = np.array(agentList).tolist()
        agentStep = {}  # 记录步数
        agentRoad = {}  # 记录路程
        for i in range(len(agentList)):# 初始化agentStep和agentRoad
            agentStep[agentList[i][0], agentList[i][1]] = 0
            agentRoad[agentList[i][0], agentList[i][1]] = [agentList[i]]
        agentSearch = self.getAgentSearch(agentList,agentRoad,goal)  # 本轮需要更新的agent，即还未抵达goal的agent
        while len(agentSearch) > 0:
            #print('////////////////////////////NEW STAGE/////////////////////////////')
            #print('//////////////')
            #print('agentSearch:',agentSearch)
            searchTimes +=1
            agentAction = self.searchAction(
                singleMap, valueMap, agentSearch, agentRoad, goal)
            if not agentAction: # 当searchAction失败或陷入死循环时会返回False，此时执行跳出
                print('SEARCH ACTION FAILD,JUMP OUT')
                agentStep = {0:0}
                agentRoad = {0:0}
                return agentStep, agentRoad
            #print('AgentAction:',agentAction)
            #print('AgentSearch',agentSearch)
            #print('AgentStep:',agentStep)
            for i in range(len(agentSearch)):  # 更新step,raod
                agentStep[agentSearch[i][0], agentSearch[i][1]] += 1
                chacheRoad = agentRoad[agentSearch[i][0], agentSearch[i][1]]
                chacheRoad.append(
                    agentAction[agentSearch[i][0], agentSearch[i][1]])
                agentRoad[agentSearch[i][0], agentSearch[i][1]] = chacheRoad
            agentSearch = self.getAgentSearch(agentList,agentRoad,goal)
            #print('///////////////////////////STAGE OVER/////////////////////////////')
            print('searchTimes:',searchTimes)
            if searchTimes >= len(agentList): # 防止无限死循环
                print('SEARCH TIME OUT OF LIMIT,JUMP OUT')
                agentStep = {0:0}
                agentRoad = {0:0}
                return agentStep, agentRoad
        return agentStep, agentRoad