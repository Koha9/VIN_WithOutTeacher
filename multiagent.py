import numpy as np
import tensorflow as tf
import gamesystem

NOMAL = 0
ME = 2
WALL = 1
GOAL = 10

class multiAgent():
    def __init__(self,imSIZE = 8,batchSize = 128):
        self.imSIZE = imSIZE
        self.batchSize = batchSize
    
    def getCongestion(self,singleMap,agentList):
        '''获取混雑区域坐标，返回一个numpy型'''
        congestion = []
        isFirst = True
        agentMap = [[0 for i in range(self.imSIZE)]for j in range(self.imSIZE)]
        for i in range(agentList.get_shape().as_list()[0]):
            agentMap[agentList[i][0]][agentList[i][1]] = ME
        agentMap = tf.constant(agentMap)
        for j in range(1,self.imSIZE-1):
            for i in range(1,self.imSIZE-1):
                sliceMap = tf.slice(agentMap,[j-1,i-1],[3,3])
                agentNumInSlice = tf.where(sliceMap == ME)
                if tf.shape(agentNumInSlice)[0]>=3:
                    sliceMapIndex = [[j-1,i-1],[j-1,i],[j-1,i+1],[j,i-1],[j,i],[j,i+1],[j+1,i-1],[j+1,i],[j+1,i+1]]
                    if isFirst:
                        congestion = sliceMapIndex
                        isFirst = False
                    else:
                        congestion+=sliceMapIndex
        congestion = np.unique(congestion,axis=0) #去重
        return congestion
    
    def getThisSingleMap(self,):
        '''获取对应该agent的障碍物地图，将其他agent视为障碍物追加于其中'''
    
    def getDistance(self,agentCDNT,goalCDNT):
        '''计算两点间距离,返回一个float32'''
        xDis = float(abs(agentCDNT[1] - goalCDNT[1]))
        yDis = float(abs(agentCDNT[0] - goalCDNT[0]))
        if xDis == yDis:
            return xDis*np.sqrt(2)
        else:
            return xDis + yDis + (np.sqrt(2) - 1.0) * min(xDis, yDis)
        
    def getRoad(self,singleMap,valueMap,singleAgent,goal):
        '''获取除了自己以外的仮経路的坐标list,返回一个list型'''
        gamesys = gamesystem.gamesystem(self.imSIZE,self.batchSize)
        road,step = gamesys.runSingleGame(singleMap,valueMap,singleAgent,goal)
        return road
    
    
    def checkIsCongestion(self,road,congestion):
        '''检查是否与Congestion相同，并返回相同处的坐标，返回值为np'''
        raod = np.array(road)
        totalCoord = np.append(congestion,road,axis=0) # 混雑和路径坐标合并为同一np
        only,counts = np.unique(totalCoord,return_counts=True,axis=0)
        index = np.where(counts>1)
        congestionCoord = only[index]
        return congestionCoord
    
    def congestionInf(self,valueMap,singleAgent,onCongestion):
        '''根据障碍物坐标更新价值地图，仅更新周围8格'''
        newValueMap = valueMap
        return newValueMap

    def checkSearchOver(self,):
        '''检查寻路是否结束并返回True or False'''
        
    def doMove(self,):
        '''执行行动'''
    
        
    def searchAction(self,singleMap,valueMap,agentList,goal):
        '''搜寻该轮agents最佳行进方向,并返回一个dic'''
        '''TODO：修改返回dic为[agent]：action'''
        agentAction = {}
        agentDict = {}
        congestion = self.getCongestion(singleMap,agentList)
        for i in range(tf.shape(agentList)[0]):
            agentDict[self.getDistance(agentList[i],goal)] = agentList[i]
        for dis in sorted(agentDict):
            isSearchOver = False
            thisSingleMap = getThisSingleMap()
            thisAgent = agentDict[dis]
            thisValueMap = valueMap
            print('DO SOMETHING')
            if not isSearchOver:
                tempRoad = self.getRoad(thisSingleMap,thisValueMap,thisAgent,goal)
                onCongestion = self.checkIsCongestion(tempRoad,congestion)
                thisValueMap = self.congestionInf(thisValueMap,thisAgent,onCongestion) # 更新valuemap(自身周边的value)
                isSearchOver = self.checkSearchOver()
            agentAction[thisAgent[0],thisAgent[1]] = tempRoad[0]
        return agentAction
    def getAgentSearch(self,):
        '''获取本轮需要更新的agent，即还未抵达goal的agent，返回一个list'''

    
    def runMulti(self,singleMap,valueMap,agentList,goal):
        agentList = np.array(agentList).tolist()
        agentStep = {} # 记录步数
        agentRoad = {} # 记录路程
        for i in range(len(agentList)):
            agentStep[agentList[i][0],agentList[i][1]] = 0
            agentRoad[agentList[i][0],agentList[i][1]] = [agentList[i]]
        agentSearch = self.getAgentSearch() #本轮需要更新的agent，即还未抵达goal的agent
        while len(agentSearch) > 0:
            agentAction = self.searchAction(singleMap,valueMap,agentList,goal)
            for i in range(len(agentSearch)):# 更新step,raod
                agentStep[agentSearch[i][0],agentSearch[i][1]]+=1
                chacheRoad = agentRoad[agentSearch[i][0],agentSearch[i][1]]
                chacheRoad.append(agentAction[agentSearch[i][0],agentSearch[i][1]])
                agentRoad[agentSearch[i][0],agentSearch[i][1]] = chacheRoad
            agentSearch = self.getAgentSearch()
        return agentList,agentRoad