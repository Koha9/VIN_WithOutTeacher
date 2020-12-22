import tensorflow as tf
import numpy as np
import os
import random
import gamesystem
import vin_multiGoal
import dataset

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # USE CPU

DATAFILE = 'G:/OneDrive/TEU/Tensor/VIN_TensorFlow-master/data/gridworld_28x28.npz'
IMSIZE = 28
WALLNUMBER = 30
GOALNUMBER = 4
LEARNING_RATE = 0.003
EPOCHS = 30
VINUM = 36
CH_I = 2
CH_H = 150
CH_Q = 10
BATCH_SIZE = 1
TRAINROUND = 600
TESTROUND = 1
USE_LOG = False
LOG_DIR = '.log/'
NOMAL = 0
ME = 2
WALL = 1
GOAL = 10
CONGESTIONINF = 1.5 # 拥挤影响常数
CONGESTION_GATE = 3 # 拥挤判断阈值
checkpoint_save_path = "./MultiGoalmodel/checkpoint/Vin.ckpt"

VIN = vin_multiGoal.VIN()
VIN.load_weights(checkpoint_save_path)


'''#X_map = tf.constant([[1., 1., 1., 1., 1., 1., 1., 1.],
                     [1., 0., 0., 0., 1., 0., 0., 1.],
                     [1., 0., 1., 0., 0., 0., 0., 1.],
                     [1., 1., 0., 0., 0., 0., 0., 1.],
                     [1., 1., 0., 1., 0., 1., 0., 1.],
                     [1., 0., 0., 1., 0., 1., 0., 1.],
                     [1., 0., 0., 0., 1., 0., 0., 1.],
                     [1., 1., 1., 1., 1., 1., 1., 1.]])
valueMap = tf.constant([[-2.5358171,   0.38238844, 10.299398,   22.275322,    9.626422,   23.438942,
                         10.446101,    6.2384744],
                        [-0.6641493,  11.328682,   25.407427,   39.544247,   58.860752,   37.445496,
                         20.883001,    9.328241],
                        [-0.7318544,  25.291689,   42.730484,   63.667698,   86.36546,    59.698338,
                         34.742287,   18.05207],
                        [-2.8683624,   5.368464,   62.70845,    84.17922,    97.84784,    79.091835,
                         53.050903,   29.90517],
                        [-6.356472,    7.976848,   46.13208,    44.112038,   89.02107,    49.01145,
                         35.42552,    15.194459],
                        [-0.9579588,  17.481594,   32.43278,    22.537163,   69.02457,    34.784977,
                         21.192383,    6.440092],
                        [-3.1106799,   5.739085,   15.110557,   29.534292,   27.589031,   18.979767,
                         10.094931,   -1.0715748],
                        [-0.13052036, -1.1766592,   1.0056585,   2.0830042,   2.210154,   -2.3517637,
                         -0.98517996,  0.94388855]])'''
#S1 = tf.constant(4)
#S2 = tf.constant(4)
#goal = tf.constant([3, 4])

trainmapfile = 'G:/OneDrive/TEU/Tensor/VIN_TensorFlow-master/data/gridworld_28x28.npz'
map_num = 0
data = np.load(trainmapfile, allow_pickle=True)
orgin_map = data['arr_0'][0][map_num][3:]

mapdata = orgin_map.reshape((-1, 28, 28, 2)).transpose((0, 3, 1, 2))
X_map = mapdata[0][0]
exit_map = mapdata[0][1]
goal = tf.where(exit_map == GOAL)[0]
S1 = tf.constant(14)
S2 = tf.constant(14)

'''testset = dataset.Dataset(filepath=DATAFILE, mode='train', imsize=IMSIZE)
X_batch, S1_batch, S2_batch, y_batch = testset.next_batch(BATCH_SIZE)
S1 = tf.dtypes.cast(S1_batch[0], dtype=tf.int64)
S2 = tf.dtypes.cast(S2_batch[0], dtype=tf.int64)
now_batch = tf.constant(X_batch[0])
X_map = tf.slice(now_batch,[0,0,0],[IMSIZE,IMSIZE,1])
X_map = tf.reshape(X_map,[-1,IMSIZE])
goal_Map = tf.slice(now_batch,[0,0,1],[IMSIZE,IMSIZE,1])
goal_Map = tf.reshape(goal_Map,[-1,IMSIZE])
goal = tf.where(goal_Map == GOAL)[0]'''

'''--------------------创建多agent与多goal--------------------'''
def makeExits(X_map, mapsize, exit_num, goal):
    '''在空地上生成除了自己和终点以外的Agent
    返回值为[agentNum,2]大小的numpy
    X_map = tensor, mapsize = int, exit_num = int, goal = tensor
    '''
    exit_xlist = [3, 22, 5]
    exit_ylist = [24, 24, 3]
    goal = np.array(goal).tolist()
    exitlist = [goal]
    exit_map = np.zeros_like(X_map)
    exit_map[goal[0]][goal[1]] = GOAL
    
    while exit_num-1:
        while True:
            flag = True  # 为了检测新的出口有没有在出口list出现过
            exit_y = exit_xlist[exit_num-1-1]
            exit_x = exit_ylist[exit_num-1-1]

            getway = [exit_x, exit_y]
            inexitlist = exitlist
            if getway in inexitlist:
                flag = False
            if (X_map[exit_y][exit_x] == 0 and flag):
                exit_map[exit_y][exit_x] = GOAL
                exitlist.append(getway)
                exit_num -= 1
                break
    exitlist = np.array(exitlist)
    return exitlist, exit_map

def makeMulti(X_map, S1, S2, goal, agentNum):
    '''在空地上生成除了自己和终点以外的Agent
        返回值为[agentNum,2]大小的numpy'''
    goal = np.array(goal).tolist()
    if np.array(tf.shape(tf.shape(goal))) == 1:
        goal = [goal]
    agentList = []
    S_map = X_map
    mask = [[0 for i in range(IMSIZE)]for j in range(IMSIZE)]
    for i in range(len(goal)):
        mask[goal[i][0]][goal[i][1]] = WALL
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
'''--------------------创建多agent与多goal--------------------'''

'''--------------------VIN部分--------------------'''
def uniteMap(obsmap, exitmap):
    obsmap = np.array(obsmap)
    obsmap = obsmap.reshape(-1)
    exitmap = exitmap.reshape(-1)
    totala = np.stack((obsmap, exitmap), 1)
    total = totala.reshape(-1)
    return total

def callVIN(obs_map, exit_map, agent_x=0, agent_y=0):
    united_map = uniteMap(obs_map, exit_map)
    sent_map = np.insert(united_map, 0, [agent_y, agent_x])
    sent_map = sent_map[tf.newaxis, ...]

    q_map = VIN(sent_map)  # vector是各个方向的值 ，q_map是价值地图

    q_map = tf.reshape(q_map,-1)
    q_map = tf.nn.softmax(q_map)
    q_map = tf.reshape(q_map, [28, 28])
    return q_map
'''--------------------VIN部分--------------------'''


#----------------------多agent运行部分--------------------
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
        if np.array(tf.shape(tf.shape(goal))) == 1:# 单个goal时添加一个维度
            goal = [goal]
        if lastAction == nowAction or checkTimes >= 16 or nowAction in goal:
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
        if np.array(tf.shape(tf.shape(goal))) == 1:
            goal = [goal]
        for i in range(tf.shape(agentSearch)[0]):  # 初始化
            agentNowList.append(
                agentRoad[agentSearch[i][0], agentSearch[i][1]][-1])
            agentNowDic[agentSearch[i][0], agentSearch[i][1]
                        ] = agentRoad[agentSearch[i][0], agentSearch[i][1]][-1]
            distance = self.imSIZE*4
            for gi in range(len(goal)):# 获取最小的于goal的距离
                temp = self.getDistance(agentSearch[i], goal[gi])
                if temp < distance:
                    distance = temp
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
                        print('road > 3,tempRoad:',tempRoad)
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
        if np.array(tf.shape(tf.shape(goal))) == 1:# 单个goal时添加一个维度
            goal = [goal]
        agentSearch = []
        for i in range(len(agentList)):
            if agentRoad[agentList[i][0], agentList[i][1]][-1] not in goal:
                agentSearch.append(agentList[i])
        return agentSearch

    def runMulti(self, singleMap, valueMap, agentList, goal):
        '''执行多Agent寻路，返回dict[agent原坐标]:[[路径]]  和  dict[agent原坐标]:[步数]
        singleMap = tensor, valueMap = tensor, agentList = tensor, goal = tensor
        singlemap:单独一张的障碍物map
        valueMap：该障碍物的value map
        agentList:就···这个
        goal：goallist'''
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
#----------------------多agent运行部分--------------------

goalList,goalMap = makeExits(X_map,IMSIZE,GOALNUMBER,goal)
agentList = makeMulti(X_map, S1, S2, goalList, 20)
valueMap = callVIN(X_map,goalMap,S1,S2)
print('X_map:',X_map)
print('goalMap:',goalMap)
print('goalList:',goalList)
print('valueMap:',valueMap)
multi = multiAgent(imSIZE=IMSIZE,batchSize=1)
agentStep,agentRoad = multi.runMulti(X_map,valueMap,agentList,goalList)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!OVER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print(agentStep)
print(agentRoad)

