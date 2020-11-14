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
