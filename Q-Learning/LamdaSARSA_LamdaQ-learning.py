#) using multi-step bootstrapping, i.e., Q(λ) and SARSA(λ).
import numpy as np
import random
import matplotlib.pyplot as plt
from math import exp
HEIGHT = 4
WIDTH = 12
startState = [3, 0]
goalState = [3, 11]
# EPSILON = 0.1
ALPHA = 0.1

MOVEUP = 0
MOVEDOWN = 1
MOVELEFT = 2
MOVERIGHT = 3
actions = [MOVEUP, MOVEDOWN,MOVELEFT, MOVERIGHT]

lamda = 0.1
actionDestination = []
def planDestinationEachState():
    for i in range(0,HEIGHT):
        actionDestination.append([])
        for j in range(0,WIDTH):
            actionDestination[-1].append([])
            UP      = [max((i-1), 0), j]
            DOWN    = [min((i+1), HEIGHT-1), j]
            LEFT    = [i, max((j-1), 0)]
            RIGHT   = [i, min((j+1), WIDTH-1)]
            actionDestination[i][j].append(UP)
            actionDestination[i][j].append(DOWN)
            actionDestination[i][j].append(LEFT)
            actionDestination[i][j].append(RIGHT)


def eachEpisodeLamdaTabularQ(Q_LamdaTabularQ):
    E = np.zeros((HEIGHT, WIDTH, 4))
    rewards = 0
    state = startState
    time = 0
    while state != goalState:
        EPSILON = exp(-time / 10.0)
        if random.random() < EPSILON:
            action = np.random.choice(actions)
        else:
            action = np.argmax(Q_LamdaTabularQ[state[0], state[1], :])
        reward = -1
        newState = actionDestination[state[0]][state[1]][action]
        if newState != startState and newState != goalState:
            if newState[0] == HEIGHT - 1:
                reward = -100
                newState = startState
        rewards += reward
        # select new action
        if random.random() < EPSILON:
            newAction = np.random.choice(actions)
        else:
            newAction = np.argmax(Q_LamdaTabularQ[newState[0], newState[1], :])
        bestAction = np.argmax(Q_LamdaTabularQ[newState[0], newState[1], :])
        denta = reward + Q_LamdaTabularQ[newState[0], newState[1], bestAction] - Q_LamdaTabularQ[state[0], state[1], action]
        E[state[0]][state[1]][action] += 1.0 #accumulating traces
        for i in range(0, HEIGHT):
            for j in range(0, WIDTH):
                for action in actions:
                    Q_LamdaTabularQ[i, j, action] += ALPHA*denta*E[i,j,action]
                    if newAction == bestAction:
                        E[i, j, action] *= lamda
                    else:
                        E[i, j, action] = 0.0
        state = newState
        time += 1
        # print time
    return rewards


def eachEpisodeLamdaSARSA(Q_LamdaSARSA):
    E = np.zeros((HEIGHT, WIDTH, 4))
    rewards = 0
    state = startState
    time = 0.0
    EPSILON1 = exp(-time / 20.0)
    if random.random() < EPSILON1:
        action = np.random.choice(actions)
    else:
        action = np.argmax(Q_LamdaSARSA[state[0], state[1], :])
    while state != goalState:
        newState = actionDestination[state[0]][state[1]][action]
        reward = -1
        if newState != startState and newState != goalState:
            if newState[0] == HEIGHT - 1:
                reward = -100
                newState = startState
        rewards += reward
        # select new action
        if random.random() < EPSILON1:
            newAction = np.random.choice(actions)
        else:
            newAction = np.argmax(Q_LamdaSARSA[newState[0], newState[1], :])
        denta = reward + Q_LamdaSARSA[newState[0], newState[1], newAction] - Q_LamdaSARSA[state[0], state[1], action]
        E[state[0]][state[1]][action] += 1.0 #accumulating traces
        for i in range(0, HEIGHT):
            for j in range(0, WIDTH):
                for action in actions:
                    Q_LamdaSARSA[i][j][action] += ALPHA * denta * E[i][j][action]
                    E[i][j][action] *= lamda
        state = newState
        action = newAction
        time += 1
        EPSILON1 = exp(-time / 20.0)
        # print time
    return rewards


def plotPolicy(Q):
    policy =[]
    for i in range(0, HEIGHT):
        policy.append([])
        for j in range(0, WIDTH):
            if [i,j] == goalState:
                arrow = u'\u2764'
            else:
                action = np.argmax(Q[i, j, :])
                if action == MOVEUP:
                    arrow = u'\u2B06'
                elif action == MOVEDOWN:
                    arrow = u'\u2B07'
                elif action == MOVELEFT:
                    arrow = u'\u2B05'
                elif action == MOVERIGHT:
                    arrow = u'\u27A1'
            policy[-1].append(arrow)
    for i in range(0, HEIGHT):
        print
        for j in range(0, WIDTH):
            print(policy[i][j]),
    print

def comparing(episodeNumber): #The results are from a single run, but smoothed.

    rewardsLamdaSarsa = np.zeros(episodeNumber)
    rewardsLamdaTabularQ = np.zeros(episodeNumber)

    Q_LamdaTabularQ = np.zeros((HEIGHT, WIDTH, 4))
    Q_LamdaSARSA = np.zeros((HEIGHT, WIDTH, 4))
    for i in range(episodeNumber):
        rewardsLamdaSarsa[i] = max(eachEpisodeLamdaSARSA(Q_LamdaSARSA), -100)
        rewardsLamdaTabularQ[i] = max(eachEpisodeLamdaTabularQ(Q_LamdaTabularQ), -100)

    #smoothing
    smoothedrewardsLamdaSarsa = np.copy(rewardsLamdaSarsa)
    smoothedRewardsLamdaTabularQ = np.copy(rewardsLamdaTabularQ)
    for i in range(10, episodeNumber):
        smoothedrewardsLamdaSarsa[i] = np.mean(rewardsLamdaSarsa[i - 10: i + 1])
        smoothedRewardsLamdaTabularQ[i] = np.mean(rewardsLamdaTabularQ[i - 10: i + 1])

    # display optimal policy
    plotPolicy(Q_LamdaSARSA)
    print('Lamda Sarsa')

    plotPolicy(Q_LamdaTabularQ)
    print('Lamda Q')

    # draw reward curves
    plt.figure(1)
    plt.plot(smoothedrewardsLamdaSarsa, label='Lamda Sarsa')
    plt.plot(smoothedRewardsLamdaTabularQ, label='Lamda Q')
    plt.xlabel('Episodes')
    plt.ylabel('rewards per episode')
    plt.legend()
    plt.show()

planDestinationEachState()
episodeNumber = 500
comparing(episodeNumber)

