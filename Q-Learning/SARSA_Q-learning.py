import numpy as np
import random
import matplotlib.pyplot as plt

HEIGHT = 4
WIDTH = 12
startState = [3, 0]
goalState = [3, 11]
EPSILON = 0.1
ALPHA = 0.1

MOVEUP = 0
MOVEDOWN = 1
MOVELEFT = 2
MOVERIGHT = 3
actions = [MOVEUP, MOVEDOWN,MOVELEFT, MOVERIGHT]


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


def eachEpisodeQLearning(Q_Qlearning):
    rewards = 0
    state = startState
    while state != goalState:
        if random.random() < EPSILON:
            action = np.random.choice(actions)
        else:
            action = np.argmax(Q_Qlearning[state[0], state[1], :])
        reward = -1
        newState = actionDestination[state[0]][state[1]][action]
        if newState != startState and newState != goalState:
            if newState[0] == HEIGHT-1:
                reward = -100
                newState = startState
        rewards += reward
        QmaxNewState = np.max(Q_Qlearning[newState[0], newState[1], :])
        Q_Qlearning[state[0], state[1], action] += ALPHA * ( reward + QmaxNewState - Q_Qlearning[state[0], state[1], action])
        state = newState
    return rewards


def eachEpisodeSARSA(Q_SARSA):
    rewards = 0
    state = startState
    if random.random() < EPSILON:
        action = np.random.choice(actions)
    else:
        action = np.argmax(Q_SARSA[state[0], state[1], :])
    while state != goalState:
        #Execute a
        newState = actionDestination[state[0]][state[1]][action]
        reward = -1
        if newState != startState and newState != goalState:
            if newState[0] == HEIGHT - 1:
                reward = -100
                newState = startState
        rewards += reward
        # select new action
        if random.random() < EPSILON:
            newAction = np.random.choice(actions)
        else:
            newAction = np.argmax(Q_SARSA[newState[0], newState[1], :])
        # update return function
        Q_SARSA[state[0], state[1], action] += ALPHA * (reward + Q_SARSA[newState[0], newState[1], newAction] - Q_SARSA[state[0], state[1], action])
        state = newState
        action = newAction

        # if state[0] == HEIGHT-1:
        #     state = startState
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

    rewardsSarsa = np.zeros(episodeNumber)
    rewardsQLearning = np.zeros(episodeNumber)

    Q_Qlearning = np.zeros((HEIGHT, WIDTH, 4))
    Q_SARSA = np.zeros((HEIGHT, WIDTH, 4))
    for i in range(episodeNumber):
        rewardsSarsa[i] = max(eachEpisodeSARSA(Q_SARSA), -100)
        rewardsQLearning[i] = max(eachEpisodeQLearning(Q_Qlearning), -100)

    #smoothing
    smoothedRewardsSarsa = np.copy(rewardsSarsa)
    smoothedRewardsQLearning = np.copy(rewardsQLearning)
    for i in range(10, episodeNumber):
        smoothedRewardsSarsa[i] = np.mean(rewardsSarsa[i - 10: i + 1])
        smoothedRewardsQLearning[i] = np.mean(rewardsQLearning[i - 10: i + 1])

    # display optimal policy
    plotPolicy(Q_SARSA)
    print('Sarsa')

    plotPolicy(Q_Qlearning)
    print('Q-Learning')

    # draw reward curves
    plt.figure(1)
    plt.plot(smoothedRewardsSarsa, label='Sarsa')
    plt.plot(smoothedRewardsQLearning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('rewards per episode')
    plt.legend()
    plt.show()

planDestinationEachState()
episodeNumber = 500
comparing(episodeNumber)

