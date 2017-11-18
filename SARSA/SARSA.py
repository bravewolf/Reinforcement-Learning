#WindyGridWorld
import numpy as np
import matplotlib.pyplot as plt
import random

HEIGHT = 7
WIDTH = 10
WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

MOVEUP = 0
MOVEDOWN = 1
MOVELEFT = 2
MOVERIGHT = 3
actions = [MOVEUP, MOVEDOWN, MOVELEFT, MOVERIGHT]

reward = -1
EPSILON = 0.1
ALPHA = 0.5

Q = np.zeros((HEIGHT, WIDTH, 4))
startState = [3, 0]
goalState = [3, 7]

actionDestination = []
for i in range(0, HEIGHT):
    actionDestination.append([])
    for j in range(0, WIDTH):
        destination = dict()
        destination[MOVEUP] = [max(i - 1 - WIND[j], 0), j]
        destination[MOVEDOWN] = [max(min(i + 1 - WIND[j], HEIGHT - 1), 0), j]
        destination[MOVELEFT] = [max(i - WIND[j], 0), max(j - 1, 0)]
        destination[MOVERIGHT] = [max(i - WIND[j], 0), min(j + 1, WIDTH - 1)]
        actionDestination[-1].append(destination)

def eachEpisode():
	timeSteps = 0
	currentState = startState

	#epsilonGreedy
	if np.random.binomial(1, EPSILON) == 1:
		currentAction = np.random.choice(actions)
	else:
		currentAction = np.argmax(Q[currentState[0], currentState[1], :])
	
	while currentState != goalState:
		newState = actionDestination[currentState[0]][currentState[1]][currentAction]
		if np.random.binomial(1, EPSILON) == 1:
			newAction = np.random.choice(actions)
		else:
			newAction = np.argmax(Q[newState[0], newState[1], :])
		Q[currentState[0], currentState[1], currentAction] += ALPHA * (reward + Q[newState[0], newState[1], newAction] - Q[currentState[0], currentState[1], currentAction])
		currentState = newState
		currentAction = newAction
		timeSteps += 1
	return timeSteps

Q_SARSA = np.zeros((HEIGHT, WIDTH, 4))
def eachEpisodeSARSA():
    timeStep = 0
    state = startState
    if random.random() < EPSILON:
        action = np.random.choice(actions)
    else:
        action = np.argmax(Q_SARSA[state[0], state[1], :])
    while state != goalState:
        #Execute a
        newState = actionDestination[state[0]][state[1]][action]
        reward = -1
        # select new action
        if random.random() < EPSILON:
            newAction = np.random.choice(actions)
        else:
            newAction = np.argmax(Q_SARSA[state[0], state[1], :])
        # update return function
        Q_SARSA[state[0], state[1], action] += ALPHA * (reward + Q_SARSA[newState[0], newState[1], newAction] - Q_SARSA[state[0], state[1], action])
        state = newState
        action = newAction
        timeStep += 1
    return timeStep

def eachEpisodeQ():
	timeSteps = 0
	currentState = startState

	# epsilonGreedy

	while currentState != goalState:
		if np.random.binomial(1, EPSILON) == 1:
			currentAction = np.random.choice(actions)
		else:
			currentAction = np.argmax(Q[currentState[0], currentState[1], :])

		newState = actionDestination[currentState[0]][currentState[1]][currentAction]
		QmaxNewState = np.max(Q[newState[0], newState[1], :])
		Q[currentState[0], currentState[1], currentAction] += ALPHA * (reward + QmaxNewState - Q[currentState[0], currentState[1], currentAction])
		currentState = newState
		timeSteps += 1
	return timeSteps

episodeLimit = 1000
count = 0
episodes = []
while count < episodeLimit:
    time = eachEpisode()
    episodes.extend([count] * time)
    count += 1

plt.figure()
plt.plot(episodes)
plt.xlabel('Time steps')
plt.ylabel('Episodes')
plt.show()
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
    for i in range(0, WIDTH):
            print(WIND[i]),

plotPolicy(Q)