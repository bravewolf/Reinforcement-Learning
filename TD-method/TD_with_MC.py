import numpy as np
import random
import matplotlib.pyplot as plt

#init value_s
value_s = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
trueValue = [0, 1/6.0, 2/6.0, 3/6.0, 4/6.0, 5/6.0, 1]
print "a", trueValue
MOVELELT = -1
MOVERIGHT = 1
gamma = 1 #discount factor

def TDprediction(value, alpha):
	state = 3
	while True:
		stateOld = state
		if random.random() < 0.5:
			state += MOVELELT
		else:
			state += MOVERIGHT
		reward = 0
		if state == 6:
			reward = 1		
		value[stateOld] += alpha * (reward + gamma*value[state] - value[stateOld])
		if state == 6:
			break
		if state == 0:
			break

def monteCarloEvaluation(value, alpha):
	state = 3
	trajectory = [3]
	while True:
		if random.random() < 0.5:
			state += MOVELELT
		else:
			state += MOVERIGHT
		reward = 0
		trajectory.append(state)
		if state == 6:
			reward = 1		
			break
		if state == 0:
			break
	for eachState in trajectory:
		value[eachState] += alpha * (reward - value[eachState])

def figure_6_7():
    TDAlpha = [0.15, 0.1, 0.05]
    MCAlpha = [0.01, 0.02, 0.03, 0.04]
    plt.figure(2)
    axisX = np.arange(0, 101)
    for alpha in TDAlpha + MCAlpha:
        totalErrors = np.zeros(101)
        if alpha in TDAlpha:
            method = 'TD'
        else:
            method = 'MC'
        for run in range(0, 100):
            errors = []
            currentStates = np.copy(value_s)
            for i in range(0, 101):
                errors.append(np.sqrt(np.sum(np.power(trueValue - currentStates, 2))/5.0))
                if method == 'TD':
                    TDprediction(currentStates, alpha=alpha)
                else:
                    monteCarloEvaluation(currentStates, alpha=alpha)
            totalErrors += np.asarray(errors)
        totalErrors /= 100
        plt.plot(axisX, totalErrors, label=method + ', alpha=' + str(alpha))
    plt.xlabel('episodes')
    plt.legend()


def figure_6_6(): #Values learned by TD(0) after various numbers of episodes
    episodes = [0, 1, 10, 100]
    value = np.copy(value_s)
    plt.figure(1)
    axisX = np.arange(0, 7)
    plt.plot(axisX, trueValue, label='true values')

    for i in range(0, episodes[-1] + 1):
        if i in episodes:
            plt.plot(axisX, value, label=str(i) + ' episodes')
        TDprediction(value, alpha=0.1)
    
    plt.xlabel('value V_s')
    plt.legend()


figure_6_6()
figure_6_7()
plt.show()
