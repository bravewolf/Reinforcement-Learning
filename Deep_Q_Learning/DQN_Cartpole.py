import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt

NR_EPISODE = 1000
DISCOUNT = 0.97
# EPSILON = 0.1
CAPACITY_N = 16384
MIN_NR_COLLECTED = 512
MINI_BATCH_SIZE = 64
NR_MAXSTEP = 500
LEARNING_RATE = 0.001

class CNN(object):
    def __init__(self, actionNumber):
        self._actionNumber = actionNumber

    def actionValue_Q(self, state, reuse):

        with slim.arg_scope([slim.fully_connected],
                            reuse=reuse,
                            activation_fn = tf.nn.relu,
                            weights_regularizer = slim.l2_regularizer(0.0005)):
            _network = slim.fully_connected(state, 5, scope="layer1")
            _network = slim.fully_connected(_network, 5, scope="layer2")
            _network = slim.fully_connected(_network, 20, scope="layer3")
            _network = slim.fully_connected(_network, self._actionNumber, activation_fn=None, scope = "actionValue")
            return _network



class replayMemory_D(object):
    def __init__(self):
        self._D = []

    def update(self, NrCollected, state, action, reward, nextState): #Queue structure
        if NrCollected < CAPACITY_N:
            self._D.append([state, action, reward, nextState])
        else:
            self._D.pop(0) #REMOVE FIRST ELEMENT
            self._D.append([state, action, reward, nextState]) #ADD TO END OF LIST


    def miniBatch(self, miniBatchSize):
        self._miniBatch = random.sample(self._D, miniBatchSize) #choose randomly from memory
        return self._miniBatch


class agent_Enviroment(object):
    def __init__(self, env, actionNumber, stateDimension, miniBatchSize):
        #initialize
        self._env = env
        self._replayMemory_D = replayMemory_D() #init replay memory D
        self._model = CNN(actionNumber) #init action-value function Q
        self._actionNumber = actionNumber
        self._stateDimension = stateDimension
        self._miniBatchSize = miniBatchSize

        #variable to compute loss
        self._states = tf.placeholder(tf.float32, shape=[None, self._stateDimension])
        self._actions = tf.placeholder(tf.float32, shape=[None, self._actionNumber])
        self._rewards = tf.placeholder(tf.float32, shape=[None, ])
        self._nextStates = tf.placeholder(tf.float32, shape=[None, self._stateDimension])

        self._currentState_CNNoutput = self._model.actionValue_Q(self._states, reuse=False)
        self._nextState_CNNoutput = self._model.actionValue_Q(self._nextStates, reuse=True)
        self._currentQ = tf.reduce_sum(tf.multiply(self._currentState_CNNoutput, self._actions), reduction_indices=1)
        self._targetQ = self._rewards + DISCOUNT * tf.reduce_max(self._nextState_CNNoutput, reduction_indices=1)
        #loss function
        self._loss = tf.reduce_mean(tf.squared_difference(self._targetQ, self._currentQ))
        lr = tf.train.exponential_decay(0.1, slim.get_or_create_global_step(), 10000, 0.98)
        self._optimizer = tf.train.AdagradOptimizer(lr).minimize(self._loss)
        # self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(self._loss)
        #tensor flow session
        # self._session = tf.Session()
        self._session = tf.InteractiveSession()
        self._session.run(tf.global_variables_initializer())


    def sampleMiniBatch(self):
        self._miniBatch = self._replayMemory_D.miniBatch(self._miniBatchSize)
        #allocate memory
        _states = np.zeros([MINI_BATCH_SIZE, self._stateDimension])
        _actions = np.zeros([MINI_BATCH_SIZE, self._actionNumber])
        _rewards = np.zeros([MINI_BATCH_SIZE])
        _nextStates = np.zeros([MINI_BATCH_SIZE, self._stateDimension])

        #copy from list to numpy array
        for i in range(MINI_BATCH_SIZE): #order in list: 0-state, 1-action, 2-reward, 3-nextState
            _states[i] = self._miniBatch[i][0]
            _actions[i][self._miniBatch[i][1]] = 1 #one-hot vector
            _rewards[i] = self._miniBatch[i][2]
            _nextStates[i] = self._miniBatch[i][3]

        return _states, _actions, _rewards, _nextStates



    def computeLoss(self, states, actions, rewards, nextStates):

        ops = [self._loss, self._optimizer]
        l, _ = self._session.run(ops, feed_dict={
            self._states: states,
            self._actions: actions,
            self._rewards: rewards,
            self._nextStates: nextStates })


    def takeAction(self, state, EPSILON):
        if np.random.uniform < EPSILON:
            return self._env.action_space.sample() #RANDOM ACTION
        else:
            # _value_Q = self._session.run(self._currentState_CNNoutput,feed_dict={self._states: np.array([state])})
            _value_Q = self._currentState_CNNoutput.eval(feed_dict={self._states: np.array([state])})
            # print _value_Q
            return np.argmax(_value_Q) #choose  action maximizing Q


    def executeAction(self, action):
        _observation, _reward, _done, _info = self._env.step(action)
        if _done:
            _reward = - 500.0 #penalty
        return _observation, _reward, _done


    def storeTransition(self, NrCollected, state, action, reward, nextState):
        self._replayMemory_D.update(NrCollected, state, action, reward, nextState)


def decay_epsilon(x, factor = 0.99):
    eps = factor ** (x / 200)
    return eps


def dqn(): #Deep Q-learning with Experience Replay
    nr_collectedData = 0 #counting how many data was collected
    env = gym.make('CartPole-v0') #Cartpole environment
    # env.seed(3141)
    # make new agent, init memory D and action-value Q
    agent = agent_Enviroment(env, env.action_space.n, env.observation_space.shape[0], MINI_BATCH_SIZE)
    EPSILON = 1.0 #take action randomly in the beginning to collect data
    rewards_list = []
    average_reward = 0.0
    for episode in range(NR_EPISODE):
        state = env.reset() #init first state s1
        for step in range(NR_MAXSTEP):
            env.render()
            action = agent.takeAction(state, EPSILON) #greedy epsilon
            nextState , reward, done = agent.executeAction(action)
            #execute action, observe reward and new state
            if done:
                nextState = np.zeros_like(nextState)
            agent.storeTransition(nr_collectedData, state, action, reward, nextState) #store transition in D
            nr_collectedData += 1
            state = nextState #replace current state by observed state

            if nr_collectedData >= MIN_NR_COLLECTED: #make sure to collect enough data to start training
                EPSILON = decay_epsilon(nr_collectedData - MIN_NR_COLLECTED)
                states, actions, rewards, nextStates = agent.sampleMiniBatch()
                agent.computeLoss(states, actions, rewards, nextStates)

            if done:
                print "episode:", episode, " - step: ", step + 1
                rewards_list.append(step+1)
                average_reward = sum(rewards_list[-100:])/100.0
                print "average reward: ", average_reward
                print "EPSILON", EPSILON
                break
        if average_reward > 195:
            print("CartPole is solved!" )

            break
    plt.plot(rewards_list)
    plt.show()

if __name__ == '__main__':
    dqn()
