import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim

NR_EPISODE = 10000
DISCOUNT = 0.9
# EPSILON = 0.1
CAPACITY_N = 10000
MINI_BATCH_SIZE = 50
NR_MAXSTEP = 200
LEARNING_RATE = 0.001

class CNN(object):
    def __init__(self, actionNumber):
        self._actionNumber = actionNumber

    def actionValue_Q(self, state, reuse):

        with slim.arg_scope([slim.fully_connected],
                            reuse=reuse,
                            activation_fn = tf.nn.tanh,
                            weights_regularizer = slim.l2_regularizer(0.001)):
            # _network = slim.fully_connected(state, 5, scope="layer1")
            # _network = slim.fully_connected(_network, 5, scope="layer2")
            # # _network = slim.fully_connected(_network, 20, scope="layer3")
            # _network = slim.fully_connected(_network, self._actionNumber, activation_fn=None, scope = "actionValue")
            # return _network
            net = slim.fully_connected(state, 48, scope='fc1')
            net = slim.fully_connected(net, 32, scope='fc2')
            net = slim.fully_connected(net, 32, scope='fc3')
            # Bottleneck layer.
            net = slim.fully_connected(net, 24, scope='fc4')
            net = slim.fully_connected(net, 48, scope='fc5')
            net = slim.fully_connected(net, self._actionNumber, activation_fn=None, scope='fc_logits')
            return net


class replayMemory_D(object):
    def __init__(self):
        self._D = [None] * CAPACITY_N

    def update(self, state, action, reward, nextState, done): #Queue structure
        self._D.pop(0) #REMOVE FIRST ELEMENT
        self._D.append([state, action, reward, nextState, done]) #ADD TO END OF LIST


    def miniBatch(self, miniBatchSize):
        self._miniBatch = random.sample(self._D, miniBatchSize) #choose randomly from memory
        # self._miniBatch
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
        self._model2 = CNN(actionNumber)
        #variable to compute loss
        self._states = tf.placeholder(tf.float32, shape=[None, self._stateDimension])
        self._actions = tf.placeholder(tf.float32, shape=[None, self._actionNumber])
        self._currentState_CNNoutput = self._model.actionValue_Q(self._states, reuse=False)
        self._nextState_CNNoutput = self._model2.actionValue_Q(self._states, reuse=True)
        #tensor flow session
        self._session = tf.Session()
        self._session.run(tf.global_variables_initializer())


    def sampleMiniBatch(self):
        self._miniBatch = self._replayMemory_D.miniBatch(self._miniBatchSize)
        #allocate memory
        _states = np.zeros([MINI_BATCH_SIZE, self._stateDimension])
        _actions = np.zeros([MINI_BATCH_SIZE, self._actionNumber])
        _rewards = np.zeros([MINI_BATCH_SIZE])
        _nextStates = np.zeros([MINI_BATCH_SIZE, self._stateDimension])
        _dones = []

        #copy from list to numpy array
        for i in range(MINI_BATCH_SIZE): #order in list: 0-state, 1-action, 2-reward, 3-nextState, 4-done
            _states[i] = self._miniBatch[i][0]
            _actions[i][self._miniBatch[i][1]] = 1 #one-hot vector
            _rewards[i] = self._miniBatch[i][2]
            _nextStates[i] = self._miniBatch[i][3]
            _dones.append(self._miniBatch[i][4])

        return _states, _actions, _rewards, _nextStates, _dones


    def computeLoss(self, states, actions, targetQ):

        self._targetQ = tf.placeholder(tf.float32, shape=[None, ])
        self._currentQ = tf.reduce_sum(tf.multiply(self._currentState_CNNoutput, self._actions), reduction_indices=1)

        #loss function
        self._loss = tf.reduce_mean(tf.squared_difference(self._targetQ, self._currentQ))

        #optimizer
        self._optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(self._loss)
        # self._optimizer = tf.train.AdagradOptimizer(learning_rate=LEARNING_RATE).minimize(self._loss)

        ops = [self._loss, self._optimizer]
        l, _ = self._session.run(ops, feed_dict={
            self._states: states,
            self._actions: actions,
            self._targetQ: targetQ})


    def takeAction(self, state, EPSILON):
        if np.random.uniform < EPSILON:
            return self._env.action_space.sample() #RANDOM ACTION
        else:
            _value_Q = self._session.run(self._currentState_CNNoutput,feed_dict={self._states: np.array([state])})
            return np.argmax(_value_Q) #choose  action maximizing Q


    def executeAction(self, action):
        _observation, _reward, _done, _info = self._env.step(action)
        return _observation, _reward, _done


    def storeTransition(self, state, action, reward, nextState, done):
        self._replayMemory_D.update(state, action, reward, nextState, done)


    def getCNNoutput(self, state):
        #return actionValueQ wrt a state
        _value_Q = self._session.run(self._nextState_CNNoutput,feed_dict={self._states: np.array([state])})
        return _value_Q


def dqn(): #Deep Q-learning with Experience Replay
    nr_collectedData = 0 #counting how many data was collected
    env = gym.make('CartPole-v0') #Cartpole environment
    # make new agent, init memory D and action-value Q
    agent = agent_Enviroment(env, env.action_space.n, env.observation_space.shape[0], MINI_BATCH_SIZE)
    EPSILON = 1.0 #take action randomly in the beginning to collect data

    for episode in range(NR_EPISODE):
        state = env.reset() #init first state s1
        # print "episode:", episode
        for step in range(NR_MAXSTEP):
            action = agent.takeAction(state,EPSILON) #greedy epsilon
            # print action
            nextState , reward, done = agent.executeAction(action) #execute action, observe reward and new state
            agent.storeTransition(state, action, reward, nextState, done) #store transition in D
            nr_collectedData += 1
            state = nextState #replace current state by observed state

            if nr_collectedData >= CAPACITY_N: #make sure to collect enough data for D
                EPSILON = 0.1
                #if enough data, sample random miniBatch of transitions from D
                states, actions, rewards, nextStates, dones = agent.sampleMiniBatch()
                targetQ = np.zeros([MINI_BATCH_SIZE]) #as y_j in the DQN paper
                for i in range(MINI_BATCH_SIZE):
                    if dones[i]: #if nextState[i] is terminal
                        targetQ[i] = rewards[i]
                    else: #for non-terminal
                        targetQ[i] = rewards[i] + DISCOUNT * np.max(agent.getCNNoutput(nextStates[i]))
                #perform gradient descent step to loss function
                # for i in range(MINI_BATCH_SIZE):
                agent.computeLoss(states, actions, targetQ)
            else:
                print "collected: ", nr_collectedData

            if done:
                print "episode:", episode, " - step: ", step
                break


if __name__ == '__main__':
    dqn()

