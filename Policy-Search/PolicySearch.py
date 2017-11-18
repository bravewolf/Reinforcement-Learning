import logging
import math
from math import pi,exp
import numpy as np
from numpy.linalg import inv,norm
from numpy import dot
import matplotlib.pyplot as plt


#Environment

class CartPoleEnv(object):

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        b = 12 * 2 * math.pi / 360
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.state = None
        self.steps_beyond_done = None

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        # force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (action + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        gaussianNoise1 = np.random.normal(0, 0.01, size = None)
        x  = x + self.tau * x_dot + gaussianNoise1
        x_dot = x_dot + self.tau * xacc + gaussianNoise1

        gaussianNoise2 = np.random.normal(0, 0.0001, size = None)
        theta = theta + self.tau * theta_dot + gaussianNoise2
        theta_dot = theta_dot + self.tau * thetaacc + gaussianNoise2

        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = -1.0

        # return np.array(self.state), reward, done, {}

        return np.array(self.state), reward

    def reset(self):
        self.state = (0.0, 0.0 ,0.0 ,0.0)
        self.steps_beyond_done = None
        # return np.array(self.state)

        return np.array(self.state)



#functions
def gaussianController(state, omega):
    state=np.reshape(state,(4,1))
    mean = np.dot(omega.T, state)
    nextAction = np.random.normal(mean, 0.001, size=None)
    # count = 0
    # while abs(nextAction) > 10.0:
    #     nextAction = np.random.normal(mean, 0.001, size = None)
    #     count += 1
    #     if count >10000:
    #         print"error"
    # if nextAction > 10.0:
    #     nextAction = 10.0
    # if nextAction < -10.0:
    #     nextAction = -10.0
    if nextAction > 1.0:
        nextAction = 1.0
    if nextAction < -1.0:
        nextAction = -1.0

    return nextAction

def computeObjectiveFunction(omega,episodeNumber,stepNumber):
    J = []
    agent = CartPoleEnv()
    for i in range(episodeNumber):
        state = agent.reset()
        totalReward = 0

        for j in range(stepNumber):
            action = gaussianController(state, omega)
            state, reward = agent.step(action)
            if reward == 1:
                totalReward += reward
            else:
                break
        J.append(totalReward)
    J_mean = np.mean(J)
    return J_mean

def collectData(omega):

    dentaOmega_List = []
    dentaJ_List = []
    J = computeObjectiveFunction(omega,episodeNumber,stepNumber)
    for i in range(numberOfSample):
        dentaOmega = np.random.uniform(-1, 1, size=4)
        omegaNew = omega + dentaOmega
        dentaJ = computeObjectiveFunction(omegaNew,1,stepNumber) - J
        dentaOmega_List.append(dentaOmega.T)
        dentaJ_List.append(dentaJ)
        # print "Collect data Sample:", i
    return np.asarray(dentaOmega_List) , np.asarray(dentaJ_List)

def FD_Gradient(omega):
    dentaOmega, dentaJ = collectData(omega)
    FD_Gradient = dot(dot(inv(dot( dentaOmega.T, dentaOmega )),dentaOmega.T), dentaJ)

    return FD_Gradient

def updatePolicy():
    # checkingCondition = 10**-5
    alpha = 0.5
    omega = np.random.uniform(-1, 1, size=4)
    # omega = np.array([0.0, 0.0,0.0,0.0])
    y=[]
    for i in range(200):
        J = computeObjectiveFunction(omega,episodeNumber,stepNumber)
        FD_Grad = FD_Gradient(omega)
        norm = np.linalg.norm(FD_Grad)
        # denta_omega, alpha = wolfeCondition(alpha, omega, FD_Grad, norm)
        # denta_omega = alpha * FD_Grad / norm
        # omega += denta_omega
        if norm != 0.0:
            omega += alpha * FD_Grad / norm
        else:
            omega += alpha*FD_Grad
        if i%20 == 0:
            print i, J, omega
        y.append(J)
        # print i,J,omega
    plt.plot(y,'r', label="Normal,part a")
    plt.legend()
    return omega

def updatePolicyPartB():
    # checkingCondition = 10**-5

    omega = np.random.uniform(-1, 1, size=4)
    # omega = np.array([0.0, 0.0,0.0,0.0])
    y=[]
    for i in range(200):
        alpha = 10.0 * exp(-i / 10.0)
        J = computeObjectiveFunction(omega,episodeNumber,stepNumber)
        FD_Grad = FD_Gradient(omega)
        norm = np.linalg.norm(FD_Grad)
        # denta_omega, alpha = wolfeCondition(alpha, omega, FD_Grad, norm)
        # denta_omega = alpha * FD_Grad / norm
        # omega += denta_omega
        if norm != 0.0:
            omega += alpha * FD_Grad / norm
        else:
            omega += alpha*FD_Grad
        if i%20 == 0:
            print i, J, omega
        y.append(J)
        # print i,J,omega
    plt.plot(y,'y', label="alpha = 10.0*exp(-i/10.0)")
    plt.legend()
    return omega

def updatePolicyWolfeCondition():
    alpha = 1.0
    omega = np.random.uniform(-1, 1, size=4)
    y=[]
    for i in range(200):
        J = computeObjectiveFunction(omega,episodeNumber,stepNumber)
        FD_Grad = FD_Gradient(omega)
        norm = np.linalg.norm(FD_Grad)

        if norm != 0.0:
            dentaGrad = FD_Grad / norm
        else:
            dentaGrad = FD_Grad
        while   computeObjectiveFunction(omega + alpha* dentaGrad,episodeNumber,stepNumber)  < J + 0.01* dot(FD_Grad.T, alpha*dentaGrad):
            # print alpha
            alpha *= 0.5
        omega += alpha*dentaGrad
        alpha *= 1.2
        if i%20 == 0:
            print i, J, omega
        y.append(J)
        # print i,J,omega
    plt.plot(y, 'g', label="Wolfe Condition")
    plt.legend()

    return omega

def updatePolicyRprob():
    alpha = 0.5
    omega = np.random.uniform(-1, 1, size=4)
    y=[]
    FD_Grad_Pre = np.zeros([4,1])
    J = computeObjectiveFunction(omega, episodeNumber, stepNumber)
    y.append(J)
    for i in range(1,300):

        FD_Grad = FD_Gradient(omega)
        checkingCondition = dot(FD_Grad_Pre.T, FD_Grad)
        if checkingCondition > 0:
            alpha *= 1.2
            if alpha > 5.0:
                alpha = 5.0
            if alpha < 0.01:
                alpha = 0.01
            FD_Grad_Pre = FD_Grad
        elif checkingCondition <0:
            alpha *= 0.5
            if alpha > 5.0:
                alpha = 5.0
            if alpha < 0.01:
                alpha = 0.01
            FD_Grad_Pre = np.zeros([4,1])
        else:
            FD_Grad_Pre = FD_Grad

        omega += alpha*np.sign(FD_Grad)


        J = computeObjectiveFunction(omega, episodeNumber, stepNumber)
        # alpha *= 1.2
        if i%20 == 0:
            print i, J, omega
        y.append(J)
        # print i,J,omega
    plt.plot(y, 'm', label="Rprob")
    plt.legend()

    return omega
def updatePolicyRprob_eachDimension():
    alpha = np.ones([4,1])*0.5
    omega = np.random.uniform(-1, 1, size=4)
    y=[]
    FD_Grad_Pre = np.zeros([4,1])
    J = computeObjectiveFunction(omega, episodeNumber, stepNumber)
    y.append(J)
    for i in range(1,200):
        FD_Grad = FD_Gradient(omega)
        for k in range(4):
            checkingCondition = FD_Grad_Pre[k]* FD_Grad[k]
            if checkingCondition > 0:
                alpha[k] *= 1.2
                if alpha[k] > 5.0:
                    alpha[k] = 5.0
                if alpha[k] < 0.01:
                    alpha[k] = 0.01
                FD_Grad_Pre[k] = FD_Grad[k]
            elif checkingCondition <0:
                alpha[k] *= 0.5
                if alpha[k] > 5.0:
                    alpha[k] = 5.0
                if alpha[k] < 0.01:
                    alpha[k] = 0.01
                FD_Grad_Pre[k] = 0
            else:
                FD_Grad_Pre[k] = FD_Grad[k]

            omega[k] += alpha[k]*np.sign(FD_Grad[k])


        J = computeObjectiveFunction(omega, episodeNumber, stepNumber)
        # alpha *= 1.2
        if i%20 == 0:
            print i, J, omega
        y.append(J)
        # print i,J,omega
    plt.plot(y, 'c', label="Rprob_for each dimension of Omega")
    plt.legend()

    return omega
def plot():

    y = []
    for x in range(200):
        a = x / (0.05 + 0.05*abs(x) )
        y.append(a)
        print a
    plt.plot(y,'g',label = "Wolfe")
    plt.legend()
    # plt.legend(wolfePlot, "Wolfe condition")
# plot()


episodeNumber = 50
stepNumber = 1000
numberOfSample = 50

normal = updatePolicy()
# wolfe = updatePolicyWolfeCondition()
# Rprob = updatePolicyRprob()
# b = updatePolicyPartB()
plt.show()


debug  = 1
#alpha = 10.0*exp(-i/10.0)