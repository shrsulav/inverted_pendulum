from InvertedPendulum import InvertedPendulum
import numpy as np
import matplotlib.pyplot as plt
from Animator import Animation

class InvertedPendulumSim:
    def __init__(self, beadMass=0.1, cartMass=1.0, rodLength=0.2, mu=10.0, timestep=0.01, duration=5, initialRodAngle=45):
        self.T = duration
        self.timestep = timestep
        self.l = rodLength
        #instantiate an inverted pendulum
        self.system = InvertedPendulum(beadMass, cartMass, rodLength, mu, timestep)
        
        #set the initial angle of the rod
        self.system.setInitialRodAngle(initialRodAngle)
        self.n = int(duration/timestep)
        self.state = np.zeros((4,self.n))
        self.state[0:4,:0] = self.system.getCurrentState()

    def simulate(self):
        self.resetSimulation
        for i in range(self.n-1):
            currentState = self.system.step()
            self.state[0,i+1] = currentState[0,0]
            self.state[1,i+1] = currentState[1,0]
            self.state[2,i+1] = currentState[2,0]
            self.state[3,i+1] = currentState[3,0]

    def plotRodAngle(self, speed=4):
        rodAngle = np.zeros((1,self.n))
        rodAngle = np.rad2deg(self.state[1,:])
        plt.axis([0,self.n, -360, 0])
        for i in range(self.n-1):
            plt.scatter(i,rodAngle[i])
            plt.pause(self.timestep/speed)

    def resetSimulation(self):
        self.system.resetState()

    def setDuration(self, duration):
        self.T = duration
    
    def getDuration(self, duration):
        return self.T

    def getCartPosition(self):
        return self.state[0,:]

    def getRodAngle(self):
        return self.state[1,:]

    def getCartVelocity(self):
        return self.state[2,:]
    
    def getRodVelocity(self):
        return self.state[3,:]
    
    def animate(self):
        anm = Animation(self.l, self.timestep, self.n, self.T)
        anm.generateAnimation(self.state, "anim.gif")

