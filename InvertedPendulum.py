import numpy as np

class InvertedPendulum:
    def __init__(self,beadMass=0.25, cartMass=2.0, rodLength=0.25, mu=10.0, timestep=0.01):
        # System parameters
        self.m = beadMass                   # mass of bead  [kg]
        self.M = cartMass                   # mass of cart  [kg]
        self.l = rodLength                  # length of rod [m]
        self.g = 9.81                       # gravitational acceleration [m s^-2]
        self.mu = mu                        # friction coefficient [kg s^-1]
        self.timestep = timestep            #simulation time delta

        self.u = 0                          # initial cart force  [kg m s^-2]
        self.currentState = np.zeros((4,1))
        #currentState[0] = position of the cart along horizontal x-axis[m]
        #currentState[1] = rod angle w.r.t the desired angle measured in CW direction [radians]
        #currentState[2] = cart velocity [ m per s]
        #currentState[3] = angular velocity of the rod [radians per s]

        self.sysParameters = np.array([beadMass, cartMass, rodLength, self.g, mu, self.u])

    def setInitialRodAngle(self, rodAngleInDegrees):
        self.resetState()
        self.currentState[1] = np.deg2rad(rodAngleInDegrees)

    def getCurrentState(self):
        return self.currentState
    
    def resetState(self):
        self.currentState = np.zeros((4,1))
    
    def setExternalForce(self, force):
        self.u = force

    def getTimestep(self):
        return self.timestep

    def stateDerivative(self, state):
        """
        @description:
            Given a state vector, this function returns the 
            state vector element's time derivative
        @params:
            state        4x1 array
        @returns:
            dxdt     4x1 array
        """
        m = self.m
        M = self.M
        l = self.l
        g = self.g
        mu = self.mu
        u = self.u

        x0, x1, x2, x3 = state
        dx0dt = x2
        dx1dt = x3
        lambdax1 = 1/(l*(M + m*np.sin(x1)**2))
        
        dx2dt = lambdax1*(m*l*l*x3**2*np.sin(x1) - m*g*l*np.sin(x1)*np.cos(x1)
                            - l*mu*x2  + u*l)
        dx3dt = lambdax1*((m + M)*g*np.sin(x1) - m*l*x3*x3*np.sin(x1)*np.cos(x1)
                            + mu*x2*np.cos(x1) - u*np.cos(x1))
        
        dxdt = np.array([dx0dt, dx1dt, dx2dt, dx3dt])
                               
        return dxdt

    def rk4step(self, state):
        """
        @description:
            Given the current state vector,
            this method returns the state vector after timestep dt
            using 4th order Runge-Kutta method
        @params:
            state         4x1 array
        @returns:
            The state vector, z, after a time, dt.
        """
        dt = self.timestep
        k1 = np.array(self.stateDerivative(state))
        k2 = np.array(self.stateDerivative(state + k1*dt/2))
        k3 = np.array(self.stateDerivative(state + k2*dt/2))
        k4 = np.array(self.stateDerivative(state + k3*dt))
        
        return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)

    def getEnergy(self, state):
        """
        @description:
            Given the the state vector, 
            this method calculates and returns the kinetic
            and potential energy of the system.
        @params:
            state: 4x1 state vector
        @returns:
            T: kinetic energy
            U: potential energy
        """
        x0, x1, x2, x3 = state
        U = self.m * self.g * self.l * np.cos(x1)
        T = 0.5*x2**2*(self.m + self.M) \
            + 0.5*self.m * x3**2 * self.l**2 \
            + self.m*x2*x3*self.l*np.cos(x1)

        return np.array([T, U])

    def step(self):
        """
        @description:
            Computes the next state of the inverted pendulum
            Assigns the computed nextState -> currentState
            Returns the state of the inverted pendulum
        @params: None
        @returns:
            nextState: 4x1 state vector
        """
        noise = np.random.uniform(-0.001, 0.001)
        self.setExternalForce(noise)
        # Compute and insert the next state vector using RK4
        nextState = self.rk4step(self.currentState)
        self.currentState = nextState
        return nextState
