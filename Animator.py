import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, Image

class Animation:
    def __init__(self, rodLength, timestep, dataPoints,duration):
        self.l = rodLength
        self.dt = timestep
        self.n = dataPoints
        self.T = duration
    
    def cartX(self,x_1):
        """ Given the value of x_1, i.e. the position
        of the cart, returns the x-coordinates of the
        cart's corners."""
        l = self.l
        return np.array([x_1 - l/8, x_1 + l/8, x_1 + l/8, x_1 - l/8, x_1 - l/8])
    
    def generateAnimation(self, Z, name):
        l = self.l
        figSizeX = 12
        figSizeY = 9
        """ This function generates two animations of the system as a function of time,
        from the two results, 'Z' and 'Z2'. The two animations are plottet underneath each other.
        Arguments:
            Z      4xN array. State vector component-values at each timestep
            Z2     4xN array. State vector component-values at each timestep
            name   string. Name of the animation file.
        """
    
      # Defining an array of the time in seconds at each step
        timeValues = np.arange(0, self.n*self.dt, self.dt)
    
        X = Z[0] + l*np.sin(Z[1])                           # x-coordinates of bead
        Y = l*np.cos(Z[1])                                  # y-coordinates of bead
        cartY = np.array([l/16, l/16, -l/16, -l/16, l/16])  # y-coordinates of cart corners
    
        fig = plt.figure(figsize=(figSizeX, figSizeY), dpi=200)
    
        # Animation/result/pendulum number one
        # Axis limitations
        xMin = np.min(X) - 1.5*l
        xMax = np.max(X) + 1.5*l
        # Adjust the y-limitations such that a circle looks like a circle.
        # i.e. no scaling.
        xDomain = xMax - xMin
        yDomain = xDomain * 0.5*figSizeY/figSizeX
        plt.xlim(xMin, xMax)
        plt.ylim(-0.5*yDomain, 0.5*yDomain)

        # Defining the different elements in the animation
        surface, = plt.plot([xMin, xMax], [-l/16, -l/16], color='black', linewidth=1)    # The surface
        tail, = plt.plot(X[0], Y[0], '--', color="blue")            # Previous position of the pendulum bead
        cart, = plt.plot(self.cartX(Z[0, 0]), cartY, color="red")        # The cart 
        rod, = plt.plot([Z[0, 0], X[0]], [0, Y[0]], color="black")  # The massless pendulum rod of length l
        bead, = plt.plot(X[0], Y[0], 'o', color="black", ms=4)      # The pendulum bead
        text = plt.text(xMax-0.1, 0.5*yDomain - 0.1, r'$t =  %.2f$s'%(timeValues[0]), # Text box with elapsed time
                {'color': 'k', 'fontsize': 10, 'ha': 'center', 'va': 'center',
                'bbox': dict(boxstyle='round', fc='w', ec='k', pad=0.2)})

        plt.xlabel('x, m')
        plt.ylabel('y, m')
        plt.title("Proportional Gain controller")

        plt.subplots_adjust(hspace=0.3)
    
        # Calculate the number of frames
        FPS = 30
        framesNum = int(FPS*self.T)
        dataPointsPerFrame = int(self.n/framesNum)

        # Animation function. This is called sequentially
        def animate(j):
            time = j*dataPointsPerFrame
            # Pendulum 1
            tail.set_xdata(X[:time])
            tail.set_ydata(Y[:time])
            cart.set_xdata(self.cartX(Z[0, time])) 
            cart.set_ydata(cartY)
            rod.set_xdata([Z[0, time], X[time]])
            rod.set_ydata([0, Y[time]])
            bead.set_xdata(X[time])
            bead.set_ydata(Y[time])
            text.set_text(r'$t =  %.2f$s'%(timeValues[time]))

        # Create animation
        anim = animation.FuncAnimation(fig, animate, frames=framesNum)

    # Save animation.
    # If this don't work for you, try using another writer 
    # (ffmpeg, mencoder, imagemagick), or another file extension
    # (.mp4, .gif, .ogg, .ogv, .avi etc.). Make sure that you
    # have the codec and the writer installed on your system.
        anim.save(name, writer='imagemagick', fps=FPS)

        # Close plot
        plt.close(anim._fig)

    # Display the animation
        #with open(name, 'rb') as file:
        #    display(Image(file.read()))