'''

lorenz_net.py

This code will train a feedforward neural network to predict trajectories within the lorenz system.

'''

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pylab as plt
import matplotlib
print(matplotlib.get_backend())
matplotlib.use('TkAgg')
import numpy as np
import scipy.integrate
from mpl_toolkits.mplot3d import Axes3D


dt = 0.01  # time step size
T = 8  # total simulation time per initial condition
t = np.arange(start=0, stop=T, step=dt)

# Set parameters of the lorenz system
b = 8/3
sig = 10
r = 28


def lorenz(t, x):
    return [sig * (x[1] - x[0]),  r * x[0] - x[0]*x[2] - x[1], x[0]*x[1] - b*x[2]]


fig = plt.figure(1)
ax = fig.gca(projection='3d')
for j in range(100):
    x0 = 30*np.random.uniform(low=-0.5, high=0.5, size=3)

    # Pass in initial conditions and relative/absolute tolerance
    y_vals = scipy.integrate.odeint(func=lorenz, y0=x0, t=t, rtol=1e-10, atol=1e-11, tfirst=True)

    ax.plot(xs=y_vals[:, 0], ys=y_vals[:, 1], zs=y_vals[:, 2])
#    Axes3D.plot(xs=x0[0], ys=x0[1], zs=x[2], 'ro')


plt.show(block=True)
