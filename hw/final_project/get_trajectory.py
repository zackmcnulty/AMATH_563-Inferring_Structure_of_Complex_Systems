import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import time
import dynamical_systems as ds


def get_trajectories(objects, tspan=(0, 10), num_steps = 1000, tstep = None):
    '''
    Calculates trajectories of the given objects and their specified dynamics throughout the system.
    Normalizes all trajectories so they lie in the grid [0,1] x [0, 1] for convenient plotting

    NOTE: x and y should be the first and second variables in the dynamical system respectively
          

    objects: list of objects to calculate trajectory for through system. These are given as a list
             of tuples in the form (dynamical function, initial condition)
    tspan: length of time to simulate system for
    tstep: time step size for numerical integration
    num_steps: number of steps to use for numerical integration; ignored if tstep is specified
    '''

    if tstep == None:
        t_vals = np.linspace(tspan[0], tspan[1], num_steps)
    else:
        t_vals = np.arange(tspan[0], tspan[1] + tstep, tstep)

    # x or y, object #, time
    all_y_vals = np.zeros((2, len(objects), len(t_vals)))

    for i, (f, y0) in enumerate(objects):
        sol = scipy.integrate.solve_ivp(f, tspan, y0, t_eval = t_vals)

        all_y_vals[0, i, :] = sol.y[0]
        all_y_vals[1, i, :] = sol.y[1]

    '''
    # Normalize each objects trajectory separately
    # Normalize the behavior of objects in the system to remain in window [0,1] x [0,1] =============
    for i in range(len(objects)):
        xmax = np.amax(all_y_vals[0, i, :])
        xmin = np.amin(all_y_vals[0, i, :])
        ymax = np.amax(all_y_vals[1, i, :])
        ymin = np.amin(all_y_vals[1, i, :])

        if xmax != xmin:
            all_y_vals[0, i, :] = (all_y_vals[0, i ,:] - xmin) / (xmax - xmin)
        else:
            all_y_vals[0, i, :] = 0.5

        if ymax != ymin:
            all_y_vals[1, i, :] = (all_y_vals[1,i,:] - ymin) / (ymax - ymin)
        else:
            all_y_vals[1, i, :] = 0.5
    '''

    # Normalize all objects motion together.
    xmax = np.amax(all_y_vals[0, :, :])
    xmin = np.amin(all_y_vals[0, :, :])
    ymax = np.amax(all_y_vals[1, :, :])
    ymin = np.amin(all_y_vals[1, :, :])

    # TODO: the x and Y values should be normalized together. This avoids distorting the motion in any given direction

    all_y_vals[0, :, :] = (all_y_vals[0, : ,:] - min(xmin, ymin)) / (max(xmax,ymax) - min(xmin, ymin))
    all_y_vals[1, :, :] = (all_y_vals[1, : ,:] - min(xmin, ymin)) / (max(xmax,ymax) - min(xmin, ymin))

    # Normalize x, y separately.
    '''
    if xmax != xmin:
        all_y_vals[0, :, :] = (all_y_vals[0, : ,:] - xmin) / (xmax - xmin)

    else: # in case the motion is fixed to avoid division by zero
        all_y_vals[0, :, :] = 0.5

    if ymax != ymin:
        all_y_vals[1, :, :] = (all_y_vals[1,:,:] - ymin) / (ymax - ymin)
    else:
        all_y_vals[1, :, :] = 0.5
    '''

    # tvals, x values, y values    where both x/y values are in the form  
    # [ object 1 x values
    #   object 2 x values
    return (t_vals, all_y_vals[0, :, :], all_y_vals[1, :, :])


# Plot objects moving throughout system ========================================================
def run_example():
    ''' Plots an example of the output of this code'''
    #objects = [ds.f_horz_spring(initial_condition = [1,0,1]), \
    #        ds.f_vert_spring(mass=5,k=1, initial_condition=[2,0,2]), \
    #        ds.f_horz_spring(initial_condition = [1,0,1])]

    objects = (ds.f_angled_spring(initial_condition = [0,0, 10], theta = np.pi/12 ), )

    (t_vals, x_vals, y_vals) = get_trajectories(objects, tstep = 0.1)

    plt.ion()
    plt.show()
    frame_rate = 60
    for t in range(len(t_vals)):
        for i in range(len(objects)):
            x = x_vals[i, t]
            y = y_vals[i, t]

            plt.plot(x, y, 'o')
            plt.xlim([-0.1,1.1])
            plt.ylim([-0.1,1.1])
            plt.text(0.05, 0.9, "time: {0:.2f} seconds".format(t_vals[t]))
       
        plt.draw()
        plt.pause(1/frame_rate)
        plt.clf()


if __name__ == "__main__":
     run_example()
