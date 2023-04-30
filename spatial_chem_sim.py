import abc
import scipy.integrate
import scipy.signal
import numpy as np
from math import ceil


DIFFUSION_KERNEL = np.array([
    [0., 1., 0.],
    [1., -4., 1.],
    [0., 1., 0.]
])


class SpacialDiffEquation(abc.ABC):
    """"
    A class for spatial differential equations with diffusion
    """

    def __init__(self, width, height, ds, diffusion_kernel, chemical_count, diffusion_rates):
        self.width = width
        self.height = height
        self.ds = ds
        self.diffusion_kernel = diffusion_kernel
        self.chemical_count = chemical_count
        self.diffusion_rates = diffusion_rates

    @abc.abstractmethod
    def coordinate_wise_diff_eq(self, t, *concentration_arrays):
        """
        The coordinate-wise part of the differential equation.

        :param t: time at this point in time
        :param concentration_arrays: tuple of (width,height)-shaped numpy arrays for each particle type containing the
        concentration of this particle at each cell at this point in time
        :return: a tuple of (width,height)-shaped numpy arrays for each particle type containing the rate of change
        of the concentration of this particle particle at each cell at this point in time
        """
        pass

    def solve(self, initial_concentrations, t_end: float, dt: float):
        """
        Solves the spatial differential equation corresponding to the coordinate wise equation plus diffusion

        :param initial_concentrations: a list of self.chemical_count (width,height)-shaped numpy arrays containing
        the initial concentrations of each particle at each cell
        :param t_end: time to end the simulation
        :param dt: timestep in the simulation
        :return: 4-dimensonal numpy array with dimension 0: time, dimension 1: particle index, dimension 2: x-coordinate
        dimension 3: y-coordinate
        """
        # vector version
        y0 = np.array(initial_concentrations).reshape(self.chemical_count*self.width*self.height)
        # an array of time steps to simulate
        time_steps = np.arange(0, ceil(t_end/dt))*dt

        # the vector differential equation to give to the scipy
        def diff_eq(y, t):
            # unwrap y
            concentrations_array = y.reshape(self.chemical_count, self.width, self.height)
            concentrations = tuple(concentrations_array[i, :, :] for i in range(self.chemical_count))
            # initialise an array for the rates of change of the concentratiosn
            dydt = np.zeros((self.chemical_count, self.width, self.height))
            # add the contribution of the coordinate wise differential equation
            dydt += np.array(self.coordinate_wise_diff_eq(t, *concentrations), dtype=float)
            # add the contribution of the diffusion
            for i in range(self.chemical_count):
                padded = np.pad(concentrations[i], pad_width=(self.diffusion_kernel.shape[0]-1)//2, mode='edge')
                convolved = scipy.signal.fftconvolve(padded, self.diffusion_kernel, mode='valid')
                dydt[i, :, :] += convolved * (self.diffusion_rates[i] / self.ds**2)
            # return dydt but in a vector shape
            return dydt.reshape(self.chemical_count*self.width*self.height)

        # solve the vector differential equation
        solution = scipy.integrate.odeint(diff_eq, y0=y0, t=time_steps)
        # return the results in a nicer shape
        return solution.reshape((len(time_steps), self.chemical_count, self.width, self.height))
