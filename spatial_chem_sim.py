import abc
import scipy.integrate
import scipy.signal
import numpy as np
from math import ceil


DIFFUSE_CONVOLVE = np.array([
    [0., 1., 0.],
    [1., -4., 1.],
    [0., 1., 0.]
])


class SpacialDiffEquation(abc.ABC):
    def __init__(self, width, height, ds, diffusion_kernel, chemical_count, diffusion_rates):
        self.width = width
        self.height = height
        self.ds = ds
        self.diffusion_kernel = diffusion_kernel
        self.chemical_count = chemical_count
        self.diffusion_rates = diffusion_rates

    @abc.abstractmethod
    def coordinate_wise_diff_eq(self, t, *concentration_arrays):
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
        y0 = np.array(initial_concentrations).reshape(self.chemical_count*self.width*self.height)
        time_steps = np.arange(0, ceil(t_end/dt))*dt

        # the differential equation to give to the scipy
        def diff_eq(y, t):
            concentrations_array = y.reshape(self.chemical_count, self.width, self.height)
            concentrations = tuple(concentrations_array[i, :, :] for i in range(self.chemical_count))
            dydt = np.zeros((self.chemical_count, self.width, self.height))
            dydt += np.array(self.coordinate_wise_diff_eq(t, *concentrations), dtype=float)
            for i in range(self.chemical_count):
                padded = np.pad(concentrations[i], pad_width=(self.diffusion_kernel.shape[0]-1)//2, mode='edge')
                convolved = scipy.signal.fftconvolve(padded, self.diffusion_kernel, mode='valid')
                dydt[i, :, :] += convolved * (self.diffusion_rates[i] / self.ds**2)
            return dydt.reshape(self.chemical_count*self.width*self.height)

        solution = scipy.integrate.odeint(diff_eq, y0=y0, t=time_steps)
        return solution.reshape((len(time_steps), self.chemical_count, self.width, self.height))
