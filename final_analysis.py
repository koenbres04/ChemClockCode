import numpy as np
from spatial_chem_sim import SpacialDiffEquation, DIFFUSION_KERNEL
from animations import grid_animation
from dataclasses import dataclass
import os
from math import pi, exp


# implement the chemical clock chemical reaction
# note that the variables p,q,x,y here are the dimensionless p_hat, q_hat, x_hat, y_hat from the article
class ChemicalClock(SpacialDiffEquation):
    def __init__(self, width, height, ds, diffusion_kernel, diffusion_rate, omega=0, nu=0):
        super().__init__(width, height, ds, diffusion_kernel, 4, 4*(diffusion_rate,))
        self.omega = omega
        self.nu = nu

    def coordinate_wise_diff_eq(self, t, p, q, x, y):
        dx_dt = p-(1+q)*x+(x**2)*y
        dy_dt = q*x-(x**2)*y
        dp_dt = -self.omega*p
        dq_dt = -self.nu*q*x
        return dp_dt, dq_dt, dx_dt, dy_dt


# a class for containing the information of one test
@dataclass(frozen=True)
class ChemOscTest:
    name: str
    init_p: np.ndarray
    init_q: np.ndarray
    init_x: np.ndarray
    init_y: np.ndarray


# create a (width,height)-shaped array with a horizontal linear gradient from start to end
def horizontal_gradient(width, height, start, end):
    return np.linspace(np.repeat(start, height), np.repeat(end, height), num=width, dtype=float).T


# create a (width,height)-shaped array with values from a gaussian
def gaussian(width, height, ds, mean, sd, total):
    result = np.zeros((width, height), dtype=float)
    mean = np.array(mean, dtype=float)
    for i, j in np.ndindex(width, height):
        result[i, j] = (total*2*pi/sd**2)*exp(-1/2*np.linalg.norm((np.array((i, j), dtype=float)*ds-mean)/sd)**2)
    return result


def main():
    # instantiate a chemical clock model and set a bunch of parameters
    model = ChemicalClock(
        width=30,
        height=30,
        ds=0.1,
        diffusion_rate=0.001,
        diffusion_kernel=DIFFUSION_KERNEL,
        omega=0,
        nu=0
    )
    t_end = 60
    dt = 0.01
    video_frame_rate = 30
    video_t_per_second = 4
    output_folder = "output"
    output_format = ".mp4"
    output_particles = [0, 1, 2, 3]
    channels = [r"$\hat p$", r"$\hat q$", r"$\hat x$", r"$\hat y$"]

    ones = np.ones((model.width, model.height), dtype=float)
    zeros = np.zeros((model.width, model.height), dtype=float)

    # create a list of tests to simulate
    # note that the p,q,x,y's here are the dimensionless versions denoted with a hat in the article
    tests = [
        ChemOscTest(name="test_const_pq",
                    init_p=1*ones,
                    init_q=5*ones,
                    init_x=1*ones,
                    init_y=zeros),
        ChemOscTest(name="test_gradient_q",
                    init_p=1*np.ones((model.width, model.height), dtype=float),
                    init_q=horizontal_gradient(model.width, model.height, 0, 10),
                    init_x=1*ones,
                    init_y=zeros),
        ChemOscTest(name="test_noise_q",
                    init_p=1*np.ones((model.width, model.height), dtype=float),
                    init_q=5*np.ones((model.width, model.height), dtype=float)
                    + 0.1*np.random.rand(model.width, model.height),
                    init_x=1*ones,
                    init_y=zeros),
        ChemOscTest(name="test_gaussian_q",
                    init_p=1*np.ones((model.width, model.height), dtype=float),
                    init_q=gaussian(model.width, model.height, model.ds, (1.5, 1.5), 0.2, 0.1),
                    init_x=1*ones,
                    init_y=zeros),
        ChemOscTest(name="test_gradient_p",
                    init_p=horizontal_gradient(model.width, model.height, 0, 10),
                    init_q=1*np.ones((model.width, model.height), dtype=float),
                    init_x=zeros,
                    init_y=zeros),
        ChemOscTest(name="test_noise_p",
                    init_p=1*np.ones((model.width, model.height), dtype=float)
                    + 0.1*np.random.rand(model.width, model.height),
                    init_q=5*np.ones((model.width, model.height), dtype=float),
                    init_x=1*ones,
                    init_y=zeros),
        ChemOscTest(name="test_gaussian_p",
                    init_p=gaussian(model.width, model.height, model.ds, (1.5, 1.5), 0.2, 0.1),
                    init_q=1*np.ones((model.width, model.height), dtype=float),
                    init_x=zeros,
                    init_y=zeros)
    ]
    # create an output folder if it does not exist
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # run the tests and save animations of the results
    for test in tests:
        print(f"Running {test.name}...")
        solution = model.solve((test.init_p, test.init_q, test.init_x, test.init_y), t_end, dt)
        print(f"Animating {test.name}...")
        solution_subset = [list(np.swapaxes(solution, 0, 1))[i] for i in output_particles]
        grid_animation(solution_subset, model.ds, model.width, model.height,
                       dt, t_end, video_frame_rate, video_t_per_second,
                       os.path.join(output_folder, f"{test.name}{output_format}"),
                       channels=channels)


if __name__ == '__main__':
    main()
