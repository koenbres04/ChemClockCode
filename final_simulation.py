import numpy as np
from spatial_chem_sim import SpacialDiffEquation, DIFFUSION_KERNEL
from dataclasses import dataclass
import os
from math import pi, exp
import time
import json


# implement the chemical clock chemical reaction
# note that the variables p,q,x,y,t here are the dimensionless p_hat, q_hat, x_hat, y_hat from the article
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


# create a (width,height)-shaped array of random noise
def noise(width, height, amplitude, seed):
    np.random.seed(seed)
    return amplitude * np.random.rand(width, height)


# create a (width,height)-shaped array with values from a gaussian
def gaussian(width, height, ds, mean, sd):
    result = np.zeros((width, height), dtype=float)
    mean = np.array(mean, dtype=float)
    for i, j in np.ndindex(width, height):
        result[i, j] = exp(-1/2*np.linalg.norm((np.array((i, j), dtype=float)*ds-mean)/sd)**2)/(2*pi*sd**2)
    return result


def main():
    # instantiate a chemical clock model
    model = ChemicalClock(
        width=30,
        height=30,
        ds=0.1,
        diffusion_rate=0.0025,
        diffusion_kernel=DIFFUSION_KERNEL,
        omega=0,
        nu=0
    )
    # set parameters of the simulation
    t_end = 400
    dt = 0.01
    # set parameters for the output video
    output_folder = "output"

    # parameters for the tests
    # note that the p,q,x,y's here are the dimensionless versions denoted with a hat in the article
    average_p = 1
    average_q = 5

    # create a list of tests to simulate
    ones = np.ones((model.width, model.height), dtype=float)
    zeros = np.zeros((model.width, model.height), dtype=float)
    area = model.width*model.width*model.ds*model.ds
    tests = [
        ChemOscTest(name="homogenous",
                    init_p=average_p*ones,
                    init_q=average_q*ones,
                    init_x=average_p*ones,
                    init_y=zeros),
        ChemOscTest(name="gradient_q",
                    init_p=average_p*ones,
                    init_q=horizontal_gradient(model.width, model.height, 0, 2*average_q),
                    init_x=average_p*ones,
                    init_y=zeros),
        ChemOscTest(name="noise_q",
                    init_p=average_p*ones,
                    init_q=average_q*(ones+noise(model.width, model.height, amplitude=0.1, seed=171267)),
                    init_x=average_p*ones,
                    init_y=zeros),
        ChemOscTest(name="gaussian_q",
                    init_p=average_p*ones,
                    init_q=average_q*area*gaussian(model.width, model.height, model.ds, (1.5, 1.5), 0.5),
                    init_x=average_p*ones,
                    init_y=zeros),
        ChemOscTest(name="gradient_p",
                    init_p=horizontal_gradient(model.width, model.height, 0, 2*average_p),
                    init_q=average_q*ones,
                    init_x=zeros,
                    init_y=zeros),
        ChemOscTest(name="noise_p",
                    init_p=average_p*(ones+noise(model.width, model.height, amplitude=0.1, seed=79565)),
                    init_q=average_q*ones,
                    init_x=average_p*ones,
                    init_y=zeros),
        ChemOscTest(name="gaussian_p",
                    init_p=average_p*area*gaussian(model.width, model.height, model.ds, (1.5, 1.5), 0.5),
                    init_q=average_q*ones,
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
        print(f"Saving {test.name}...")
        test_folder = os.path.join(output_folder, f"{test.name}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.mkdir(test_folder)
        np.save(os.path.join(test_folder, "raw.npy"), solution)
        params_dict = {
            "width": model.width,
            "height": model.height,
            "ds": model.ds,
            "dt": dt,
            "t_end": t_end
        }
        with open(os.path.join(test_folder, "params.json"), "w") as params_file:
            json.dump(params_dict, params_file)


if __name__ == '__main__':
    main()
