import numpy as np
from spatial_chem_sim import SpacialDiffEquation, DIFFUSE_CONVOLVE
from animations import grid_animation
from dataclasses import dataclass
import os


class BasicChemClock(SpacialDiffEquation):
    def __init__(self, width, height, ds, diffusion_kernel, xy_diffusion_rate, alpha, beta):
        super().__init__(width, height, ds, diffusion_kernel, 2, 2*(xy_diffusion_rate,))
        self.alpha = alpha
        self.beta = beta

    def coordinate_wise_diff_eq(self, t, x, y):
        return 1-(1+self.alpha)*x+self.beta*(x**2)*y, self.alpha*x-self.beta*(x**2)*y


class ChemicalClock(SpacialDiffEquation):
    def __init__(self, width, height, ds, diffusion_kernel, xy_diffusion_rate, p_diffusion_rate, q_diffusion_rate,
                 alpha_per_q, beta_per_p_squared):
        super().__init__(width, height, ds, diffusion_kernel, 4,
                         (p_diffusion_rate, q_diffusion_rate, xy_diffusion_rate, xy_diffusion_rate))
        self.alpha_per_q = alpha_per_q
        self.beta_per_p_squared = beta_per_p_squared

    def coordinate_wise_diff_eq(self, t, p, q, x, y):
        dp_dt = np.zeros((self.width, self.height), dtype=float)
        dq_dt = np.zeros((self.width, self.height), dtype=float)
        alpha = self.alpha_per_q*q
        beta = self.beta_per_p_squared*p*p
        dx_dt = 1-(1+alpha)*x+beta*(x**2)*y
        dy_dt = alpha*x-beta*(x**2)*y
        return dp_dt, dq_dt, dx_dt, dy_dt


def old_testing():
    model = BasicChemClock(
        width=30,
        height=30,
        ds=0.1,
        diffusion_kernel=DIFFUSE_CONVOLVE,
        xy_diffusion_rate=0.0025,
        alpha=5,
        beta=1
    )
    x0 = np.random.rand(model.width, model.height)*2
    y0 = np.zeros((model.width, model.height), dtype=float)
    y0[15, 15] = 10
    t_end = 40
    dt = 0.01
    video_frame_rate = 30
    video_t_per_second = 1
    output_folder = "output"

    solution = model.solve((x0, y0), t_end, dt)

    grid_animation(solution[:, 1, :, :], model.ds, model.width, model.height, dt, video_frame_rate, video_t_per_second,
                   os.path.join(output_folder, "old_test.mp4"))


@dataclass(frozen=True)
class ChemOscTest:
    file_name: str
    init_p: np.ndarray
    init_q: np.ndarray


def horizontal_gradient(width, height, start, end):
    return np.linspace(np.repeat(start, height), np.repeat(end, height), num=width, dtype=float)


def main():
    model = ChemicalClock(
        width=100,
        height=100,
        ds=0.05,
        p_diffusion_rate=0,
        q_diffusion_rate=0,
        xy_diffusion_rate=0.001,
        diffusion_kernel=DIFFUSE_CONVOLVE,
        alpha_per_q=1,
        beta_per_p_squared=1
    )
    t_end = 60
    dt = 0.01
    video_frame_rate = 30
    video_t_per_second = 4
    output_folder = "output"

    x0 = np.zeros((model.width, model.height), dtype=float)
    y0 = np.zeros((model.width, model.height), dtype=float)

    tests = [
        ChemOscTest(file_name="test_1.mp4",
                    init_p=1*np.ones((model.width, model.height), dtype=float),
                    init_q=5*np.ones((model.width, model.height), dtype=float)),
        ChemOscTest(file_name="test_2.mp4",
                    init_p=1*np.ones((model.width, model.height), dtype=float),
                    init_q=horizontal_gradient(model.width, model.height, 0, 10)),
        ChemOscTest(file_name="test_3.mp4",
                    init_p=1*np.ones((model.width, model.height), dtype=float),
                    init_q=5*np.ones((model.width, model.height), dtype=float)
                    + 0.1*np.random.rand(model.width, model.height))
    ]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for i, test in enumerate(tests):
        print(f"Solving {i+1}-th test...")
        solution = model.solve((test.init_p, test.init_q, x0, y0), t_end, dt)
        print(f"Animating {i+1}-th test...")
        grid_animation(solution[:, 3, :, :], model.ds, model.width, model.height,
                       dt, t_end, video_frame_rate, video_t_per_second,
                       os.path.join(output_folder, test.file_name))


if __name__ == '__main__':
    main()
