import numpy as np
from spatial_chem_sim import SpacialDiffEquation, DIFFUSE_CONVOLVE
from animations import grid_animation
from dataclasses import dataclass
import os


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


@dataclass(frozen=True)
class ChemOscTest:
    name: str
    init_p: np.ndarray
    init_q: np.ndarray


def horizontal_gradient(width, height, start, end):
    return np.linspace(np.repeat(start, height), np.repeat(end, height), num=width, dtype=float)


def main():
    model = ChemicalClock(
        width=30,
        height=30,
        ds=0.1,
        diffusion_rate=0.001,
        diffusion_kernel=DIFFUSE_CONVOLVE,
        omega=0,
        nu=0
    )
    t_end = 60
    dt = 0.01
    video_frame_rate = 30
    video_t_per_second = 4
    output_folder = "output"
    output_format = ".mp4"
    output_particles = [1, 3]

    x0 = np.zeros((model.width, model.height), dtype=float)
    y0 = np.zeros((model.width, model.height), dtype=float)

    tests = [
        ChemOscTest(name="test_1",
                    init_p=1*np.ones((model.width, model.height), dtype=float),
                    init_q=5*np.ones((model.width, model.height), dtype=float)),
        ChemOscTest(name="test_2",
                    init_p=1*np.ones((model.width, model.height), dtype=float),
                    init_q=horizontal_gradient(model.width, model.height, 0, 10)),
        ChemOscTest(name="test_3",
                    init_p=1*np.ones((model.width, model.height), dtype=float),
                    init_q=5*np.ones((model.width, model.height), dtype=float)
                    + 0.1*np.random.rand(model.width, model.height))
    ]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for test in tests:
        print(f"Running {test.name}...")
        solution = model.solve((test.init_p, test.init_q, x0, y0), t_end, dt)
        for i in output_particles:
            print(f"Animating particle {i}...")
            grid_animation(solution[:, i, :, :], model.ds, model.width, model.height,
                           dt, t_end, video_frame_rate, video_t_per_second,
                           os.path.join(output_folder, f"{test.name}_particle_{i}{output_format}"))


if __name__ == '__main__':
    main()
