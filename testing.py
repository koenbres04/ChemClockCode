import numpy as np
from spatial_chem_sim import SpacialDiffEquation, DIFFUSE_CONVOLVE
from animations import grid_animation


class BasicChemClock(SpacialDiffEquation):
    def __init__(self, width, height, diffusion_kernel, xy_diffusion_rate, alpha, beta):
        super().__init__(width, height, diffusion_kernel, 2, 2*(xy_diffusion_rate,))
        self.alpha = alpha
        self.beta = beta

    def coordinate_wise_diff_eq(self, t, x, y):
        return 1-(1+self.alpha)*x+self.beta*(x**2)*y, self.alpha*x-self.beta*(x**2)*y


def main():
    model = BasicChemClock(
        width=30,
        height=30,
        diffusion_kernel=DIFFUSE_CONVOLVE,
        xy_diffusion_rate=0.025,
        alpha=1,
        beta=1
    )
    x0 = np.random.rand(model.width, model.height)*2
    y0 = np.zeros((model.width, model.height), dtype=float)
    t_end = 10
    dt = 0.01

    solution = model.solve((x0, y0), t_end, dt)

    grid_animation(solution[::10, 1, :, :], 10*dt)


if __name__ == '__main__':
    main()
