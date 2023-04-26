import matplotlib as plt
import numpy as np
from spatial_chem_sim import SpacialDiffEquation, DIFFUSE_CONVOLVE
from io import StringIO
from math import ceil


class BasicChemClock(SpacialDiffEquation):
    def __init__(self, width, height, diffusion_kernel, xy_diffusion_rate, alpha, beta):
        super().__init__(width, height, diffusion_kernel, 2, 2*(xy_diffusion_rate,))
        self.alpha = alpha
        self.beta = beta

    def diff_eq(self, t, x, y):
        return 1-(1+self.alpha)*x+self.beta*(x**2)*y, self.alpha*x-self.beta*(x**2)*y


def temp_print(array, scale=1.):
    builder = StringIO()
    for y in range(array.shape[1]):
        for x in range(array.shape[0]):
            value = array[x, y]
            if value <= scale:
                builder.write("  ")
            elif value <= 2*scale:
                builder.write(". ")
            elif value <= 3*scale:
                builder.write("- ")
            elif value <= 4*scale:
                builder.write("* ")
            else:
                builder.write("# ")
        builder.write("\n")
    print(builder.getvalue())


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

    # dit is de '(aantal tijdstappen) x 2 x width x height' array met resulaten
    solution = model.solve((x0, y0), t_end, dt)

    # temp printen
    for i in range(0, ceil(t_end/dt), 20):
        print(" ")
        print(f"t = {i*dt:2f}")
        temp_print(solution[i, 1, :, :], scale=0.25)


if __name__ == '__main__':
    main()
