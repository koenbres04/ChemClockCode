from io import StringIO
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


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


def grid_animation(values, ds, width, height, t_end, dt, video_frame_rate, video_t_per_second, filename):

    def anim_func(frame):
        density.set_array(values[frame, :, :].ravel())
        return density,

    tot_frames = values.shape[0]

    u = np.arange(0, width*ds, ds)
    v = np.arange(0, height*ds, ds)

    fig, ax = plt.subplots()
    density = ax.pcolormesh(u, v, values[0, :, :], cmap="viridis", shading="auto",
                            vmin=np.min(values),
                            vmax=np.max(values))
    fig.colorbar(density)

    anim = animation.FuncAnimation(fig, anim_func,
                                   frames=tot_frames,
                                   blit=True)
    anim.save(filename, writer='ffmpeg', fps=video_frame_rate)






