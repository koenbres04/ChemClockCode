import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def grid_animation(values, ds, width, height, dt, video_frame_rate, video_t_per_second, filename):

    def anim_func(frame):
        density.set_array(values[round(frame/video_frame_rate*video_t_per_second/dt), :, :].ravel())
        return density,

    tot_frames = round(values.shape[0]*dt/video_t_per_second*video_frame_rate)

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
