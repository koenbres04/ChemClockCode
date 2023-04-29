import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def grid_animation(values, ds, width, height, dt, t_end, video_frame_rate, video_t_per_second, filename):
    fig, ax = plt.subplots()
    tot_frames = round(values.shape[0]*dt/video_t_per_second*video_frame_rate)
    timestamps = np.linspace(0, t_end, tot_frames)

    def anim_func(frame):
        density.set_array(values[round(frame/video_frame_rate*video_t_per_second/dt), :, :].ravel())
        time_text.set_text(f"t={round(timestamps[frame], 2)} s")
        return density, time_text

    u = np.arange(0, width*ds, ds)
    v = np.arange(0, height*ds, ds)
    density = ax.pcolormesh(u, v, values[0, :, :], cmap="viridis", shading="auto",
                            vmin=np.min(values),
                            vmax=np.max(values))
    time_text = ax.text(width*ds*0.01, height*ds*0.94, "t=0", fontsize=10)
    fig.colorbar(density)

    anim = animation.FuncAnimation(fig, anim_func,
                                   frames=tot_frames,
                                   blit=True)
    anim.save(filename, writer='ffmpeg', fps=video_frame_rate)
