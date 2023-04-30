import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def grid_animation(values: list, ds: float, width: int, height: int, dt: float, t_end,
                   video_frame_rate: int, video_t_per_second, filename: str, channels: list):
    """
    Visualizes the two-dimensional solution of a differential equation
    :param values: list of arrays with solutions of each particle where time is on the first axis
    :param ds: timestep in space
    :param width: the amount of steps in the x-axis
    :param height: the amount of steps in the y-axis
    :param dt: timestep in the simulation
    :param t_end: total time in simulation
    :param video_frame_rate: FPS of returned mp4
    :param video_t_per_second: determines how many simulation seconds pass in each real time second
    :param filename: name of output file
    :param channels: list with names of the particles in order of the value list
    :return: saves a mp4 video animation of the given solutions
    """
    num2shape = {"1": 1, "2": (2, 1), "3": (3, 1), "4": (2, 2)}
    num_particles = len(values)
    if num_particles == 1:
        fig, ax = plt.subplots(figsize=(10, 10))
        axs = [ax]
    else:
        fig_shape = num2shape[str(num_particles)]
        fig, axs = plt.subplots(fig_shape[0], fig_shape[1], figsize=(10, 10))
        axs = axs.flatten()
    tot_frames = round(values[0].shape[0]*dt/video_t_per_second*video_frame_rate)
    timestamps = np.linspace(0, t_end, tot_frames)

    def anim_func(frame):
        for i, dens, time_i in zip(range(num_particles), densities, time_texts):
            dens.set_array(values[i][round(frame/video_frame_rate*video_t_per_second/dt), :, :].ravel())
            time_i.set_text(f"t={round(timestamps[frame], 2)} s")
        return *densities, *time_texts

    u = np.arange(0, width*ds, ds)
    v = np.arange(0, height*ds, ds)
    densities = [axs[i].pcolormesh(u, v, values[i][0, :, :], cmap="viridis", shading="auto",
                                vmin=np.min(values[i]),
                                vmax=np.max(values[i])) for i in range(num_particles)]
    time_texts = [ax.text(width*ds*0.01, height*ds*0.90, "t=0", fontsize=10) for ax in axs]
    for i, density in enumerate(densities):
        fig.colorbar(density, ax=axs[i])
        axs[i].set_title(f"partice {channels[i]} over time")

    anim = animation.FuncAnimation(fig, anim_func,
                                   frames=tot_frames,
                                   blit=True)
    anim.save(filename, writer='ffmpeg', fps=video_frame_rate)
