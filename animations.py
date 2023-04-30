import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def grid_animation(values: list[np.ndarray], ds: float, width: int, height: int, dt: float, t_end,
                   video_frame_rate: int, video_t_per_second, filename: str, channels: list[str],
                   track_point=None):
    """
    Visualizes the two-dimensional solution of a differential equation

    :param values: list of arrays with solutions of each particle where time is on the first axis
    :param ds: timestep in space
    :param width: the amount of steps in the x-axis
    :param height: the amount of steps in the y-axis
    :param dt: timestep in the simulation
    :param t_end: total time in simulation
    :param video_frame_rate: FPS of returned video
    :param video_t_per_second: determines how many simulation seconds pass in each real time second
    :param filename: name of output file
    :param channels: list with names of the particles in order of the value list
    :param track_point: optional (x, y) coordinates to show over time next to the other plots
    """
    # create a mesh for the space that we simulate over
    u = np.arange(0, width*ds, ds)
    v = np.arange(0, height*ds, ds)

    # used for the shapes of the figures
    if track_point is not None:
        # get data for track_point
        i_u_nearest = (np.abs(u - track_point[0])).argmin()
        i_v_nearest = (np.abs(v - track_point[1])).argmin()
        track_values = [value[:, i_u_nearest, i_v_nearest] for value in values]
        num2shape = {"1": (1, 2, 6, 4), "2": (2, 2, 12, 8), "3": (3, 2, 18, 14), "4": (2, 4, 24, 16)}
        ratio = [1, 2]
    else:
        num2shape = {"1": (1, 1, 5, 4), "2": (2, 1, 10, 4), "3": (3, 1, 15, 4), "4": (2, 2, 10, 8)}
        ratio = [1]
    # amount of particles that are shown
    num_particles = len(values)
    # initialize figure and axis
    fig_shape = num2shape[str(num_particles)]
    fig, axs = plt.subplots(fig_shape[0], fig_shape[1],
                            figsize=(fig_shape[2], fig_shape[3]),
                            width_ratios=ratio*fig_shape[0])
    axs = list(axs.flatten())
    if track_point is not None:
        density_axs = axs[1::2]
        track_axs = axs[::2]
    else:
        density_axs = axs
        track_axs = []
    tot_frames = round(values[0].shape[0]*dt/video_t_per_second*video_frame_rate)
    timestamps = np.linspace(0, t_end, tot_frames)

    # animation function
    def anim_func(frame):
        for i, dens, time_i, time_line in zip(range(num_particles), densities, time_texts, track_lines):
            dens.set_array(values[i][round(frame/video_frame_rate*video_t_per_second/dt), :, :].ravel())
            time_i.set_text(f"t={round(timestamps[frame], 2)} s")
            time_line.set_xdata(np.array([timestamps[frame]]*2))
        return *densities, *time_texts, *track_lines

    # initialize the density and normal plots and add a time text
    densities = [density_axs[i].pcolormesh(u, v, values[i][0, :, :], cmap="viridis", shading="auto",
                                vmin=np.min(values[i]),
                                vmax=np.max(values[i])) for i in range(num_particles)]
    time_texts = [ax.text(width*ds*0.01, height*ds*0.90, "t=0", fontsize=10) for ax in density_axs]
    if track_point is not None:
        track_lines = []
        for i, ax in enumerate(track_axs):
            ax.plot(np.linspace(0, t_end, len(track_values[i])),
                    track_values[i], color="red", linewidth=1.2)
            line, = ax.plot([0]*2, [np.min(track_values[i])-0.05, np.max(track_values[i])+0.05],
                               color="black", linewidth=1.1, linestyle="dotted")
            ax.set(xlabel=r"$\hat{t}$", ylabel=channels[i])
            track_lines.append(line)
        for ax in density_axs:
            ax.scatter(track_point[0], track_point[1], c="red", s=100, zorder=10, alpha=0.9)
    # add some titles and color bars to the side
    for i, density in enumerate(densities):
        fig.colorbar(density, ax=density_axs[i])
        density_axs[i].set_title(f"{channels[i]} over time")

    # run the animation
    anim = animation.FuncAnimation(fig, anim_func,
                                   frames=tot_frames,
                                   blit=True)
    # save
    anim.save(filename, writer='ffmpeg', fps=video_frame_rate)
