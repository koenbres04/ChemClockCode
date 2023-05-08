import json
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
import numpy as np
from math import ceil


class Visualize:
    def __init__(self, input_map: str):
        """
        Visualizes the two-dimensional solution of a differential equation
        :param input_map: path to simulation file
        """
        # load the values and parameters
        values_path = os.path.join(input_map, "raw.npy")
        self.values = np.load(values_path)
        with open(os.path.join(input_map, "params.json"), "r") as read_json:
            params = json.load(read_json)
        # unpack the dict
        self.width = params["width"]
        self.height = params["height"]
        self.ds = params["ds"]
        self.dt = params["dt"]
        self.t_end = params["t_end"]

    def plot_frames(self, filename: str, channel: str, output_particle: int,
                    min_t: float, max_t: float, num_frames: int):
        """
        Visualize the two-dimensional solution by plotting different slide timestamps next to each other
        :param filename: name of output file
        :param channel: string name of particle name
        :param output_particle: integer that indexes the particle
        :param min_t: time in the simulation to start
        :param max_t: time in the simulation to stop
        :param num_frames: total number of frames we want to show
        """
        # cap the min_t and max_t
        max_t = min(self.t_end, max_t)
        min_t = max(0.0, min_t)

        # retrieve relevant information
        values = np.swapaxes(self.values, 0, 1)[output_particle]
        timestamps = np.linspace(min_t, max_t, num_frames)
        u = np.arange(0, self.width*self.ds, self.ds)
        v = np.arange(0, self.height*self.ds, self.ds)
        # initialize the plot
        num_cols = 5
        num_rows = ceil(num_frames/5)
        fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(num_cols*4, num_rows*4))
        axs = list(axs.flatten())
        # initialize the density and normal plots and add a time text
        for t, ax in zip(timestamps, axs):
            dens = ax.pcolormesh(u, v, values[round(values.shape[0]*t/self.t_end), :, :], cmap="viridis",
                                 shading="auto",
                                 vmin=np.min(values),
                                 vmax=np.max(values))
            ax.set_title(f"{channel} on t={round(t, 2)}")
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        norm = matplotlib.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
        colorscale = matplotlib.cm.ScalarMappable(norm=norm, cmap="viridis")
        fig.colorbar(colorscale, cax=cbar_ax, ticks=np.linspace(np.min(values), np.max(values), 10))
        # delete empty plots
        for ax in axs[num_frames:]:
            fig.delaxes(ax)
        # save the plot
        plt.savefig(filename)

    def track_point(self, filename: str, channels: list[str],
                    output_particles: list[int], min_t: float, max_t: float, track_point):
        """
        Visualizes the two-dimensional solution of a differential equation
        :param filename: name of output file
        :param channels: list with names of the particles in order of the value list
        :param output_particles: a list with indices of the particles we want to show
        :param min_t: time in the simulation to start the animation
        :param max_t: time in the simulation to end the animation
        :param track_point: optional (x, y) coordinates to show over time next to the other plots
        """
        # cap the min_t and max_t
        max_t = min(self.t_end, max_t)
        min_t = max(0.0, min_t)

        values = [list(np.swapaxes(self.values, 0, 1))[i] for i in output_particles]
        # get data for track_point
        u = np.arange(0, self.width*self.ds, self.ds)
        v = np.arange(0, self.height*self.ds, self.ds)
        i_u_nearest = (np.abs(u - track_point[0])).argmin()
        i_v_nearest = (np.abs(v - track_point[1])).argmin()
        track_values = [value[:, i_u_nearest, i_v_nearest] for value in values]
        num2shape = {"1": (1, 1, 6, 4), "2": (2, 1, 12, 8), "3": (3, 1, 12, 8), "4": (2, 2, 20, 15)}
        # amount of particles that are shown
        num_particles = len(values)
        # initialize figure and axis
        fig_shape = num2shape[str(num_particles)]
        fig, axs = plt.subplots(fig_shape[0], fig_shape[1],
                                figsize=(fig_shape[2], fig_shape[3]))
        axs = list(axs.flatten())
        for i, ax in enumerate(axs):
            ax.plot(np.linspace(min_t, max_t, len(track_values[i])), track_values[i], color="black", linewidth=1.2)
            ax.set(xlabel=r"$\hat{t}$", ylabel=channels[i])
            ax.set_title(f"{channels[i]} over time")
        plt.savefig(filename)

    def grid_animation(self, video_frame_rate: int, video_t_per_second, filename: str,
                       channels: list[str], output_particles: list[int], min_t: float, max_t: float, track_point=None):
        """
        Visualizes the two-dimensional solution of a differential equation
        :param video_frame_rate: FPS of returned video
        :param video_t_per_second: determines how many simulation seconds pass in each real time second
        :param filename: name of output file
        :param channels: list with names of the particles in order of the value list
        :param output_particles: a list with indices of the particles we want to show
        :param min_t: time in the simulation to start the animation
        :param max_t: time in the simulation to end the animation
        :param track_point: optional (x, y) coordinates to show over time next to the other plots
        """
        # cap the min_t and max_t
        max_t = min(self.t_end, max_t)
        min_t = max(0.0, min_t)

        values = [list(np.swapaxes(self.values, 0, 1))[i] for i in output_particles]

        # create a mesh for the space that we simulate over
        u = np.arange(0, self.width*self.ds, self.ds)
        v = np.arange(0, self.height*self.ds, self.ds)

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
        tot_frames = round(values[0].shape[0]*self.dt/video_t_per_second*video_frame_rate)
        timestamps = np.linspace(min_t, max_t, tot_frames)

        # animation function
        def anim_func(frame):
            if track_point is not None:
                for i, dens, time_i, time_line in zip(range(num_particles), densities, time_texts, track_lines):
                    dens.set_array(values[i][round(frame/video_frame_rate*video_t_per_second/self.dt), :, :].ravel())
                    time_i.set_text(f"t={round(timestamps[frame], 2)} s")
                    time_line.set_xdata(np.array([timestamps[frame]]*2))
                return *densities, *time_texts, *track_lines
            else:
                for i, dens, time_i in zip(range(num_particles), densities, time_texts):
                    dens.set_array(values[i][round(frame/video_frame_rate*video_t_per_second/self.dt), :, :].ravel())
                    time_i.set_text(f"t={round(timestamps[frame], 2)} s")
                return *densities, *time_texts

        # initialize the density and normal plots and add a time text
        densities = [density_axs[i].pcolormesh(u, v, values[i][0, :, :], cmap="viridis", shading="auto",
                                               vmin=np.min(values[i]),
                                               vmax=np.max(values[i])) for i in range(num_particles)]
        time_texts = [ax.text(self.width*self.ds*0.01, self.height*self.ds*0.90, "t=0", fontsize=10) for ax in density_axs]
        if track_point is not None:
            track_lines = []
            for i, ax in enumerate(track_axs):
                ax.plot(np.linspace(min_t, max_t, len(track_values[i])), track_values[i], color="red", linewidth=1.2)
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


def frames_test(test_name):
    # folder to read from
    output_folder = "output"
    # parameters
    output_file_name = "framesTest"
    min_t = 0
    max_t = 30
    output_format = ".png"
    output_particle = 3
    channel = r"$\hat y$"
    num_frames = 17

    # load data
    found_folder = None
    for test_folder in os.listdir(output_folder):
        if test_folder.startswith(test_name):
            found_folder = test_folder
            break
    output_subfolder = os.path.join(output_folder, found_folder)

    # generate the animation
    print(f"Creating {found_folder}...")
    anim = Visualize(output_subfolder)
    anim.plot_frames(os.path.join(output_subfolder, f"{output_file_name}{output_format}"),
                     channel, output_particle, min_t, max_t, num_frames)


def track_test(test_name):
    # folder to read from
    output_folder = "output"
    # parameters
    output_file_name = "track"
    min_t = 0
    max_t = 30
    output_format = ".png"
    output_particles = [0, 1, 2, 3]
    channels = [r"$\hat p$", r"$\hat q$", r"$\hat x$", r"$\hat y$"]
    track_point = (1, 1)

    # load data
    found_folder = None
    for test_folder in os.listdir(output_folder):
        if test_folder.startswith(test_name):
            found_folder = test_folder
            break
    output_subfolder = os.path.join(output_folder, found_folder)

    # generate the animation
    print(f"Creating {found_folder}...")
    anim = Visualize(output_subfolder)
    anim.track_point(os.path.join(output_subfolder, f"{output_file_name}{output_format}"),
                     channels, output_particles, min_t, max_t, track_point=track_point)


def animate_test(test_name):
    # folder to read from
    output_folder = "output"
    # parameters
    output_file_name = "animation"
    video_frame_rate = 30
    video_t_per_second = 4
    min_t = 0
    max_t = 30
    output_format = ".mp4"
    output_particles = [0, 1, 2, 3]
    channels = [r"$\hat p$", r"$\hat q$", r"$\hat x$", r"$\hat y$"]
    track_point = (1, 1)

    # load data
    found_folder = None
    for test_folder in os.listdir(output_folder):
        if test_folder.startswith(test_name):
            found_folder = test_folder
            break
    output_subfolder = os.path.join(output_folder, found_folder)

    # generate the animation
    print(f"Animating {found_folder}...")
    anim = Visualize(output_subfolder)
    anim.grid_animation(video_frame_rate, video_t_per_second,
                        os.path.join(output_subfolder, f"{output_file_name}{output_format}"),
                        channels, output_particles, min_t, max_t, track_point=None)


if __name__ == '__main__':
    #animate_test(input("Give the name of the test:\n"))
    frames_test(input("Give the name of the test:\n"))
