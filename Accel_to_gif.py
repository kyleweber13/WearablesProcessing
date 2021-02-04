import pandas as pd
import matplotlib.pyplot as plt
import imageio
import os
import numpy as np
from Filtering import filter_signal
import ImportEDF
import matplotlib.dates as mdates
from datetime import timedelta
import Filtering


class GifGenerator:

    def __init__(self, wrist_file=None, ankle_file=None, start_time=None, stop_time=None,
                 sample_rate=75, wrist_obj=None, ankle_obj=None,
                 output_dir=None, remove_gravity=False, remove_high_f=False, remove_dc=True):

        self.wrist_file = wrist_file
        self.wrist_obj = wrist_obj
        self.ankle_file = ankle_file
        self.ankle_obj = ankle_obj

        self.lw = None
        self.la = None

        self.start_time = start_time
        self.stop_time = stop_time
        self.sample_rate = sample_rate

        self.output_dir = output_dir
        self.rem_gravity = remove_gravity
        self.rem_dc = remove_dc
        self.rem_highf = remove_high_f

    def import_data(self):

        print("\nImporting data...")

        if self.wrist_obj is None:
            if "csv" in self.wrist_file:
                self.lw = pd.read_csv(self.wrist_file, skiprows=100)
            if "edf" in self.wrist_file:
                d = ImportEDF.GENEActiv(filepath=self.wrist_file, load_raw=True)

                self.lw = pd.DataFrame(list(zip(d.timestamps, d.x, d.y, d.z, [None for i in range(len(d.timestamps))],
                                                [None for i in range(len(d.timestamps))],
                                                [None for i in range(len(d.timestamps))])))

            self.lw.columns = ["Timestamp", "x", "y", "z", "light", 'button', 'temperature']
            self.lw["Timestamp"] = pd.to_datetime(self.lw["Timestamp"], format="%Y-%m-%d %H:%M:%S:%f")

            self.lw["Timestamp"] = pd.date_range(start=self.lw['Timestamp'].iloc[0], periods=self.lw.shape[0],
                                                 freq='{}ms'.format(1000/self.sample_rate))

        if self.ankle_obj is None:
            if "csv" in self.ankle_file:
                self.la = pd.read_csv(self.ankle_file, skiprows=100)
            if "edf" in self.ankle_file:
                d = ImportEDF.GENEActiv(filepath=self.ankle_file, load_raw=True)

                # d.sample_rate = 50

                self.la = pd.DataFrame(list(zip(d.timestamps, d.x, d.y, d.z, [None for i in range(len(d.timestamps))],
                                                [None for i in range(len(d.timestamps))],
                                                [None for i in range(len(d.timestamps))])))

            self.la.columns = ["Timestamp", "x", "y", "z", "light", 'button', 'temperature']
            self.la["Timestamp"] = pd.to_datetime(self.la["Timestamp"], format="%Y-%m-%d %H:%M:%S:%f")

            self.la["Timestamp"] = pd.date_range(start=self.la['Timestamp'].iloc[0], periods=self.la.shape[0],
                                                 freq='{}ms'.format(1000/self.sample_rate))

    def remove_gravity(self):

        if self.rem_gravity:
            print("-Filtering data to remove gravity...")

            self.lw["x_filt"] = filter_signal(data=self.lw["x"], filter_type="highpass", high_f=0.1, filter_order=2,
                                              sample_f=self.sample_rate)
            self.lw["y_filt"] = filter_signal(data=self.lw["y"], filter_type="highpass", high_f=0.1, filter_order=2,
                                              sample_f=self.sample_rate)
            self.lw["z_filt"] = filter_signal(data=self.lw["z"], filter_type="highpass", high_f=0.1, filter_order=2,
                                              sample_f=self.sample_rate)

            self.la["x_filt"] = filter_signal(data=self.la["x"], filter_type="highpass", high_f=0.1, filter_order=2,
                                              sample_f=self.sample_rate)
            self.la["y_filt"] = filter_signal(data=self.la["y"], filter_type="highpass", high_f=0.1, filter_order=2,
                                              sample_f=self.sample_rate)
            self.la["z_filt"] = filter_signal(data=self.la["z"], filter_type="highpass", high_f=0.1, filter_order=2,
                                              sample_f=self.sample_rate)

    def remove_dc(self):

        if self.rem_dc:
            print("\nFiltering to remove DC...")

            self.lw["x_filt"] = [i - np.mean(self.lw["x"]) for i in self.lw["x"]]
            self.lw["y_filt"] = [i - np.mean(self.lw["y"]) for i in self.lw["y"]]
            self.lw["z_filt"] = [i - np.mean(self.lw["z"]) for i in self.lw["z"]]

            self.la["x_filt"] = [i - np.mean(self.la["x"]) for i in self.la["x"]]
            self.la["y_filt"] = [i - np.mean(self.la["y"]) for i in self.la["y"]]
            self.la["z_filt"] = [i - np.mean(self.la["z"]) for i in self.la["z"]]

    def remove_high_freq(self):

        if self.rem_highf:
            print("\nFiltering data to remove high-frequency noise...")

            self.lw["x_filt"] = filter_signal(data=self.lw["x"], filter_type="lowpass", low_f=5, filter_order=2,
                                              sample_f=self.sample_rate)
            self.lw["y_filt"] = filter_signal(data=self.lw["y"], filter_type="lowpass", low_f=5, filter_order=2,
                                              sample_f=self.sample_rate)
            self.lw["z_filt"] = filter_signal(data=self.lw["z"], filter_type="lowpass", low_f=5, filter_order=2,
                                              sample_f=self.sample_rate)

            self.la["x_filt"] = filter_signal(data=self.la["x"], filter_type="lowpass", low_f=5, filter_order=2,
                                              sample_f=self.sample_rate)
            self.la["y_filt"] = filter_signal(data=self.la["y"], filter_type="lowpass", low_f=5, filter_order=2,
                                              sample_f=self.sample_rate)
            self.la["z_filt"] = filter_signal(data=self.la["z"], filter_type="lowpass", low_f=5, filter_order=2,
                                              sample_f=self.sample_rate)

    def create_plot(self, start=None, stop=None, use_timestamps=False, slide_window=False, window_len=None,
                    acc_min=None, acc_max=None, column_suffix="", show_plot=True, save_plot=False, fname=""):

        if start is None and stop is None:
            start_stamp = self.start_time
            stop_stamp = self.stop_time
        if start is not None and stop is not None:
            start_stamp = start
            stop_stamp = stop
        if start_stamp is None and stop_stamp is None:
            start_stamp = self.lw["Timestamp"].iloc[0]
            stop_stamp = self.lw["Timestamp"].iloc[-1]

        lw = self.lw.loc[(self.lw["Timestamp"] >= pd.to_datetime(start_stamp)) &
                         (self.lw["Timestamp"] < pd.to_datetime(stop_stamp))]

        la = self.la.loc[(self.la["Timestamp"] >= pd.to_datetime(start_stamp)) &
                         (self.la["Timestamp"] < pd.to_datetime(stop_stamp))]

        lw_x = [i for i in lw["x{}".format(column_suffix)]]
        lw_y = [i for i in lw["y{}".format(column_suffix)]]
        lw_z = [i for i in lw["z{}".format(column_suffix)]]

        la_x = [i for i in la["x{}".format(column_suffix)]]
        la_y = [i for i in la["y{}".format(column_suffix)]]
        la_z = [i for i in la["z{}".format(column_suffix)]]

        if acc_max is None and acc_min is None:
            # Min/max data used for xlims and ylims
            min_x = min([min(lw_x), min(la_x)])
            min_y = min([min(lw_y), min(la_y)])
            min_z = min([min(lw_z), min(la_z)])

            max_x = max([max(lw_x), max(la_x)])
            max_y = max([max(lw_y), max(la_y)])
            max_z = max([max(lw_z), max(la_z)])

            min_all = min([min_x, min_y, min_z])
            max_all = max([max_x, max_y, max_z])

        if acc_max is not None and acc_min is not None:
            min_all = acc_min
            max_all = acc_max

        if not use_timestamps:
            time = [i / self.sample_rate for i in range(lw.shape[0])]
        if use_timestamps:
            time = [i for i in lw["Timestamp"]]

        # PLOTTING ----------------------------------------------------------------------------------------------------
        if not show_plot:
            plt.ioff()

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 6))

        if save_plot:
            plt.subplots_adjust(right=.75, left=.07, hspace=.3)

        ax1.plot(time, lw_x, color='black')
        ax1.plot(time, lw_y, color='red')
        ax1.plot(time, lw_z, color='dodgerblue')
        ax1.axvline(time[-1], color='limegreen')

        ax2.plot(time, la_x, color='black')
        ax2.plot(time, la_y, color='red')
        ax2.plot(time, la_z, color='dodgerblue')
        ax2.axvline(time[-1], color='limegreen')

        if not slide_window:

            if window_len is not None:
                if not use_timestamps:
                    # ax1.set_xlim(0, len(lw_x)/self.sample_rate)
                    # ax2.set_xlim(0, len(la_x)/self.sample_rate)
                    ax1.set_xlim(0, window_len)
                    ax2.set_xlim(0, window_len)
                if use_timestamps:
                    ax1.set_xlim(start_stamp, stop_stamp)
                    ax2.set_xlim(start_stamp, stop_stamp)

        if slide_window:
            if not use_timestamps:
                if time[-1] <= 12.5:
                    ax1.set_xlim(0, 15)
                    ax2.set_xlim(0, 15)
                if time[-1] > 12.5:
                    ax1.set_xlim(time[-1] - 12.5, time[-1] + 2.5)
                    ax2.set_xlim(time[-1] - 12.5, time[-1] + 2.5)

            if use_timestamps:
                if (time[-1] - time[0]).total_seconds() <= 15:
                    ax1.set_xlim(start_stamp, stop_stamp)
                    ax2.set_xlim(start_stamp, stop_stamp)
                if (time[-1] - time[0]).total_seconds() > 15:
                    ax1.set_xlim(time[-1] + timedelta(seconds=-12.5), time[-1] + timedelta(seconds=2.5))
                    ax2.set_xlim(time[-1] + timedelta(seconds=-12.5), time[-1] + timedelta(seconds=2.5))

        ax1.set_ylim(min_all - .5, max_all + .5)
        ax2.set_ylim(min_all - .5, max_all + .5)

        ax1.set_ylabel("Acceleration")
        ax2.set_ylabel("Acceleration")

        if not use_timestamps:
            ax2.set_xlabel("Seconds")
        if use_timestamps:
            ax2.set_xlabel("Timestamp")

        ax1.set_title("Left Wrist")
        ax2.set_title("Left Ankle")

        if save_plot:
            # create file name and append it to a list
            filename = "{}.png".format(fname)

            plt.savefig(self.output_dir + filename)
            plt.close()

    def make_gif(self, start=None, stop=None, plot_period_ms=40, slide_window=True, column_suffix=""):

        frame_stamps = pd.date_range(start=start, end=stop, freq="{}ms".format(plot_period_ms))

        lw = self.lw.loc[(self.lw["Timestamp"] >= start) & (self.lw["Timestamp"] < stop)]
        la = self.la.loc[(self.la["Timestamp"] >= start) & (self.la["Timestamp"] < stop)]

        window_len = (stop - start).total_seconds()

        # Min/max data used for xlims and ylims
        min_x = min([min(lw["x".format(column_suffix)]), min(la["x".format(column_suffix)])])
        min_y = min([min(lw["y".format(column_suffix)]), min(la["y".format(column_suffix)])])
        min_z = min([min(lw["z".format(column_suffix)]), min(la["z".format(column_suffix)])])

        max_x = max([max(lw["x".format(column_suffix)]), max(la["x".format(column_suffix)])])
        max_y = max([max(lw["y".format(column_suffix)]), max(la["y".format(column_suffix)])])
        max_z = max([max(lw["z".format(column_suffix)]), max(la["z".format(column_suffix)])])

        min_all = min([min_x, min_y, min_z])
        max_all = max([max_x, max_y, max_z])

        filenames = []

        for i, frame in enumerate(frame_stamps[1:]):
            print("-Generating plot {} of {}...".format(i + 1, len(frame_stamps) - 1))

            self.create_plot(start=start, stop=frame, use_timestamps=False, slide_window=slide_window,
                             window_len=window_len, save_plot=True,
                             fname="Image_" + str(i), acc_min=min_all, acc_max=max_all, show_plot=False)

            filenames.append("Image_{}.png".format(i))

        print("\nCombining images into gif...")
        with imageio.get_writer(self.output_dir + "Output.gif", mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(self.output_dir + filename)
                writer.append_data(image)

        # Remove files
        for filename in set(filenames):
            os.remove(self.output_dir + filename)

        print("\nComplete.")


def create_plot_gif(wrist_file=None, ankle_file=None, start_time=None, stop_time=None,
                    sample_rate=75, plot_period_ms=100, wrist_obj=None, ankle_obj=None,
                    output_dir=None,
                    slide_window=False, remove_gravity=False, remove_high_f=False, remove_dc=True):

    print("\nImporting data...")

    if wrist_obj is None:
        if "csv" in wrist_file:
            lw = pd.read_csv(wrist_file, skiprows=100)
        if "edf" in wrist_file:
            d = ImportEDF.GENEActiv(filepath=wrist_file, load_raw=True)

            d.sample_rate = 50

            lw = pd.DataFrame(list(zip(d.timestamps, d.x, d.y, d.z, [None for i in range(len(d.timestamps))],
                              [None for i in range(len(d.timestamps))], [None for i in range(len(d.timestamps))])))

        lw.columns = ["Timestamp", "x", "y", "z", "light", 'button', 'temperature']
        lw["Timestamp"] = pd.to_datetime(lw["Timestamp"], format="%Y-%m-%d %H:%M:%S:%f")

    if start_time is not None and stop_time is not None:
        lw = lw.loc[(lw["Timestamp"] >= pd.to_datetime(start_time)) &
                    (lw["Timestamp"] < pd.to_datetime(stop_time))]

    if ankle_obj is None:
        if "csv" in ankle_file:
            la = pd.read_csv(ankle_file, skiprows=100)
        if "edf" in ankle_file:
            d = ImportEDF.GENEActiv(filepath=ankle_file, load_raw=True)

            d.sample_rate = 50

            la = pd.DataFrame(list(zip(d.timestamps, d.x, d.y, d.z, [None for i in range(len(d.timestamps))],
                                       [None for i in range(len(d.timestamps))],
                                       [None for i in range(len(d.timestamps))])))

        la.columns = ["Timestamp", "x", "y", "z", "light", 'button', 'temperature']
        la["Timestamp"] = pd.to_datetime(la["Timestamp"], format="%Y-%m-%d %H:%M:%S:%f")

    if start_time is not None and stop_time is not None:
        la = la.loc[(la["Timestamp"] >= pd.to_datetime(start_time)) &
                    (la["Timestamp"] < pd.to_datetime(stop_time))]

    filenames = []

    # Converts cropped data to list
    time = [i / sample_rate for i in range(lw.shape[0])]
    lw_x = [i for i in lw["x"]]
    lw_y = [i for i in lw["y"]]
    lw_z = [i for i in lw["z"]]
    la_x = [i for i in la["x"]]
    la_y = [i for i in la["y"]]
    la_z = [i for i in la["z"]]

    if remove_gravity:

        print("-Filtering data to remove gravity...")

        lw_x = filter_signal(data=lw_x, filter_type="highpass", high_f=0.1, filter_order=2, sample_f=sample_rate)
        lw_y = filter_signal(data=lw_y, filter_type="highpass", high_f=0.1, filter_order=2, sample_f=sample_rate)
        lw_z = filter_signal(data=lw_z, filter_type="highpass", high_f=0.1, filter_order=2, sample_f=sample_rate)
        la_x = filter_signal(data=la_x, filter_type="highpass", high_f=0.1, filter_order=2, sample_f=sample_rate)
        la_y = filter_signal(data=la_y, filter_type="highpass", high_f=0.1, filter_order=2, sample_f=sample_rate)
        la_z = filter_signal(data=la_z, filter_type="highpass", high_f=0.1, filter_order=2, sample_f=sample_rate)

    if remove_high_f:

        print("-Filtering data to remove high frequency...")

        lw_x = filter_signal(data=lw_x, filter_type="lowpass", low_f=5, filter_order=2, sample_f=sample_rate)
        lw_y = filter_signal(data=lw_y, filter_type="lowpass", low_f=5, filter_order=2, sample_f=sample_rate)
        lw_z = filter_signal(data=lw_z, filter_type="lowpass", low_f=5, filter_order=2, sample_f=sample_rate)
        la_x = filter_signal(data=la_x, filter_type="lowpass", low_f=5, filter_order=2, sample_f=sample_rate)
        la_y = filter_signal(data=la_y, filter_type="lowpass", low_f=5, filter_order=2, sample_f=sample_rate)
        la_z = filter_signal(data=la_z, filter_type="lowpass", low_f=5, filter_order=2, sample_f=sample_rate)

    if remove_dc:
        print("\n-Removing DC component from signal...")

        lw_x = [i - np.mean(lw_x) for i in lw_x]
        lw_y = [i - np.mean(lw_y) for i in lw_y]
        lw_z = [i - np.mean(lw_z) for i in lw_z]
        la_x = [i - np.mean(la_x) for i in la_x]
        la_y = [i - np.mean(la_y) for i in la_y]
        la_z = [i - np.mean(la_z) for i in la_z]

    min_x = min([min(lw_x), min(la_x)])
    min_y = min([min(lw_y), min(la_y)])
    min_z = min([min(lw_z), min(la_z)])

    max_x = max([max(lw_x), max(la_x)])
    max_y = max([max(lw_y), max(la_y)])
    max_z = max([max(lw_z), max(la_z)])

    min_all = min([min_x, min_y, min_z])
    max_all = max([max_x, max_y, max_z])

    plot_rate = int(np.ceil(plot_period_ms / (1000 / sample_rate)))
    if plot_rate == 0:
        plot_rate = 1

    print("\n-Data will be plotted in {}ms increments...\n".format(plot_period_ms))

    for i in range(0, lw.shape[0], plot_rate):

        print("-Generating plot {} of {}...".format(int((i/plot_rate))+1, int(len(range(0, lw.shape[0], plot_rate)))))

        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 6))
        plt.subplots_adjust(right=.75, left=.07, hspace=.3)

        ax1.plot(time[:i], lw_x[:i], color='black')
        ax1.plot(time[:i], lw_y[:i], color='red')
        ax1.plot(time[:i], lw_z[:i], color='dodgerblue')
        ax1.axvline(time[i], color='limegreen')

        ax2.plot(time[:i], la_x[:i], color='black')
        ax2.plot(time[:i], la_y[:i], color='red')
        ax2.plot(time[:i], la_z[:i], color='dodgerblue')
        ax2.axvline(time[i], color='limegreen')

        ax1.set_ylim(min_all - .5, max_all + .5)
        ax2.set_ylim(min_all - .5, max_all + .5)

        if not slide_window:
            ax1.set_xlim(0, len(lw_x)/sample_rate)
            ax2.set_xlim(0, len(la_x)/sample_rate)

        if slide_window:
            if time[i] <= 12.5:
                ax1.set_xlim(0, 15)
                ax2.set_xlim(0, 15)
            if time[i] > 12.5:
                ax1.set_xlim(time[i]-7.5, time[i]+7.5)
                ax2.set_xlim(time[i]-7.5, time[i]+7.5)

        ax2.set_xlabel("Time (seconds)")
        ax1.set_ylabel("Acceleration")
        ax2.set_ylabel("Acceleration")
        ax1.set_title("Left Wrist")
        ax2.set_title("Left Ankle")
        ax1.set_ylabel("Acceleration")

        # create file name and append it to a list
        filename = f'{i}.png'
        filenames.append(filename)

        plt.savefig(output_dir + filename)
        plt.close()

    # build gif
    print("\nCombining images into gif...")
    with imageio.get_writer(output_dir + "Output.gif", mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(output_dir + filename)
            writer.append_data(image)

    # Remove files
    for filename in set(filenames):
        os.remove(output_dir + filename)

    print("\nComplete.")

    return lw, la


data_l = GifGenerator(wrist_file="/Users/kyleweber/Desktop/Family & Friends Day Data/Ben Videos and Data/Ben_LW_Accelerometer.edf",
                    ankle_file="/Users/kyleweber/Desktop/Family & Friends Day Data/Ben Videos and Data/Ben_LA_Accelerometer.edf",
                    start_time=None, stop_time=None,
                    wrist_obj=None, ankle_obj=None,
                    sample_rate=50, remove_gravity=False, remove_dc=False, remove_high_f=True,
                    output_dir="/Users/kyleweber/Desktop/Family & Friends Day Data/Photos/")

data_l.import_data()
# data.remove_gravity()
data_l.remove_high_freq()
# data.remove_dc()

data_r = GifGenerator(wrist_file="/Users/kyleweber/Desktop/Family & Friends Day Data/Ben Videos and Data/Ben_RW_Accelerometer.edf",
                    ankle_file="/Users/kyleweber/Desktop/Family & Friends Day Data/Ben Videos and Data/Ben_RA_Accelerometer.edf",
                    start_time=None, stop_time=None,
                    wrist_obj=None, ankle_obj=None,
                    sample_rate=50, remove_gravity=False, remove_dc=False, remove_high_f=True,
                    output_dir="/Users/kyleweber/Desktop/Family & Friends Day Data/Photos/")
data_r.import_data()
data_r.remove_high_freq()

# Pizza video
"""data.create_plot(start=pd.to_datetime("2021-01-22 19:12:52") + timedelta(seconds=189),
                 stop=pd.to_datetime("2021-01-22 19:12:52") + timedelta(seconds=209), use_timestamps=False,
                 show_plot=True, save_plot=False, column_suffix="_filt", slide_window=False)

# Walk video
walkvideo_start = pd.to_datetime("2021-01-21 12:16:53")
data.create_plot(start=walkvideo_start + timedelta(seconds=94),
                 stop=walkvideo_start + timedelta(seconds=114), use_timestamps=False,
                 show_plot=True, save_plot=False, column_suffix="_filt", slide_window=False)

data.make_gif(start=walkvideo_start + timedelta(seconds=94),
              stop=walkvideo_start + timedelta(seconds=119), plot_period_ms=90,
              slide_window=True, column_suffix="_filt")
"""
# Slip
# data.create_plot(start="2021-01-23 11:45:00", stop=pd.to_datetime("2021-01-23 11:45:00") + timedelta(seconds=180), use_timestamps=True)

# create_plot(data, None, None, True)
# create_plot(data, start=None, stop=None, use_timestamps=True)

"""
lw = data.lw.loc[(data.lw["Timestamp"] >= "2021-01-23 11:45:00") & (data.lw["Timestamp"] < "2021-01-23 11:47:00")]
la = data.la.loc[(data.la["Timestamp"] >= "2021-01-23 11:45:00") & (data.la["Timestamp"] < "2021-01-23 11:47:00")]

rw = data2.lw.loc[(data2.lw["Timestamp"] >= "2021-01-23 11:45:00") & (data2.lw["Timestamp"] < "2021-01-23 11:47:00")]
ra = data2.la.loc[(data2.la["Timestamp"] >= "2021-01-23 11:45:00") & (data2.la["Timestamp"] < "2021-01-23 11:47:00")]

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(10, 6))
ax1.plot(lw["Timestamp"], lw["x"], color='black', label='X')
ax1.plot(lw["Timestamp"], lw["y"], color='red', label='Y')
ax1.plot(lw["Timestamp"], lw["z"], color='dodgerblue', label='Z')
ax1.legend()
ax1.set_title("LWrist")

ax2.plot(rw["Timestamp"], rw["x"], color='black', label='X')
ax2.plot(rw["Timestamp"], rw["y"], color='red', label='Y')
ax2.plot(rw["Timestamp"], rw["z"], color='dodgerblue', label='Z')
ax2.set_title("RWrist")

ax3.plot(la["Timestamp"], la["x"], color='black', label='X')
ax3.plot(la["Timestamp"], la["y"], color='red', label='Y')
ax3.plot(la["Timestamp"], la["z"], color='dodgerblue', label='Z')
ax3.set_title("LAnkle")

ax4.plot(ra["Timestamp"], ra["x"], color='black', label='X')
ax4.plot(ra["Timestamp"], ra["y"], color='red', label='Y')
ax4.plot(ra["Timestamp"], ra["z"], color='dodgerblue', label='Z')
ax4.set_title("RAnkle")

xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S.%f")
ax4.xaxis.set_major_formatter(xfmt)
plt.xticks(rotation=45, fontsize=6)
"""

"""
video_start = pd.to_datetime("2021-01-22 18:54:52")
pizza_start = video_start + timedelta(seconds=2*60+48)
start_time = pd.to_datetime("2021-01-22 19:07:23")
stop_time = pizza_start + timedelta(seconds=25)

wrist_file = "/Users/kyleweber/Desktop/Family & Friends Day Data/Ben Videos and Data/Ben_LW_Accelerometer.EDF"
ankle_file = "/Users/kyleweber/Desktop/Family & Friends Day Data/Ben Videos and Data/Ben_LA_Accelerometer.EDF"

lw = ImportEDF.GENEActiv(filepath=wrist_file, load_raw=True)

lw = pd.DataFrame(list(zip(lw.timestamps, lw.x, lw.y, lw.z, [None for i in range(len(lw.timestamps))],
                  [None for i in range(len(lw.timestamps))], [None for i in range(len(lw.timestamps))])))
lw.columns = ["Timestamp", "x", "y", "z", "light", 'button', 'temperature']
# lw["Timestamp"] = pd.to_datetime(lw["Timestamp"], format="%Y-%m-%d %H:%M:%S:%f")

lw["Timestamp"] = pd.date_range(start=lw["Timestamp"].iloc[0], freq="20ms", periods=lw.shape[0])

la = ImportEDF.GENEActiv(filepath=ankle_file, load_raw=True)
la = pd.DataFrame(list(zip(la.timestamps, la.x, la.y, la.z, [None for i in range(len(la.timestamps))],
                  [None for i in range(len(la.timestamps))], [None for i in range(len(la.timestamps))])))
la.columns = ["Timestamp", "x", "y", "z", "light", 'button', 'temperature']
# la["Timestamp"] = pd.to_datetime(la["Timestamp"], format="%Y-%m-%d %H:%M:%S:%f")
la["Timestamp"] = pd.date_range(start=la["Timestamp"].iloc[0], freq="20ms", periods=la.shape[0])

lw_crop = lw.loc[(lw["Timestamp"] >= pizza_start) & (lw["Timestamp"] < stop_time)]
la_crop = la.loc[(la["Timestamp"] >= pizza_start) & (la["Timestamp"] < stop_time)]

fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 6))
ax1.plot(lw_crop["Timestamp"][::3], lw_crop["x"][::3], color='black')
ax1.plot(lw_crop["Timestamp"][::3], lw_crop["y"][::3], color='red')
ax1.plot(lw_crop["Timestamp"][::3], lw_crop["z"][::3], color='dodgerblue')
ax1.set_title("LWrist")

ax2.plot(la_crop["Timestamp"][::3], la_crop["x"][::3], color='black')
ax2.plot(la_crop["Timestamp"][::3], la_crop["y"][::3], color='red')
ax2.plot(la_crop["Timestamp"][::3], la_crop["z"][::3], color='dodgerblue')
ax2.set_title("LAnkle")

xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
ax2.xaxis.set_major_formatter(xfmt)
plt.xticks(rotation=45, fontsize=6)
"""
