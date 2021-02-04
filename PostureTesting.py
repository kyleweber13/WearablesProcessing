import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Filtering import filter_signal
import matplotlib.dates as mdates
from datetime import timedelta
import math
import ImportEDF

xfmt = mdates.DateFormatter("%H:%M:%S")


class Data:

    def __init__(self, la_file=None, ra_file=None, lw_file=None, rw_d_file=None, rw_p_file=None, bf_file=None,
                 event_log=None, fs=30):

        self.la_file = la_file
        self.ra_file = ra_file
        self.lw_file = lw_file
        self.rw_d_file = rw_d_file
        self.rw_p_file = rw_p_file
        self.bf_file = bf_file
        self.event_log = event_log
        self.fs = fs
        self.bf_fs = 25

        self.df_la = None
        self.df_ra = None
        self.df_lw = None
        self.df_rw_d = None
        self.df_rw_p = None
        self.df_bf = None
        self.df_event = None

        self.la_posture = None
        self.ra_posture = None
        self.lw_posture = None
        self.rw_d_posture = None
        self.rw_p_posture = None
        self.bf_posture = None

    def import_files(self):

        print("\nImporting data...")

        if self.la_file is not None:
            print("-Left ankle")
            self.df_la = pd.read_csv(self.la_file, skiprows=100, usecols=[0, 1, 2, 3, 6])
            self.df_la.columns = ["Timestamp", "x", "y", "z", "temperature"]
            self.df_la["Timestamp"] = pd.to_datetime(self.df_la["Timestamp"], format="%Y-%m-%d %H:%M:%S:%f")

        if self.ra_file is not None:
            print("-Right ankle")
            self.df_ra = pd.read_csv(self.ra_file, skiprows=100, usecols=[0, 1, 2, 3, 6])
            self.df_ra.columns = ["Timestamp", "x", "y", "z", "temperature"]
            self.df_ra["Timestamp"] = pd.to_datetime(self.df_ra["Timestamp"], format="%Y-%m-%d %H:%M:%S:%f")

        if self.lw_file is not None:
            print("-Left wrist")
            self.df_lw = pd.read_csv(self.lw_file, skiprows=100, usecols=[0, 1, 2, 3, 6])
            self.df_lw.columns = ["Timestamp", "x", "y", "z", "temperature"]
            self.df_lw["Timestamp"] = pd.to_datetime(self.df_lw["Timestamp"], format="%Y-%m-%d %H:%M:%S:%f")

        if self.rw_d_file is not None:
            print("-Right wrist #1")
            self.df_rw_d = pd.read_csv(self.rw_d_file, skiprows=100, usecols=[0, 1, 2, 3, 6])
            self.df_rw_d.columns = ["Timestamp", "x", "y", "z", "temperature"]
            self.df_rw_d["Timestamp"] = pd.to_datetime(self.df_rw_d["Timestamp"], format="%Y-%m-%d %H:%M:%S:%f")

        if self.rw_p_file is not None:
            print("-Right wrist #2")
            self.df_rw_p = pd.read_csv(self.rw_p_file, skiprows=100, usecols=[0, 1, 2, 3, 6])
            self.df_rw_p.columns = ["Timestamp", "x", "y", "z", "temperature"]
            self.df_rw_p["Timestamp"] = pd.to_datetime(self.df_rw_p["Timestamp"], format="%Y-%m-%d %H:%M:%S:%f")

        if self.bf_file is not None:
            print("-Bittium Faros")

            import pyedflib

            file = pyedflib.EdfReader(self.bf_file)
            x = file.readSignal(chn=0, start=0)
            y = file.readSignal(chn=1, start=0)
            z = file.readSignal(chn=2, start=0)
            self.bf_fs = file.getSampleFrequencies()[1]
            starttime = file.getStartdatetime()
            timestamps = pd.date_range(start=starttime, end=starttime + timedelta(seconds=len(x)/self.bf_fs),
                                       freq="{}ms".format(1000/self.bf_fs))

            self.df_bf = pd.DataFrame(list(zip(timestamps, x, y, z)),
                                      columns=["Timestamp", "x", "y", "z"])

            file.close()

        if self.event_log is not None:
            print("-Event log")
            self.df_event = pd.read_excel(self.event_log)
            self.df_event["Start"] = pd.to_datetime(self.df_event["Start"])
            self.df_event["Stop"] = pd.to_datetime(self.df_event["Stop"])

        print("Complete.")

    def filter_accels(self, low_f=0.05, high_f=15, filter_type="lowpass"):
        print("\nFiltering data...")

        # Left ankle
        if self.df_la is not None:
            print("-Left ankle")
            self.df_la["x_filt"] = filter_signal(data=self.df_la["x"], filter_type=filter_type,
                                                 low_f=low_f, high_f=high_f, sample_f=self.fs)
            self.df_la["y_filt"] = filter_signal(data=self.df_la["y"], filter_type=filter_type,
                                                 low_f=low_f, high_f=high_f, sample_f=self.fs)
            self.df_la["z_filt"] = filter_signal(data=self.df_la["z"], filter_type=filter_type,
                                                 low_f=low_f, high_f=high_f, sample_f=self.fs)

        # Right ankle
        if self.df_ra is not None:
            print("-Right ankle")
            self.df_ra["x_filt"] = filter_signal(data=self.df_ra["x"], filter_type=filter_type,
                                                 low_f=low_f, high_f=high_f, sample_f=self.fs)
            self.df_ra["y_filt"] = filter_signal(data=self.df_ra["y"], filter_type=filter_type,
                                                 low_f=low_f, high_f=high_f, sample_f=self.fs)
            self.df_ra["z_filt"] = filter_signal(data=self.df_ra["z"], filter_type=filter_type,
                                                 low_f=low_f, high_f=high_f, sample_f=self.fs)

        # Left wrist
        if self.df_lw is not None:
            print("-Left wrist")
            self.df_lw["x_filt"] = filter_signal(data=self.df_lw["x"], filter_type=filter_type,
                                                 low_f=low_f, high_f=high_f, sample_f=self.fs)
            self.df_lw["y_filt"] = filter_signal(data=self.df_lw["y"], filter_type=filter_type,
                                                 low_f=low_f, high_f=high_f, sample_f=self.fs)
            self.df_lw["z_filt"] = filter_signal(data=self.df_lw["z"], filter_type=filter_type,
                                                 low_f=low_f, high_f=high_f, sample_f=self.fs)

        # Right wrist 1
        if self.df_rw_d is not None:
            print("-Right wrist #1")
            self.df_rw_d["x_filt"] = filter_signal(data=self.df_rw_d["x"], filter_type=filter_type,
                                                   low_f=low_f, high_f=high_f, sample_f=self.fs)
            self.df_rw_d["y_filt"] = filter_signal(data=self.df_rw_d["y"], filter_type=filter_type,
                                                   low_f=low_f, high_f=high_f, sample_f=self.fs)
            self.df_rw_d["z_filt"] = filter_signal(data=self.df_rw_d["z"], filter_type=filter_type,
                                                   low_f=low_f, high_f=high_f, sample_f=self.fs)

        # Right wrist 2
        if self.df_rw_p is not None:
            print("-Right wrist #2")
            self.df_rw_p["x_filt"] = filter_signal(data=self.df_rw_p["x"], filter_type=filter_type,
                                                   low_f=low_f, high_f=high_f, sample_f=self.fs)
            self.df_rw_p["y_filt"] = filter_signal(data=self.df_rw_p["y"], filter_type=filter_type,
                                                   low_f=low_f, high_f=high_f, sample_f=self.fs)
            self.df_rw_p["z_filt"] = filter_signal(data=self.df_rw_p["z"], filter_type=filter_type,
                                                   low_f=low_f, high_f=high_f, sample_f=self.fs)

        # Bittium Faros
        if self.df_bf is not None:
            print("-Bittium Faros")
            self.df_bf["x_filt"] = filter_signal(data=self.df_bf["x"], filter_type=filter_type,
                                                 low_f=low_f, high_f=high_f, sample_f=self.bf_fs)
            self.df_bf["y_filt"] = filter_signal(data=self.df_bf["y"], filter_type=filter_type,
                                                 low_f=low_f, high_f=high_f, sample_f=self.bf_fs)
            self.df_bf["z_filt"] = filter_signal(data=self.df_bf["z"], filter_type=filter_type,
                                                 low_f=low_f, high_f=high_f, sample_f=self.bf_fs)

        print("Complete.")

    def plot_filtered(self, show_events=True):

        print("\nPlotting filtered data...")

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(12, 9))
        plt.subplots_adjust(hspace=.3)

        plt.suptitle("007_Posture_Testing - Event Log Data")

        if self.lw_file is not None:
            ax1.plot(self.df_lw["Timestamp"], self.df_lw["x_filt"], color='black', label="x")
            ax1.plot(self.df_lw["Timestamp"], self.df_lw["y_filt"], color='red', label="y")
            ax1.plot(self.df_lw["Timestamp"], self.df_lw["z_filt"], color='dodgerblue', label="z")
            ax1.set_title("Left wrist")
            ax1.set_ylim(ax1.get_ylim()[0], 1.5)
        ax1.set_ylabel("G")
        ax1.legend(loc='upper right')

        if self.rw_p_file is not None:
            ax2.plot(self.df_rw_p["Timestamp"], self.df_rw_p["x_filt"], color='black', label="x")
            ax2.plot(self.df_rw_p["Timestamp"], self.df_rw_p["y_filt"], color='red', label="y")
            ax2.plot(self.df_rw_p["Timestamp"], self.df_rw_p["z_filt"], color='dodgerblue', label="z")
            ax2.set_title("Right wrist")
            ax2.set_ylim(ax2.get_ylim()[0], 1.5)
        ax2.set_ylabel("G")
        ax2.legend(loc='upper right')

        if self.la_file is not None:
            ax3.plot(self.df_la["Timestamp"], self.df_la["x_filt"], color='black', label="x")
            ax3.plot(self.df_la["Timestamp"], self.df_la["y_filt"], color='red', label="y")
            ax3.plot(self.df_la["Timestamp"], self.df_la["z_filt"], color='dodgerblue', label="z")
            ax3.set_title("Left ankle")
            ax3.set_ylim(ax3.get_ylim()[0], 1.5)
        ax3.set_ylabel("G")
        ax3.legend(loc='upper right')

        if self.ra_file is not None:
            ax4.plot(self.df_ra["Timestamp"], self.df_ra["x_filt"], color='black', label="x")
            ax4.plot(self.df_ra["Timestamp"], self.df_ra["y_filt"], color='red', label="y")
            ax4.plot(self.df_ra["Timestamp"], self.df_ra["z_filt"], color='dodgerblue', label="z")
            ax4.set_title("Right ankle")
            ax4.set_ylim(ax4.get_ylim()[0], 1.5)
        ax4.set_ylabel("G")
        ax4.legend(loc='upper right')

        ax4.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        color_dict = {0: 'red', 1: 'limegreen', 2: 'orange', 3: 'dodgerblue', 4: 'yellow', 5: 'purple', 6: 'grey',
                      7: 'lightcoral', 8: 'turquoise', 9: 'brown'}

        if show_events:
            for row in self.df_event.itertuples():

                ax1.fill_between(x=[row.Start, row.Stop], y1=-1, y2=1,
                                 color=color_dict[row.Index], alpha=.5)
                ax1.text(x=row.Start + timedelta(seconds=5), y=1.1, s=row.Event, fontsize=7)

                ax2.fill_between(x=[row.Start, row.Stop], y1=-1, y2=1,
                                 color=color_dict[row.Index], alpha=.5)
                ax2.text(x=row.Start + timedelta(seconds=5), y=1.1, s=row.Event, fontsize=7)

                ax3.fill_between(x=[row.Start, row.Stop], y1=-1, y2=1,
                                 color=color_dict[row.Index], alpha=.5)
                ax3.text(x=row.Start + timedelta(seconds=5), y=1.1, s=row.Event, fontsize=7)

                ax4.fill_between(x=[row.Start, row.Stop], y1=-1, y2=1,
                                 color=color_dict[row.Index], alpha=.5)
                ax4.text(x=row.Start + timedelta(seconds=5), y=1.1, s=row.Event, fontsize=7)

    def calculate_posture(self, device, epoch_len=5):

        print("\nCalculating posture for {} acclerometer in {}-second windows...".format(device, epoch_len))

        x_vals = []
        y_vals = []
        z_vals = []
        posture = []

        if device == "la":
            data = self.df_la
        if device == "lw":
            data = self.df_lw
        if device == "rw_d":
            data = self.df_rw_d
        if device == "rw_p":
            data = self.df_rw_p
        if device == "ra":
            data = self.df_ra

        for i in np.arange(0, data.shape[0], self.fs * epoch_len):
            x_avg = np.mean(data["x_filt"].iloc[i:i + self.fs * epoch_len])
            y_avg = np.mean(data["y_filt"].iloc[i:i + self.fs * epoch_len])
            z_avg = np.mean(data["z_filt"].iloc[i:i + self.fs * epoch_len])

            x_vals.append(x_avg)
            y_vals.append(y_avg)
            z_vals.append(z_avg)

            conditon_found = False

            if device == "la":
                if y_avg < 0 and abs(y_avg) / abs(z_avg) >= 1.5 and abs(y_avg) / abs(x_avg) >= 1.5:
                    posture.append("sit/stand/walk")
                    conditon_found = True
                if y_avg > 0 and abs(y_avg) / abs(z_avg) >= 1.5 and abs(y_avg) / abs(x_avg) >= 1.5:
                    posture.append("upside down")
                    conditon_found = True

                if x_avg > 0 and abs(x_avg) / abs(y_avg) >= 1.5 and abs(x_avg) / abs(z_avg) >= 1.5:
                    posture.append("prone")
                    conditon_found = True
                if x_avg < 0 and abs(x_avg) / abs(y_avg) >= 2 and abs(x_avg) / abs(z_avg) >= 1.5:
                    posture.append("supine/feet up")
                    conditon_found = True

                if z_avg < 0 and abs(z_avg) / abs(y_avg) >= 2 and abs(z_avg) / abs(x_avg) >= 1.5:
                    posture.append("lying right")
                    conditon_found = True
                if z_avg > 0 and abs(z_avg) / abs(y_avg) >= 2 and abs(z_avg) / abs(x_avg) >= 1.5:
                    posture.append("lying left")
                    conditon_found = True

                if not conditon_found:
                    posture.append("Other")

        df_avg = pd.DataFrame(list(zip(data["Timestamp"].iloc[::self.fs * epoch_len],
                                       x_vals, y_vals, z_vals, posture)),
                              columns=["Timestamp", "avg_x", "avg_y", "avg_z", "Posture"])

        del x_vals, y_vals, z_vals, posture

        print("Complete.")

        return df_avg

    def calculate_inclination_angle(self, dataframe=None, show_data=False, epoch_len=15):

        if epoch_len > 15:
            print("\nEpoch length more than 15 seconds? Please don't do that.")
            return None

        if dataframe is not None:

            """Calculates average value of each axis in 15-second windows. Gravitational component not removed."""
            x = [i for i in dataframe["x"]]
            y = [i for i in dataframe["y"]]
            z = [i for i in dataframe["z"]]

            avg_x, avg_y, avg_z, vm, x_angle, y_angle, z_angle = [], [], [], [], [], [], []

            for i in range(0, len(x), epoch_len * self.fs):
                ax = np.mean(x[i:i+self.fs*epoch_len])
                ay = np.mean(y[i:i+self.fs*epoch_len])
                az = np.mean(z[i:i+self.fs*epoch_len])

                mag = (ax**2 + ay**2 + az**2)**(1/2)

                avg_x.append(ax)
                avg_y.append(ay)
                avg_z.append(az)
                vm.append(mag)

                x_ang = math.acos((ax/mag))
                x_angle.append(x_ang*180/math.pi)
                y_ang = math.acos((ay/mag))
                y_angle.append(y_ang*180/math.pi)
                z_ang = math.acos((az/mag))
                z_angle.append(z_ang*180/math.pi)

            df = pd.DataFrame(list(zip(dataframe["Timestamp"].iloc[::self.fs*epoch_len], avg_x, avg_y, avg_z, vm,
                                       x_angle, y_angle, z_angle)),
                              columns=["Timestamp", "X", "Y", "Z", "VM", "X_angle", "Y_angle", "Z_angle"])

            if show_data:

                fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(12, 9))
                plt.subplots_adjust(hspace=.25)
                plt.suptitle("Inclination Angle ({}-second epochs)".format(epoch_len))

                ax1.plot(df["Timestamp"], df["X"], color='black', label="X")
                ax1.plot(df["Timestamp"], df["Y"], color='red', label="Y")
                ax1.plot(df["Timestamp"], df["Z"], color='dodgerblue', label="Z")
                ax1.set_ylabel("Average G's")
                ax1.set_title("Lowpass filtered accelerometer data")
                ax1.legend()

                ax2.plot(df["Timestamp"], df["X_angle"], color='black', label="X")
                ax2.plot(df["Timestamp"], df["Y_angle"], color='red', label="Y")
                ax2.plot(df["Timestamp"], df["Z_angle"], color='dodgerblue', label="Z")

                ax2.axhline(y=90, color='grey', linestyle='dashed', label='Horizontal')
                ax2.set_title("Inclination angle")
                ax2.set_ylabel("Degrees")
                ax2.legend(loc='lower right')

                ax2.xaxis.set_major_formatter(xfmt)
                plt.xticks(rotation=45, fontsize=8)

                for row in self.df_event.itertuples():
                    ax1.fill_between(x=[row.Start, row.Stop], y1=-1, y2=1,
                                     color="grey" if row.Index%2 == 0 else "lightgrey", alpha=.5)
                    ax1.text(x=row.Start + timedelta(seconds=5), y=1, s=row.Event, fontsize=7)

                    ax2.fill_between(x=[row.Start, row.Stop], y1=0, y2=180,
                                     color="grey" if row.Index%2 == 0 else "lightgrey", alpha=.5)
                    ax2.text(x=row.Start + timedelta(seconds=5), y=180, s=row.Event, fontsize=7)

            return df

        if dataframe is None:
            print("-Data does not exist. Try again.")

    def plot_angles(self, epoch_len=15, show_events=True):

        lw_data = self.calculate_inclination_angle(dataframe=self.df_lw, show_data=False, epoch_len=epoch_len)
        la_data = self.calculate_inclination_angle(dataframe=self.df_la, show_data=False, epoch_len=epoch_len)
        ra_data = self.calculate_inclination_angle(dataframe=self.df_ra, show_data=False, epoch_len=epoch_len)

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(12, 9))

        ax1.set_title("Left Wrist")
        ax2.set_title("Left Ankle")
        ax3.set_title("Right Ankle")

        color_dict = {0: 'red', 1: 'limegreen', 2: 'orange', 3: 'dodgerblue', 4: 'yellow', 5: 'purple', 6: 'grey',
                      7: 'lightcoral', 8: 'turquoise', 9: 'brown'}

        if lw_data is not None:
            ax1.plot(lw_data["Timestamp"], lw_data["X_angle"], color='black', label="X")
            ax1.plot(lw_data["Timestamp"], lw_data["Y_angle"], color='red', label="Y")
            ax1.plot(lw_data["Timestamp"], lw_data["Z_angle"], color='dodgerblue', label="Z")
            ax1.axhline(y=90, color='grey', linestyle='dashed', label='Horizontal')
            ax1.set_yticks(np.arange(0, 225, 45))
            ax1.set_ylabel("Degrees")
            ax1.legend(loc='lower right')

            if show_events:

                for row in self.df_event.itertuples():
                    ax1.fill_between(x=[row.Start, row.Stop], y1=0, y2=180,
                                     color=color_dict[row.Index], alpha=.5)
                    ax1.text(x=row.Start + timedelta(seconds=5), y=180, s=row.Event, fontsize=7)

        if la_data is not None:
            ax2.plot(la_data["Timestamp"], la_data["X_angle"], color='black', label="X")
            ax2.plot(la_data["Timestamp"], la_data["Y_angle"], color='red', label="Y")
            ax2.plot(la_data["Timestamp"], la_data["Z_angle"], color='dodgerblue', label="Z")
            ax2.axhline(y=90, color='grey', linestyle='dashed', label='Horizontal')
            ax2.set_yticks(np.arange(0, 225, 45))
            ax2.set_ylabel("Degrees")
            ax2.legend(loc='lower right')

            if show_events:

                for row in self.df_event.itertuples():
                    ax2.fill_between(x=[row.Start, row.Stop], y1=0, y2=180,
                                     color=color_dict[row.Index], alpha=.5)
                    ax2.text(x=row.Start + timedelta(seconds=5), y=180, s=row.Event, fontsize=7)

        if ra_data is not None:
            ax3.plot(ra_data["Timestamp"], ra_data["X_angle"], color='black', label="X")
            ax3.plot(ra_data["Timestamp"], ra_data["Y_angle"], color='red', label="Y")
            ax3.plot(ra_data["Timestamp"], ra_data["Z_angle"], color='dodgerblue', label="Z")
            ax3.axhline(y=90, color='grey', linestyle='dashed', label='Horizontal')
            ax3.set_yticks(np.arange(0, 225, 45))
            ax3.set_ylabel("Degrees")
            ax3.legend(loc='lower right')

            if show_events:

                for row in self.df_event.itertuples():
                    ax3.fill_between(x=[row.Start, row.Stop], y1=0, y2=180,
                                     color=color_dict[row.Index], alpha=.5)
                    ax3.text(x=row.Start + timedelta(seconds=5), y=180, s=row.Event, fontsize=7)

        ax3.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

    def calculate_posture_angles(self, device, epoch_len=5, plot_posture=True):
        """Calculates posture for individual device. Calls self.calculate_inclination_angle.

        :argument
        -device: which device's data to use. E.g. 'left ankle'
        -epoch_len: epoch length (seconds)
        -plot_posture: boolean

        :return
        -dataframe containing acceleromter, angle, and posture data
        """

        if device == "LA" or device == "left ankle" or device == "la" or \
                device == "leftankle" or device == "Left ankle":

            data = self.calculate_inclination_angle(dataframe=self.df_la, show_data=False, epoch_len=epoch_len)

            orient = []
            for row in data.itertuples():

                # Shin vertical = sit/stand
                if row.Y_angle >= 135 and row.Y_angle > row.X_angle and 45 <= row.X_angle <= 135 and \
                        row.Y_angle > row.Z_angle and 45 <= row.Z_angle <= 135:
                    orient.append("Sit/stand")

                # Shin horizontal = feet up (supine/reclined)
                elif row.X_angle >= 135 and row.X_angle > row.Y_angle and row.X_angle > row.Z_angle:
                    orient.append("Supine/recline")

                # Shin horizontal = feet up (prone)
                elif row.X_angle <= 90 and row.X_angle < row.Y_angle and row.X_angle < row.Z_angle <= 135:
                    orient.append("Prone")

                # Shin horizontal = lying on right side
                elif row.Z_angle >= 135 and row.Z_angle > row.X_angle and 45 <= row.X_angle <= 135 and \
                        row.Z_angle > row.Y_angle and 45 <= row.Y_angle <= 135:
                    orient.append("Lying right")

                # Shin horizontal = lying on left side
                elif row.Z_angle <= 90 and row.Z_angle < row.X_angle and 45 <= row.X_angle <= 135 and \
                        row.Z_angle < row.Y_angle and 45 <= row.Y_angle <= 135:
                    orient.append("Lying left")

                else:
                    orient.append("Other")

        if device == "bf_v" or device == "BF_V":
            data = self.calculate_inclination_angle(dataframe=self.df_bf, show_data=False, epoch_len=epoch_len)

            pass

        if plot_posture:
            fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(12, 9))
            plt.suptitle(device)
            ax1.plot(data["Timestamp"], data["X_angle"], color='black', label="X")
            ax1.plot(data["Timestamp"], data["Y_angle"], color='red', label="Y")
            ax1.plot(data["Timestamp"], data["Z_angle"], color='dodgerblue', label="Z")
            ax1.legend()
            ax1.set_ylabel("Angle")
            ax2.plot(data["Timestamp"], orient, color='black', marker="o", markersize=2)

            ylim0 = ax2.get_ylim()[0]
            ylim1 = ax2.get_ylim()[1]

            for row in self.df_event.itertuples():
                ax2.fill_between(x=[row.Start, row.Stop], y1=ylim0, y2=ylim1,
                                 color="grey" if row.Index%2 == 0 else "dodgerblue", alpha=.5)
                ax2.text(x=row.Start + timedelta(seconds=5), y="Sit/stand", s=row.Event, fontsize=8)

            ax2.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45, fontsize=8)

        data["Posture"] = orient

        return data


"""data = Data(la_file="O:/OBI/Personal Folders/Kyle Weber/Posture/007_Posture_SampleDataset/Test1/007_LAnkle.csv",
            ra_file="O:/OBI/Personal Folders/Kyle Weber/Posture/007_Posture_SampleDataset/Test1/007_RAnkle.csv",
            lw_file="O:/OBI/Personal Folders/Kyle Weber/Posture/007_Posture_SampleDataset/Test1/007_LWrist.csv",
            rw_d_file="O:/OBI/Personal Folders/Kyle Weber/Posture/007_Posture_SampleDataset/Test1/007_RWrist_Distal.csv",
            rw_p_file="O:/OBI/Personal Folders/Kyle Weber/Posture/007_Posture_SampleDataset/Test1/007_RWrist_Proximal.csv",
            event_log="O:/OBI/Personal Folders/Kyle Weber/Posture/007_Posture_SampleDataset/Test1/EventLog.xlsx")
"""

data = Data(la_file="O:/OBI/Personal Folders/Kyle Weber/Posture/007_Posture_SampleDataset/Test2/Test_LA.csv",
            lw_file="O:/OBI/Personal Folders/Kyle Weber/Posture/007_Posture_SampleDataset/Test2/Test_LW.csv",
            bf_file="O:/OBI/Personal Folders/Kyle Weber/Posture/007_Posture_SampleDataset/Test2/Test_BF_V.edf",
            event_log="O:/OBI/Personal Folders/Kyle Weber/Posture/007_Posture_SampleDataset/Test2/EventLog.xlsx")

data.import_files()
data.filter_accels(low_f=0.1, high_f=15, filter_type="lowpass")
# data.plot_filtered(show_events=False)
# data.la_posture = data.calculate_posture(device="la", epoch_len=5)

# data.plot_angles(epoch_len=5, show_events=True)

# la_posture = data.calculate_posture_angles(device='Left ankle', epoch_len=5, plot_posture=True)
