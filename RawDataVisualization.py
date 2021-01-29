import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from ImportEDF import GENEActiv
from Filtering import filter_signal

xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")


class Data:

    def __init__(self, la_filepath=None, rw_filepath=None, lw_filepath=None, subj_id=None,
                 nonwear_file=None, sleep_file=None):

        self.subj_id = subj_id
        self.la = GENEActiv(filepath=la_filepath.format(subj_id), load_raw=True)
        self.rw = GENEActiv(filepath=rw_filepath.format(subj_id), load_raw=True)
        self.lw = GENEActiv(filepath=lw_filepath.format(subj_id), load_raw=True)

        if nonwear_file is not None:
            nw_file = pd.read_csv(nonwear_file)
            self.nw_file = nw_file.loc[nw_file["ID"] == int(subj_id)]

        if sleep_file is not None:
            self.sleep_file = pd.read_csv(sleep_file.format(subj_id))
            self.sleep_file.columns = ["Night", "Asleep", "Awake"]

        self.df_la_posture = None
        self.df_lw_posture = None
        self.df_rw_posture = None

    def filter_data(self, low_f=.05, high_f=15, filter_type="bandpass"):

        print("\nFiltering data...")

        # Left ankle
        self.la.x_filt = filter_signal(data=self.la.x, filter_type=filter_type,
                                       low_f=low_f, high_f=high_f, sample_f=self.la.sample_rate)
        self.la.y_filt = filter_signal(data=self.la.y, filter_type=filter_type,
                                       low_f=low_f, high_f=high_f, sample_f=self.la.sample_rate)
        self.la.z_filt = filter_signal(data=self.la.z, filter_type=filter_type,
                                       low_f=low_f, high_f=high_f, sample_f=self.la.sample_rate)

        # Left wrist
        self.lw.x_filt = filter_signal(data=self.lw.x, filter_type=filter_type,
                                       low_f=low_f, high_f=high_f, sample_f=self.lw.sample_rate)
        self.lw.y_filt = filter_signal(data=self.lw.y, filter_type=filter_type,
                                       low_f=low_f, high_f=high_f, sample_f=self.lw.sample_rate)
        self.lw.z_filt = filter_signal(data=self.lw.z, filter_type=filter_type,
                                       low_f=low_f, high_f=high_f, sample_f=self.lw.sample_rate)

        # Right wrist
        self.rw.x_filt = filter_signal(data=self.rw.x, filter_type=filter_type,
                                       low_f=low_f, high_f=high_f, sample_f=self.rw.sample_rate)
        self.rw.y_filt = filter_signal(data=self.rw.y, filter_type=filter_type,
                                       low_f=low_f, high_f=high_f, sample_f=self.rw.sample_rate)
        self.rw.z_filt = filter_signal(data=self.rw.z, filter_type=filter_type,
                                       low_f=low_f, high_f=high_f, sample_f=self.rw.sample_rate)

        print("Complete.")

    def calculate_posture(self, device="la", epoch_len=5):

        print("\nCalculating posture for {} acclerometer in {}-second windows...".format(device, epoch_len))

        x_vals = []
        y_vals = []
        z_vals = []
        posture = []

        if device == "la":
            data = self.la
        if device == "lw":
            data = self.lw
        if device == "rw":
            data = self.rw

        for i in np.arange(0, len(data.x_filt), data.sample_rate * epoch_len):
            x_avg = np.mean(data.x_filt[i:i + data.sample_rate * epoch_len])
            y_avg = np.mean(data.y_filt[i:i + data.sample_rate * epoch_len])
            z_avg = np.mean(data.z_filt[i:i + data.sample_rate * epoch_len])

            x_vals.append(x_avg)
            y_vals.append(y_avg)
            z_vals.append(z_avg)

            conditon_found = False

            if device == "la":
                if y_avg < 0 and abs(y_avg) / abs(z_avg) >= 1.5 and abs(y_avg) / abs(x_avg) >= 1.5:
                    posture.append("sit/stand")
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

        df_avg = pd.DataFrame(list(zip(data.timestamps[::data.sample_rate * epoch_len],
                                       x_vals, y_vals, z_vals, posture)),
                              columns=["Timestamp", "avg_x", "avg_y", "avg_z", "Posture"])

        del x_vals, y_vals, z_vals, posture

        return df_avg

    def plot_data(self, axes=("x", "y", "z"), fs=75, show_nonwear=True, show_sleep=True):

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(14, 9))

        plt.suptitle("{} ({}Hz)".format(self.subj_id, fs))
        ax1.set_title("Left Wrist")
        ax1.set_ylabel("G")
        lw_ds = int(self.lw.sample_rate / fs)

        ax2.set_title("Right Wrist")
        ax2.set_ylabel("G")
        rw_ds = int(self.rw.sample_rate / fs)

        ax3.set_title("Left Ankle")
        ax3.set_ylabel("G")
        la_ds = int(self.la.sample_rate / fs)

        if "x" in axes:
            ax1.plot(self.lw.timestamps[::lw_ds], self.lw.x[::lw_ds], color='black', label="X")
            ax2.plot(self.rw.timestamps[::rw_ds], self.rw.x[::rw_ds], color='black', label="X")
            ax3.plot(self.la.timestamps[::la_ds], self.la.x[::la_ds], color='black', label="X")
        if "y" in axes:
            ax1.plot(self.lw.timestamps[::lw_ds], self.lw.y[::lw_ds], color='red', label="Y")
            ax2.plot(self.rw.timestamps[::rw_ds], self.rw.y[::rw_ds], color='red', label="Y")
            ax3.plot(self.la.timestamps[::la_ds], self.la.y[::la_ds], color='red', label="Y")
        if "z" in axes:
            ax1.plot(self.lw.timestamps[::lw_ds], self.lw.z[::lw_ds], color='dodgerblue', label="Z")
            ax2.plot(self.rw.timestamps[::rw_ds], self.rw.z[::rw_ds], color='dodgerblue', label="Z")
            ax3.plot(self.la.timestamps[::la_ds], self.la.z[::la_ds], color='dodgerblue', label="Z")

        if show_nonwear:
            for row in self.nw_file.loc[self.nw_file["location"]=="LW"].itertuples():
                ax1.fill_between(x=[row.start_time, row.end_time], y1=-8, y2=8, color='grey', alpha=.5)
            for row in self.nw_file.loc[self.nw_file["location"]=="RW"].itertuples():
                ax2.fill_between(x=[row.start_time, row.end_time], y1=-8, y2=8, color='grey', alpha=.5)
            for row in self.nw_file.loc[self.nw_file["location"]=="LA"].itertuples():
                ax3.fill_between(x=[row.start_time, row.end_time], y1=-8, y2=8, color='grey', alpha=.5)
        if show_sleep:
            for row in self.sleep_file.itertuples():
                ax1.fill_between(x=[row.Asleep, row.Awake],
                                 y1=-8, y2=8, color='blue', alpha=.5)
            for row in self.sleep_file.itertuples():
                ax2.fill_between(x=[row.Asleep, row.Awake],
                                 y1=-8, y2=8, color='blue', alpha=.5)
            for row in self.sleep_file.itertuples():
                ax3.fill_between(x=[row.Asleep, row.Awake],
                                 y1=-8, y2=8, color='blue', alpha=.5)

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')
        ax3.legend(loc='upper left')

        ax3.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

    def plot_posture(self, device="la", data_type="raw"):

        if device == "la":
            raw = self.la
            avg = self.df_la_posture
        if device == "lw":
            raw = self.lw
            avg = self.df_lw_posture
        if device == "rw":
            raw = self.rw
            avg = self.df_rw_posture

        epoch_len = int((avg["Timestamp"].iloc[1]-avg["Timestamp"].iloc[0]).total_seconds())

        print("\nPlotting time series posture data for {} accelerometer "
              "({} accel data; {}-second epochs)".format(device, data_type, epoch_len))

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(12, 9))
        plt.suptitle("{}-second posture data".format(epoch_len))
        ax1.set_title("{}_{}".format(device, data_type))

        if data_type == "raw":
            ax1.plot(raw.timestamps[::3], raw.x[::3], color='black', label='x')
            ax1.plot(raw.timestamps[::3], raw.y[::3], color='red', label='y')
            ax1.plot(raw.timestamps[::3], raw.z[::3], color='dodgerblue', label='z')

        if data_type == "avg":
            ax1.plot(avg["Timestamp"], avg["avg_x"], color='black', label='x_avg')
            ax1.plot(avg["Timestamp"], avg["avg_y"], color='red', label="y_avg")
            ax1.plot(avg["Timestamp"], avg["avg_z"], color='dodgerblue', label="z_avg")
        ax1.set_ylabel("G")
        ax1.legend(loc='upper left')

        ax2.set_title("Posture data")
        ax2.plot(avg["Timestamp"], avg["Posture"], color='black')

        ax2.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)


data = Data(subj_id=1027,
            la_filepath="C:/Users/ksweber/Desktop/OND06_LAnkle/OND06_SBH_{}_GNAC_ACCELEROMETER_LAnkle.EDF",
            rw_filepath="C:/Users/ksweber/Desktop/OND06_RWrist_PD/OND06_SBH_{}_GNAC_ACCELEROMETER_RWrist.EDF",
            lw_filepath="D:/Accelerometer/OND06_SBH_{}_GNAC_ACCELEROMETER_LWrist.EDF",
            # nonwear_file="O:/Data/ReMiNDD/Processed Data/GENEActiv_Nonwear/ReMiNDDNonWearReformatted_GAgoldstandarddataset_16Dec2020.csv",
            # sleep_file="O:/Data/ReMiNDD/Processed Data/Sleep/asleep_wearing_start_stop_times/asleep_wearing_{}.csv"
            )

data.filter_data(low_f=.025, high_f=10, filter_type='lowpass')

data.df_la_posture = data.calculate_posture(device="la", epoch_len=5)
data.plot_posture(data, device='la', data_type='avg')

# data.plot_data(axes=["x", "y"], fs=25, show_nonwear=True, show_sleep=True)

# Break
# la_avg.plot(x="Timestamp", y=["avg_x", "avg_y", "avg_z"], color=["black", "red", "dodgerblue"])
