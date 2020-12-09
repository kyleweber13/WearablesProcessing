import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import Filtering
from datetime import datetime
from datetime import timedelta
import numpy as np
import ECG
import ImportEDF
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import Button
from sklearn import preprocessing

xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")


class ANNE:

    def __init__(self, subj_id=None, chest_ecg_file=None, chest_acc_file=None,
                 chest_out_vital_file=None, limb_ppg_file=None, limb_out_vital_file=None,
                 log_file=None):

        self.subj_id = subj_id

        self.chest_acc = None
        self.chest_ecg = None
        self.limb_ppg = None

        self.df_chest = None
        self.df_chest_epoch = None
        self.df_limb = None
        self.df_event = None

        self.epoch_hr = None
        self.epoch_acc = None

        self.chest_start_time = None
        self.limb_start_time = None

        self.chest_ecg_file = chest_ecg_file
        self.chest_acc_file = chest_acc_file
        self.chest_out_vital_file = chest_out_vital_file
        self.limb_ppg_file = limb_ppg_file
        self.limb_out_vital_file = limb_out_vital_file
        self.log_file = log_file

        self.chest_acc_fs = 1
        self.chest_accz_fs = 1
        self.chest_ecg_fs = 1
        self.limb_ppg_fs = 1

    def import_data(self):

        t0 = datetime.now()

        print("\nImporting data...")

        # EVENT LOG ---------------------------------------------------------------------------------------------------
        if self.log_file is not None:

            print("-Importing event log file...")

            self.df_event = pd.read_excel(self.log_file)
            self.df_event["ID"] = [str(i).split("_")[0] for i in self.df_event["ID"]]
            self.df_event = self.df_event.loc[self.df_event["ID"] == self.subj_id]

            self.df_event.columns = ["ID", "LogType", "StartDate", "StartTime", "EndDate", "EndTime", "Task",
                                     "Devices", "Comments", "Notes"]

        # =============================================== CHEST ANNE DATA =============================================

        # VITAL INFO --------------------------------------------------------------------------------------------------
        if self.chest_out_vital_file is not None:

            print("-Importing chest ANNE out_vital file...")

            self.df_chest = pd.read_csv(self.chest_out_vital_file)
            self.df_chest.columns = ["time_ms", "time_s", "epoch_ms", "hr_bpm", "hr_sqi", "ecg_leadon",
                                     "ecg_valid", "rr_rpm", "apnea_s", "rr_sqi", "accx_g", "accy_g", "accz_g",
                                     "chesttemp_c", "hr_alarm", "rr_alarm", "spo2_alarm", "chesttemp_alarm",
                                     "limbtemp_alarm", "apnea_alarm", "exception", "chest_off", "limb_off"]

            # Calculates timestamp. Accounts for daylight savings time
            unix_start = self.df_chest.iloc[0]["epoch_ms"] / 1000

            if datetime.strptime("2020-03-08", "%Y-%m-%d").date() <= datetime.utcfromtimestamp(unix_start).date() <=\
                    datetime.strptime("2020-11-01", "%Y-%m-%d").date():
                self.chest_start_time = datetime.utcfromtimestamp(unix_start) + timedelta(seconds=-4 * 3600)

            if datetime.strptime("2020-03-08", "%Y-%m-%d").date() >= datetime.utcfromtimestamp(unix_start).date() or\
                    datetime.utcfromtimestamp(unix_start).date() >= datetime.strptime("2020-11-01", "%Y-%m-%d").date():
                self.chest_start_time = datetime.utcfromtimestamp(unix_start) + timedelta(seconds=-5 * 3600)

            stop_time = self.chest_start_time + timedelta(seconds=self.df_chest.shape[0] / 5)
            self.df_chest["Timestamp"] = pd.date_range(start=self.chest_start_time, end=stop_time,
                                                       periods=self.df_chest.shape[0])

            # Converts 0's to Nones for some data
            self.df_chest["hr_bpm"] = [i if i > 0 else None for i in self.df_chest["hr_bpm"]]
            self.df_chest["rr_rpm"] = [i if i > 0 else None for i in self.df_chest["rr_rpm"]]

            # Removes redundant timestamp data
            self.df_chest.drop("time_ms", axis=1)
            self.df_chest.drop("time_s", axis=1)
            self.df_chest.drop("epoch_ms", axis=1)

        # ACCELEROMETER ----------------------------------------------------------------------------------------------
        if self.chest_acc_file is not None:

            print("-Importing chest ANNE accelerometer file...")

            self.chest_acc = pd.read_csv(self.chest_acc_file)

            # Calculates sample rate
            self.chest_accz_fs = 1000 / (self.chest_acc["time(ms)"].iloc[1] - self.chest_acc["time(ms)"].iloc[0])
            self.chest_acc_fs = self.chest_accz_fs / 8

            # Calculates timestamps
            stop_time = self.chest_start_time + timedelta(seconds=self.chest_acc.shape[0] / self.chest_accz_fs)
            self.chest_acc["Timestamp"] = pd.date_range(start=self.chest_start_time, end=stop_time,
                                                        periods=self.chest_acc.shape[0])

            self.chest_acc.drop("time(ms)", axis=1)

        # ECG ---------------------------------------------------------------------------------------------------------
        if self.chest_ecg_file is not None:

            print("-Importing chest ANNE ECG file...")

            self.chest_ecg = pd.read_csv(self.chest_ecg_file)

            # Calculates sample rate
            self.chest_ecg_fs = 1000 / (self.chest_ecg["time(ms)"].iloc[1] - self.chest_ecg["time(ms)"].iloc[0])

            # Calculates timestamps
            stop_time = self.chest_start_time + timedelta(seconds=self.chest_ecg.shape[0] / self.chest_ecg_fs)
            self.chest_ecg["Timestamp"] = pd.date_range(start=self.chest_start_time, end=stop_time,
                                                        periods=self.chest_ecg.shape[0])

            self.chest_ecg.drop("time(ms)", axis=1)

        # =============================================== LIMB ANNE DATA ==============================================

        # VITAL INFO --------------------------------------------------------------------------------------------------
        if self.limb_out_vital_file is not None:

            print("-Importing limb ANNE out_vital file...")

            self.df_limb = pd.read_csv(self.limb_out_vital_file)
            self.df_limb.columns = ["time_ms", "time_s", "epoch_ms", "spO2_perc", "pr_bpm", "pi_perc",
                                    "spo2_sqi", "ppg_attach", "ppg_valid", "limb_temp", "hr_alarm", "rr_alarm",
                                    "spo2_alarm", "chesttemp_alarm", "limbtemp_alarm", "apnea_alarm", "exception",
                                    "chest_off", "limb_off"]

            # Calculates timestamps. Accounts for daylight savings time
            unix_start = self.df_limb.iloc[0]["epoch_ms"] / 1000

            if datetime.strptime("2020-03-08", "%Y-%m-%d").date() <= datetime.utcfromtimestamp(unix_start).date() <=\
                    datetime.strptime("2020-11-01", "%Y-%m-%d").date():
                self.limb_start_time = datetime.utcfromtimestamp(unix_start) + timedelta(seconds=-4 * 3600)

            if datetime.strptime("2020-03-08", "%Y-%m-%d").date() >= datetime.utcfromtimestamp(unix_start).date() or\
                    datetime.utcfromtimestamp(unix_start).date() >= datetime.strptime("2020-11-01", "%Y-%m-%d").date():
                self.limb_start_time = datetime.utcfromtimestamp(unix_start) + timedelta(seconds=-5 * 3600)

            # Calculates timestamps
            stop_time = self.limb_start_time + timedelta(seconds=self.df_limb.shape[0] / 5)
            self.df_limb["Timestamp"] = pd.date_range(start=self.limb_start_time, end=stop_time,
                                                      periods=self.df_limb.shape[0])

            # Converts 0s to Nones for some data
            self.df_limb["pr_bpm"] = [i if i > 0 else None for i in self.df_limb["pr_bpm"]]
            self.df_limb["spO2_perc"] = [i if i > 0 else None for i in self.df_limb["spO2_perc"]]

            # Removes redundant timestamp data
            self.df_limb.drop("time_ms", axis=1)
            self.df_limb.drop("time_s", axis=1)
            self.df_limb.drop("epoch_ms", axis=1)

        # PPG ---------------------------------------------------------------------------------------------------------
        if self.limb_ppg_file is not None:

            print("-Importing limb ANNE PPG file...")

            self.limb_ppg = pd.read_csv(self.limb_ppg_file)

            # Calculates sample rate
            self.limb_ppg_fs = 1000 / (self.limb_ppg["time(ms)"].iloc[1] - self.limb_ppg["time(ms)"].iloc[0])

            # Calculates timestamps
            stop_time = self.limb_start_time + timedelta(seconds=self.limb_ppg.shape[0] / self.limb_ppg_fs)

            self.limb_ppg["Timestamp"] = pd.date_range(start=self.limb_start_time, end=stop_time,
                                                       periods=self.limb_ppg.shape[0])

            self.limb_ppg.drop("time(ms)", axis=1)

        t1 = datetime.now()
        proc_time = (t1 - t0).total_seconds()
        print("Data import complete ({} seconds)".format(round(proc_time, 1)))

    def filter_ecg_data(self, filter_type="bandpass", low_f=0.67, high_f=30):

        self.chest_ecg["ecg_filt"] = Filtering.filter_signal(data=self.chest_ecg['ecg'], filter_type=filter_type,
                                                             low_f=low_f, high_f=high_f, sample_f=self.chest_ecg_fs)

    def filter_acc_data(self, filter_type="bandpass", low_f=0.05, high_f=10):

        self.chest_acc["x_filt"] = Filtering.filter_signal(data=self.chest_acc['x'], filter_type=filter_type,
                                                           low_f=low_f, high_f=high_f, sample_f=self.chest_acc_fs)

        self.chest_acc["y_filt"] = Filtering.filter_signal(data=self.chest_acc['y'], filter_type=filter_type,
                                                           low_f=low_f, high_f=high_f, sample_f=self.chest_acc_fs)

        self.chest_acc["z_filt"] = Filtering.filter_signal(data=self.chest_acc['z'], filter_type=filter_type,
                                                           low_f=low_f, high_f=high_f, sample_f=self.chest_accz_fs)

    def plot_ecg(self, sample_rate, show_filtered=False):

        ratio = int(self.chest_ecg_fs / sample_rate)
        actual_fs = self.chest_ecg_fs / ratio

        fig, ax = plt.subplots(1, figsize=(12, 7))
        ax.set_title("ECG data ({} Hz)".format(round(actual_fs, 1)))
        ax.plot(self.chest_ecg["Timestamp"][::ratio], self.chest_ecg["ecg"][::ratio],
                color='red', label="Raw")

        if show_filtered:
            ax.plot(self.chest_ecg["Timestamp"][::ratio], self.chest_ecg["ecg_filt"][::ratio],
                    color='black', label="Filtered")

        ax.legend(loc='upper right')
        ax.set_ylabel("Voltage")
        ax.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

    def plot_acc(self, sample_rate, show_filtered=False):

        ratio_xy = int(self.chest_acc_fs / sample_rate)
        actual_fs_xy = self.chest_acc_fs / ratio_xy

        ratio_z = int(self.chest_accz_fs / sample_rate)
        actual_fs_z = self.chest_accz_fs / ratio_z

        if not show_filtered:

            fig, ax1 = plt.subplots(1, sharex='col', figsize=(12, 7))
            plt.title("Accel data (all {} Hz)".format(round(actual_fs_xy, 1)))

            ax1.plot(self.chest_acc["Timestamp"][::ratio_xy], self.chest_acc["x"][::ratio_xy],
                     color='red', label="X")
            ax1.plot(self.chest_acc["Timestamp"][::ratio_xy], self.chest_acc["y"][::ratio_xy],
                     color='black', label="Y")
            ax1.plot(self.chest_acc["Timestamp"][::ratio_z], self.chest_acc["z"][::ratio_z],
                     color='dodgerblue', label="Z")
            ax1.set_ylabel("G's")
            ax1.legend(loc='upper right')
            ax1.set_title("Raw")
            ax1.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45, fontsize=8)

        if show_filtered:

            fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(12, 7))
            plt.suptitle("Accel data (all {} Hz)".format(round(actual_fs_xy, 1)))

            ax1.plot(self.chest_acc["Timestamp"][::ratio_xy], self.chest_acc["x"][::ratio_xy],
                     color='red', label="X")
            ax1.plot(self.chest_acc["Timestamp"][::ratio_xy], self.chest_acc["y"][::ratio_xy],
                     color='black', label="Y")
            ax1.plot(self.chest_acc["Timestamp"][::ratio_z], self.chest_acc["z"][::ratio_z],
                     color='dodgerblue', label="Z")
            ax1.set_ylabel("G's")
            ax1.legend(loc='upper right')
            ax1.set_title("Raw")

            try:
                ax2.plot(self.chest_acc["Timestamp"][::ratio_xy], self.chest_acc["x_filt"][::ratio_xy],
                         color='red', label="X")
                ax2.plot(self.chest_acc["Timestamp"][::ratio_xy], self.chest_acc["y_filt"][::ratio_xy],
                         color='black', label="Y")
                ax2.plot(self.chest_acc["Timestamp"][::ratio_z], self.chest_acc["z_filt"][::ratio_z],
                         color='dodgerblue', label="Z")

                ax2.set_ylabel("G's")
                ax2.legend(loc='upper right')
                ax2.set_title("Filtered")

            except KeyError:
                print("-No filtered data found. Please filter data (ANNE.filter_acc_data()) and try again.")

            ax2.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45, fontsize=8)

    def plot_events(self):

        # Device Removal ----------------------------------------------------------------------------------------------
        nw = self.df_event.loc[self.df_event["LogType"] == "Device Removal"]

        for row in nw.itertuples():
            if row.Index != nw.index[-1]:
                plt.fill_betweenx(x1=datetime.strptime(str(row.StartDate.date()) + " " +
                                                       str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                                  x2=datetime.strptime(str(row.EndDate.date()) + " " +
                                                       str(row.EndTime), "%Y-%m-%d %H:%M:%S"),
                                  y=plt.ylim(), color='grey', alpha=.5)
            if row.Index == nw.index[-1]:
                plt.fill_betweenx(x1=datetime.strptime(str(row.StartDate.date()) + " " +
                                                       str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                                  x2=datetime.strptime(str(row.EndDate.date()) + " " +
                                                       str(row.EndTime), "%Y-%m-%d %H:%M:%S"),
                                  y=plt.ylim(), color='grey', alpha=.5, label="Nonwear")

        # Sleep -------------------------------------------------------------------------------------------------------
        sleep = self.df_event.loc[self.df_event["LogType"] == "Sleep"]

        for row in sleep.itertuples():
            if row.Index != sleep.index[-1]:
                plt.fill_betweenx(x1=datetime.strptime(str(row.StartDate.date()) + " " +
                                                       str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                                  x2=datetime.strptime(str(row.EndDate.date()) + " " +
                                                       str(row.EndTime), "%Y-%m-%d %H:%M:%S"),
                                  y=plt.ylim(), color='darkblue', alpha=.5)
            if row.Index == sleep.index[-1]:
                plt.fill_betweenx(x1=datetime.strptime(str(row.StartDate.date()) + " " +
                                                       str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                                  x2=datetime.strptime(str(row.EndDate.date()) + " " +
                                                       str(row.EndTime), "%Y-%m-%d %H:%M:%S"),
                                  y=plt.ylim(), color='darkblue', alpha=.5, label="Sleep")

        # Seated speech passage ---------------------------------------------------------------------------------------
        df = self.df_event.loc[self.df_event["LogType"] == "Daily Task"]
        df = df.loc[df["Task"] == "Seated speech passage"]

        for row in df.itertuples():
            if row.Index != df.index[-1]:
                plt.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='orange', linestyle='dashed')
            if row.Index == df.index[-1]:
                plt.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='orange', linestyle='dashed', label="Seated speech")

        # Outdoor Walk ------------------------------------------------------------------------------------------------
        df = self.df_event.loc[self.df_event["LogType"] == "Daily Task"]
        df = df.loc[df["Task"] == "Outdoor walk protocol"]

        for row in df.itertuples():
            if row.Index != df.index[-1]:
                plt.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='green', linestyle='dashed')
            if row.Index == df.index[-1]:
                plt.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='green', linestyle='dashed', label="Outdoor walk")

        # Indoor Walk ------------------------------------------------------------------------------------------------
        df = self.df_event.loc[self.df_event["LogType"] == "Daily Task"]
        df = df.loc[df["Task"] == "Indoor walk protocol"]

        for row in df.itertuples():
            if row.Index != df.index[-1]:
                plt.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='dodgerblue', linestyle='dashed')
            if row.Index == df.index[-1]:
                plt.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                            str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='dodgerblue', linestyle='dashed', label="Indoor walk")

        plt.legend(loc='upper right')

    def epoch_chest_hr(self, epoch_len=15):

        print("-Epoching ANNE chest data...")

        hr = []
        timestamp = self.df_chest["Timestamp"][::epoch_len * 5]

        for index in range(0, self.df_chest.shape[0], epoch_len * 5):

            if len([i for i in self.df_chest.iloc[index:index + epoch_len * 5]["hr_bpm"]]) >= int(epoch_len / 3):
                hr.append(np.mean([i for i in self.df_chest.iloc[index:index + epoch_len * 5]["hr_bpm"] if
                                   i is not None]))
            if len([i for i in self.df_chest.iloc[index:index + epoch_len * 5]["hr_bpm"]]) < int(epoch_len / 3):
                hr.append(None)

        df = pd.DataFrame(list(zip(timestamp, hr)), columns=["Timestamp", "hr_bpm"])

        print("Complete.")

        return df

    def epoch_chest_acc(self, epoch_len=15):

        print("-Epoching ANNE chest accelerometer data...")

        df = self.chest_acc.iloc[::8]
        df = df.reset_index()

        vm = (np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df['z'] ** 2) - 1)
        vm = [i if i >= 0 else 0 for i in vm]

        svm = []
        for i in range(0, len(vm), int(epoch_len * self.chest_acc_fs)):
            svm.append(round(sum(vm[i:i + int(epoch_len * self.chest_acc_fs)]), 2))

        timestamps = anne.chest_acc["Timestamp"].iloc[::8 * int(epoch_len * self.chest_acc_fs)]

        data = pd.DataFrame(list(zip(timestamps, svm)), columns=["Timestamp", "SVM"])

        print("Complete.")

        return data


def crop_data(bf_file):
    """Function that crops ANNE or Bittium data to align at start of collection."""

    bf_index = 0

    if bf_file is None or anne.df_chest is None:
        return 0

    bf_start, bf_end, bf_fs, bf_dur = ImportEDF.check_file(filepath=bf_file, print_summary=False)
    chest_start = anne.df_chest["Timestamp"].iloc[0]
    limb_start = anne.df_limb["Timestamp"].iloc[0]

    if bf_start < chest_start:
        print("-Bittium data file will be cropped at start to match ANNE collection period...")
        bf_index = (bf_start - chest_start).total_seconds() * bf_fs
    if bf_start > chest_start:
        print("-Cropping ANNE data to start of Bittium collection...")
        anne_chest_ecg = int((bf_start - chest_start).total_seconds() * anne.chest_ecg_fs)
        anne_chest_acc = int((bf_start - chest_start).total_seconds() * anne.chest_accz_fs)
        anne_chest_vital = int((bf_start - chest_start).total_seconds() * 5)

        anne.chest_ecg = anne.chest_ecg.loc[anne.chest_ecg["Timestamp"] >= bf_start]
        anne.chest_acc = anne.chest_acc.loc[anne.chest_acc["Timestamp"] >= bf_start]

    return bf_index


# =========================================================== SET UP ==================================================

bittium_file = "C:/Users/ksweber/Desktop/007_ANNEValidation/007_Stingray.EDF"

anne = ANNE(subj_id="007",
            chest_acc_file="C:/Users/ksweber/Desktop/007_ANNEValidation/ChestC1515_accl.csv",
            chest_ecg_file="C:/Users/ksweber/Desktop/007_ANNEValidation/ChestC1515_ecg.csv",
            chest_out_vital_file="C:/Users/ksweber/Desktop/007_ANNEValidation/ChestC1515_out_vital.csv",
            limb_ppg_file="C:/Users/ksweber/Desktop/007_ANNEValidation/Limb1307_ppg.csv",
            limb_out_vital_file="C:/Users/ksweber/Desktop/007_ANNEValidation/Limb1307_out_vital.csv",
            log_file="C:/Users/ksweber/Desktop/ANNE_Validation_Logs.xlsx")
anne.import_data()

bittium_offset = crop_data(bf_file=bittium_file)
bf = None

bf = ECG.ECG(subject_id=anne.subj_id, filepath=bittium_file,
             output_dir=None, processed_folder=None,
             processed_file=None, ecg_downsample=1,
             age=26, start_offset=bittium_offset if bittium_offset is not None else 0, end_offset=0,
             rest_hr_window=60, n_epochs_rest=30,
             epoch_len=15, load_accel=True,
             filter_data=False, low_f=1, high_f=30, f_type="bandpass",
             load_raw=True, from_processed=False)


anne.epoch_hr = anne.epoch_chest_hr(epoch_len=15)
anne.epoch_acc = anne.epoch_chest_acc(epoch_len=15)

# Filters ecg data
anne.filter_ecg_data(filter_type='bandpass', low_f=.67, high_f=25)

# Plots ecg data. Able to plot just raw or raw and filtered. Able to adjust sample rate.
# anne.plot_ecg(show_filtered=True, sample_rate=125)

# Filters chest accelerometer data
# anne.filter_acc_data(filter_type='bandpass', low_f=0.05, high_f=10)

# Plots accel data. Able to plot just raw or raw and filtered. Able to adjust sample rate.
# anne.plot_acc(sample_rate=25, show_filtered=True)


def epoch_hr_blandaltman():

    means = []
    diffs = []
    for b, a in zip(bf.valid_hr, anne.epoch_hr["hr_bpm"]):
        if b is not None and not np.isnan(a):
            means.append((b+a)/2)
            diffs.append(b-a)

    loa = np.std(diffs) * 1.96
    bias = np.mean(diffs)

    fig, ax = plt.subplots(1, figsize=(12, 7))
    ax.scatter(x=means, y=diffs, color='black', s=5)

    ax.axhline(bias + loa, color='red', linestyle='dashed', label='Upper LOA ({}bpm)'.format(round(bias+loa, 1)))
    ax.axhline(bias, color='black', linestyle='dashed', label='Bias ({}bpm)'.format(round(bias, 1)))
    ax.axhline(bias - loa, color='red', linestyle='dashed', label='Lower LOA ({}bpm)'.format(round(bias-loa, 1)))

    ax.fill_between(x=[min(means), max(means)], y1=plt.ylim()[0], y2=bias - loa, color='red', alpha=.25)
    ax.fill_between(x=[min(means), max(means)], y1=bias + loa, y2=plt.ylim()[1], color='red', alpha=.25)
    ax.fill_between(x=[min(means), max(means)], y1=bias - loa, y2=bias + loa, color='green', alpha=.25)

    ax.set_xlabel("Mean HR")
    ax.set_ylabel("Difference (BF - ANNE)")
    plt.legend()

    perc_same = len([i for i in diffs if (bias - loa) <= i <= (bias + loa)]) / len(diffs) * 100
    ax.set_title("Bland-Altman Comparison ({}% agreement)".format(round(perc_same, 1)))


# ==================================================== DATA VISUALIZATION =============================================

"""Adds events to an open plot"""
# anne.plot_events()

"""Compares 15-second averaged HR between chest ANNE and Bittium Faros"""
# epoch_hr_blandaltman()


class DataViewer:

    def __init__(self, anne_obj=None, bf_obj=None, fig_width=10, fig_height=6):

        self.anne = anne_obj
        self.bf = bf_obj

        self.fig_width = fig_width
        self.fig_height = fig_height

        if self.bf is not None:
            self.bf.accel_x = [i / 1000 for i in self.bf.accel_x]
            self.bf.accel_y = [i / 1000 for i in self.bf.accel_y]
            self.bf.accel_z = [i / 1000 for i in self.bf.accel_z]
            self.bf.accel_vm = [i / 1000 for i in self.bf.accel_vm]

        self.hr_plot_dict = {"ANNE Chest": False, "ANNE Limb": False, "BittiumFaros": False}
        self.hr_data_dict = {"HR": False, "Raw": False, "Filt.": False}
        self.accel_plot_dict = {"ANNE Chest": False, "ANNE Limb": False, "BittiumFaros": False,
                                "WristGA": False, "AnkleGA": False}
        self.accel_axis_dict = {"x": False, "y": False, "z": False, "SVM": False}
        self.temp_plot_dict = {"ANNE Chest": False, "ANNE Limb": False, "WristGA": False, "AnnkleGA": False}
        self.misc_plot_dict = {"ANNE Limb ppg": False, "ANNE Limb sO2": False,
                               "ANNE Chest Resp.": False, "ECG Validity": False}
        self.show_events = False

        self.check1 = None
        self.check1_data = None
        self.check2 = None
        self.check2axis = None
        self.check3 = None
        self.check4 = None
        self.reload_button = None
        self.raw_button = None
        self.cooked_button = None
        self.events_button = None
        self.help_button = None

    def plot_data(self):

        plt.close("all")
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(self.fig_width, self.fig_height))
        plt.subplots_adjust(right=.8, hspace=.28)
        ax4.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        """============================================== Heart Rate ==============================================="""
        ax1.set_title("Heart Rate Data")

        if not self.hr_data_dict["Raw"] and not self.hr_data_dict["Filt."] and self.hr_data_dict["HR"]:
            ax1.set_ylabel("bpm")
        if (self.hr_data_dict["Raw"] or self.hr_data_dict["Filt."]) and not self.hr_data_dict["HR"]:
            ax1.set_ylabel("Voltage (scaled)")

        # ANNE Chest
        if self.hr_plot_dict["ANNE Chest"]:
            if self.hr_data_dict["HR"]:
                ax1.plot(self.anne.df_chest["Timestamp"], self.anne.df_chest["hr_bpm"],
                         color='red', label='ANNE Chest')
            if self.hr_data_dict["Raw"]:
                ax1.plot(self.anne.chest_ecg["Timestamp"][::4],
                         preprocessing.scale(self.anne.chest_ecg["ecg"][::4]),
                         color='red', label='Chest ANNE')
            if self.hr_data_dict["Filt."]:
                ax1.plot(self.anne.chest_ecg["Timestamp"][::4],
                         preprocessing.scale(self.anne.chest_ecg["ecg_filt"][::4]),
                         color='red', label='Chest ANNE')

        # ANNE Limb
        if self.hr_plot_dict["ANNE Limb"] and self.hr_data_dict["HR"]:
            ax1.plot(self.anne.df_limb["Timestamp"], self.anne.df_limb["pr_bpm"],
                     color='dodgerblue', label='ANNE Limb')

        # Bittium Faros
        if self.hr_plot_dict["BittiumFaros"]:
            if self.hr_data_dict["HR"]:
                ax1.plot(self.bf.epoch_timestamps, self.bf.valid_hr, color='black', label='BittiumFaros')
            if self.hr_data_dict["Raw"]:
                ax1.plot(self.bf.timestamps[::2], preprocessing.scale(self.bf.raw[::2]),
                         color='black', label="BittiumFaros")
            if self.hr_data_dict["Filt."]:
                ax1.plot(self.bf.timestamps[::2], preprocessing.scale(self.bf.filtered[::2]),
                         color='black', label="BittiumFaros")

        if True in self.hr_plot_dict.values():
            ax1.legend(loc='upper left')

        # rax1 = plt.axes([.81, .72, .18, .16])
        rax1 = plt.axes([.81, .72, .18, .16])
        rax1_2 = plt.axes([.935, .72, .055, .16])
        self.check1 = CheckButtons(rax1, ("ANNE Chest", "ANNE Limb", "BittiumFaros"),
                                   (False, False, False))
        self.check1_data = CheckButtons(rax1_2, ("HR", "Raw", "Filt."), (False, False, False))

        """============================================= Accelerometer ============================================="""
        ax2.set_title("Accelerometer Data")

        # Colors based on how much data is being plotted
        if [i for i in self.accel_plot_dict.values()].count(True) == 1:
            colors = ['red', 'dodgerblue', 'black']
        if [i for i in self.accel_plot_dict.values()].count(True) > 1 and \
                [i for i in self.accel_axis_dict.values()].count(True) > 1:
            print("\n-Accelerometer data is going to be a mess...")

        # ANNE Chest
        if self.accel_plot_dict["ANNE Chest"] and self.anne.chest_ecg_file is not None:
            if [i for i in self.accel_plot_dict.values()].count(True) > 1:
                colors = ['red', 'red', 'red']

            if self.accel_axis_dict["x"]:
                ax2.plot(self.anne.chest_acc["Timestamp"][::2], self.anne.chest_acc["x"][::2],
                         label="ANNE_x", color=colors[0])
            if self.accel_axis_dict["y"]:
                ax2.plot(self.anne.chest_acc["Timestamp"][::2], self.anne.chest_acc["y"][::2],
                         label="ANNE_y", color=colors[1])
            if self.accel_axis_dict["z"]:
                ax2.plot(self.anne.chest_acc["Timestamp"][::16], self.anne.chest_acc["z"][::16],
                         label="ANNE_z", color=colors[2])

            if self.accel_axis_dict["SVM"]:
                ax2.plot(self.anne.epoch_acc["Timestamp"], self.anne.epoch_acc["SVM"], label="ANNE Chest", color='red')

        """if self.accel_plot_dict["ANNE Limb"]:
            pass"""

        # Bittium Faros
        if self.accel_plot_dict["BittiumFaros"]:
            if [i for i in self.accel_plot_dict.values()].count(True) > 1:
                colors = ['black', 'black', 'black']

            ratio = int(self.bf.sample_rate / self.bf.accel_sample_rate)

            if self.accel_axis_dict["x"]:
                ax2.plot(self.bf.timestamps[::ratio], self.bf.accel_x, label="BF_x", color=colors[0])
            if self.accel_axis_dict["y"]:
                ax2.plot(self.bf.timestamps[::ratio], self.bf.accel_y, label="BF_y", color=colors[1])
            if self.accel_axis_dict["z"]:
                ax2.plot(self.bf.timestamps[::ratio], self.bf.accel_z, label="BF_z", color=colors[2])

            if self.accel_axis_dict["SVM"]:
                ax2.plot(self.bf.epoch_timestamps[:len(self.bf.svm)], self.bf.svm, label="BF_SVM", color='black')

        """Placeholder for Wrist GA"""

        """Placeholder for Ankle GA"""

        if True in self.accel_plot_dict.values():
            ax2.legend(loc='upper left')

        rax2 = plt.axes([.81, .5175, .115, .16])
        rax2_2 = plt.axes([.925, .5175, .065, .16])

        self.check2 = CheckButtons(rax2, ("ANNE Chest", "ANNE Limb", "BittiumFaros", "WristGA", "AnkleGA"),
                                   (False, False, False, False, False))
        self.check2axis = CheckButtons(rax2_2, ("x", "y", "z", "SVM"), (False, False, False, False))

        """============================================= Temperature ==============================================="""
        ax3.set_title("Temperature Data")
        ax3.set_ylabel("Celcius")

        rax3 = plt.axes([.81, .3125, .18, .16])

        self.check3 = CheckButtons(rax3, ("ANNE Chest", "ANNE Limb", "WristGA", "AnkleGA"),
                                   (False, False, False, False, False))

        if self.temp_plot_dict["ANNE Chest"]:
            ax3.plot(self.anne.df_chest["Timestamp"], self.anne.df_chest["chesttemp_c"],
                     color='red', label="Chest ANNE")

        if self.temp_plot_dict["ANNE Limb"]:
            ax3.plot(self.anne.df_limb["Timestamp"], self.anne.df_limb["limb_temp"],
                     color='dodgerblue', label="Limb ANNE")

        if True in self.temp_plot_dict.values():
            ax3.legend(loc='upper left')

        """============================================= Miscellaneous ============================================="""
        ax4.set_title("Miscellaneous Data")

        rax4 = plt.axes([.81, .11, .18, .16])
        self.check4 = CheckButtons(rax4, ("ANNE Limb ppg", "ANNE Limb sO2", "ANNE Chest Resp.", "ECG Validity"),
                                   (False, False, False, False))

        if self.misc_plot_dict["ANNE Limb ppg"]:
            ax4.plot(self.anne.limb_ppg["Timestamp"][::2], self.anne.limb_ppg["red"][::2],
                     color='red', label='Red light')
            ax4.plot(self.anne.limb_ppg["Timestamp"][::2], self.anne.limb_ppg["ir"][::2],
                     color='grey', label='IR light')

        if self.misc_plot_dict["ANNE Limb sO2"]:
            ax4.plot(self.anne.df_limb["Timestamp"], self.anne.df_limb["spO2_perc"],
                     color='dodgerblue', label='ANNE Limb')
            ax4.set_ylim(0, 100)
            ax4.set_ylabel("Percent")

        if self.misc_plot_dict["ANNE Chest Resp."]:
            ax4.plot(self.anne.df_chest["Timestamp"], self.anne.df_chest["rr_rpm"], color='red', label="ANNE Chest")
            ax4.set_ylabel("Breaths/min")

        if self.misc_plot_dict["ECG Validity"]:
            ax4.plot(self.bf.epoch_timestamps[:len(self.bf.epoch_validity)], self.bf.epoch_validity,
                     color='black', label='BittiumFaros')

            anne_data = ["Valid" if not np.isnan(i) else "Invalid" for i in self.anne.df_chest["hr_bpm"]]
            ax4.plot(self.anne.df_chest["Timestamp"], anne_data, linestyle='dashed', color="red", label="ANNE Chest")

        if True in self.misc_plot_dict.values():
            ax4.legend(loc='upper left')

        """============================================= Event Plotting ============================================"""
        if self.show_events:

            plt.suptitle("Check Python console for event colour coding.")

            print("\nEvent colour code:")
            print("-Grey shaded = nonwear (any)")
            print("-Dark blue shaded = sleep")
            print("-Orange line = seated speech passage")
            print("-Green line = outdoor walk")
            print("-Purple = indoor walk")

            df_event = self.anne.df_event

            # Device Removal
            nw = df_event.loc[df_event["LogType"] == "Device Removal"]

            for row in nw.itertuples():
                ax1.fill_betweenx(x1=datetime.strptime(str(row.StartDate.date()) + " " +
                                                       str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                                  x2=datetime.strptime(str(row.EndDate.date()) + " " +
                                                       str(row.EndTime), "%Y-%m-%d %H:%M:%S"),
                                  y=[0, 200], color='grey', alpha=.5)
                ax2.fill_betweenx(x1=datetime.strptime(str(row.StartDate.date()) + " " +
                                                       str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                                  x2=datetime.strptime(str(row.EndDate.date()) + " " +
                                                       str(row.EndTime), "%Y-%m-%d %H:%M:%S"),
                                  y=ax2.get_ylim(), color='grey', alpha=.5)
                ax3.fill_betweenx(x1=datetime.strptime(str(row.StartDate.date()) + " " +
                                                       str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                                  x2=datetime.strptime(str(row.EndDate.date()) + " " +
                                                       str(row.EndTime), "%Y-%m-%d %H:%M:%S"),
                                  y=ax3.get_ylim(), color='grey', alpha=.5)
                ax4.fill_betweenx(x1=datetime.strptime(str(row.StartDate.date()) + " " +
                                                       str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                                  x2=datetime.strptime(str(row.EndDate.date()) + " " +
                                                       str(row.EndTime), "%Y-%m-%d %H:%M:%S"),
                                  y=ax4.get_ylim(), color='grey', alpha=.5)

            # Sleep
            sleep = df_event.loc[df_event["LogType"] == "Sleep"]

            for row in sleep.itertuples():
                ax1.fill_betweenx(x1=datetime.strptime(str(row.StartDate.date()) + " " +
                                                       str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                                  x2=datetime.strptime(str(row.EndDate.date()) + " " +
                                                       str(row.EndTime), "%Y-%m-%d %H:%M:%S"),
                                  y=[0, 200], color='darkblue', alpha=.25)
                ax2.fill_betweenx(x1=datetime.strptime(str(row.StartDate.date()) + " " +
                                                       str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                                  x2=datetime.strptime(str(row.EndDate.date()) + " " +
                                                       str(row.EndTime), "%Y-%m-%d %H:%M:%S"),
                                  y=ax2.get_ylim(), color='darkblue', alpha=.25)
                ax3.fill_betweenx(x1=datetime.strptime(str(row.StartDate.date()) + " " +
                                                       str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                                  x2=datetime.strptime(str(row.EndDate.date()) + " " +
                                                       str(row.EndTime), "%Y-%m-%d %H:%M:%S"),
                                  y=ax3.get_ylim(), color='darkblue', alpha=.25)
                ax4.fill_betweenx(x1=datetime.strptime(str(row.StartDate.date()) + " " +
                                                       str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                                  x2=datetime.strptime(str(row.EndDate.date()) + " " +
                                                       str(row.EndTime), "%Y-%m-%d %H:%M:%S"),
                                  y=ax4.get_ylim(), color='darkblue', alpha=.25)

            # Seated speech passage
            df = df_event.loc[df_event["LogType"] == "Daily Task"]
            df = df.loc[df["Task"] == "Seated speech passage"]

            for row in df.itertuples():
                ax1.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='orange', linestyle='dashed')
                ax2.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='orange', linestyle='dashed')
                ax3.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='orange', linestyle='dashed')
                ax4.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='orange', linestyle='dashed')

            # Outdoor Walk
            df = df_event.loc[df_event["LogType"] == "Daily Task"]
            df = df.loc[df["Task"] == "Outdoor walk protocol"]

            for row in df.itertuples():
                ax1.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='green', linestyle='dashed')
                ax2.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='green', linestyle='dashed')
                ax3.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='green', linestyle='dashed')
                ax4.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='green', linestyle='dashed')
            # Indoor Walk
            df = df_event.loc[df_event["LogType"] == "Daily Task"]
            df = df.loc[df["Task"] == "Indoor walk protocol"]

            for row in df.itertuples():
                ax1.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='mediumorchid', linestyle='dashed')
                ax2.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='mediumorchid', linestyle='dashed')
                ax3.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='mediumorchid', linestyle='dashed')
                ax4.axvline(x=datetime.strptime(str(row.StartDate.date()) + " " +
                                                str(row.StartTime), "%Y-%m-%d %H:%M:%S"),
                            color='mediumorchid', linestyle='dashed')

        ax4.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        """================================================ Buttons ==============================================="""

        rax_help = plt.axes([0.84, .01 + .043, 0.15, .043])
        self.help_button = Button(rax_help, 'Print descriptions', color='crimson')
        self.help_button.on_clicked(self.print_desc)

        rax_reload = plt.axes([0.917, 0.005, 0.074, .042])
        self.reload_button = Button(rax_reload, 'Reload', color='chartreuse')
        self.reload_button.on_clicked(self.get_values)

        rax_events = plt.axes([0.84, 0.005, 0.074, .043])
        self.events_button = Button(rax_events, 'Show/Hide\nEvents', color='mediumturquoise')
        self.events_button.on_clicked(self.set_events)

    def get_values(self, event):
        print("\nReloading...")

        ax1_vals = self.check1.get_status()
        self.hr_plot_dict.update(zip([i for i in self.hr_plot_dict.keys()], ax1_vals))

        ax1_vals_data = self.check1_data.get_status()
        self.hr_data_dict.update(zip([i for i in self.hr_data_dict.keys()], ax1_vals_data))

        ax2_vals = self.check2.get_status()
        self.accel_plot_dict.update(zip([i for i in self.accel_plot_dict.keys()], ax2_vals))

        ax2_vals_axis = self.check2axis.get_status()
        self.accel_axis_dict.update(zip([i for i in self.accel_axis_dict.keys()], ax2_vals_axis))

        ax3_vals = self.check3.get_status()
        self.temp_plot_dict.update(zip([i for i in self.temp_plot_dict.keys()], ax3_vals))

        ax4_vals = self.check4.get_status()
        self.misc_plot_dict.update(zip([i for i in self.misc_plot_dict.keys()], ax4_vals))

        self.plot_data()

    def set_events(self, event):
        self.show_events = not self.show_events

        self.plot_data()

    def reset_plot(self):

        self.hr_plot_dict = {"ANNE Chest": False, "ANNE Limb": False, "BittiumFaros": False}
        self.accel_plot_dict = {"ANNE Chest": False, "ANNE Limb": False, "BittiumFaros": False,
                                "WristGA": False, "AnkleGA": False}
        self.accel_axis_dict = {"x": False, "y": False, "z": False, "SVM": False}
        self.temp_plot_dict = {"ANNE Chest": False, "ANNE Limb": False, "WristGA": False, "AnnkleGA": False}
        self.misc_plot_dict = {"ANNE Limb ppg": False, "ANNE Limb sO2": False,
                               "ANNE Chest Resp.": False, "ECG Validity": False}

        self.show_events = False

    @staticmethod
    def print_desc(event):

        print("\n========================= PLOT OPERATION AND DESCRIPTION ========================= ")

        print("\nOPERATION:")
        print("-Check off the box for all the data you wish to see. Then click the 'reload' button.")
        print("-To show or hide the events from the ANNE validation protocol, click the 'Show/Hide Events' button.")

        print("\nDATA DESCRIPTIONS:")

        print("Axis #1:")
        print("-Left column:")
        print("     -ANNE Chest: show ANNE chest data.")
        print("     -ANNE Limb: show ANNE limb data.")
        print("     -BittiumFaros: show Bittium Faros data.")
        print("-Right column:")
        print("     -HR: shows HR.")
        print("          -ANNE Chest: 200ms intervals derived from ECG.")
        print("          -ANNE Limb: 200ms intervals derived from PPG.")
        print("          -BittiumFaros: 15s intervals derived from ECG.")

        print("     -Raw: shows raw ECG data. Scaled so mean = 0 and SD = 1.")
        print("     -Filt.: shows filtered ECG data. Scaled so mean = 0 and SD = 1.")

        print("\nAxis #2:")
        print("-Left column")
        print("     -ANNE Chest: raw accelerometer data from chest ANNE. Downsampled by a factor of 2.")
        print("     -ANNE Limb: temporary placeholder until we figure out where this data is...")
        print("     -BittiumFaros: raw accelerometer data from Bittium Faros. 25Hz.")
        print("     -WristGA: temporary placeholder.")
        print("     -AnkleGA: temporary placeholder.")
        print("-Right column:")
        print("     -x/y/z: which axis/axes to plot.")
        print("          -If viewing multiple devices, it looks stupid if you pick multiple axes.")
        print("          -Stick to multi-axial on one device or uniaxial on multiple devices.")
        print("     -SVM: sum of vector magnitudes in 15-second epochs.")

        print("\nAxis #3:")
        print("-ANNE Chest: temperature data in 200ms intervals from chest ANNE.")
        print("-ANNE Limb: temperature data in 200ms intervals from limb ANNE.")
        print("-WristGA: temporary placeholder.")
        print("-AnkleGA: temporary placeholder.")

        print("\nAxis #4:")
        print("-ANNE Limb ppg: both light frequencies (red, ir) from ANNE limb PPG. Downsampled by a factor of 2.")
        print("-ANNE Limb sO2: percent oxygen saturation in 200ms intervals from limb ANNE.")
        print("-ANNE Chest Resp.: respiration rate in 200ms intervals from chest ANNE.")
        print("-ECG Validity: ECG signal quality from chest ANNE and Bittium Faros.")
        print("     -ANNE Chest: unknown algorithm, 200ms intervals.")
        print("     -Bittium Faros: modified Orphanidou et al. (2015) algorithm, 15s intervals.")


test = DataViewer(anne_obj=anne, bf_obj=bf, fig_width=12, fig_height=9)
test.plot_data()
test.reset_plot()

# TODO
# Add raw ECG to ax1. Add description to print_desc()
# Something is up with ANNE chest accel sample rate --> data "drifts" through collection
# add bland-altman with ability to change data
