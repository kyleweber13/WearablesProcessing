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
import pyedflib

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
        self.df_limb = None

        self.df_event = None

        self.epoch_hr = None
        self.epoch_acc = None
        self.epoch_limb = None

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

            if "csv" in self.chest_out_vital_file or "CSV" in self.chest_out_vital_file:

                self.df_chest = pd.read_csv(self.chest_out_vital_file)
                self.df_chest.columns = ["time_ms", "time_s", "epoch_ms", "hr_bpm", "hr_sqi", "ecg_leadon",
                                         "ecg_valid", "rr_rpm", "apnea_s", "rr_sqi", "accx_g", "accy_g", "accz_g",
                                         "chesttemp_c", "hr_alarm", "rr_alarm", "spo2_alarm", "chesttemp_alarm",
                                         "limbtemp_alarm", "apnea_alarm", "exception", "chest_off", "limb_off"]

                # Calculates timestamp. Accounts for daylight savings time
                unix_start = self.df_chest.iloc[0]["epoch_ms"] / 1000

                if datetime.strptime("2020-03-08", "%Y-%m-%d").date() <= \
                        datetime.utcfromtimestamp(unix_start).date() <= \
                        datetime.strptime("2020-11-01", "%Y-%m-%d").date():
                    self.chest_start_time = datetime.utcfromtimestamp(unix_start) + timedelta(seconds=-4 * 3600)

                if datetime.strptime("2020-03-08", "%Y-%m-%d").date() >= \
                        datetime.utcfromtimestamp(unix_start).date() \
                        or \
                        datetime.utcfromtimestamp(unix_start).date() >= \
                        datetime.strptime("2020-11-01", "%Y-%m-%d").date():
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

            if "edf" in self.chest_out_vital_file or "EDF" in self.chest_out_vital_file:
                file = pyedflib.EdfReader(self.chest_out_vital_file)

                self.df_chest = pd.DataFrame(columns=[i for i in file.getSignalLabels()])
                for chn, col_name in enumerate(file.getSignalLabels()):
                    self.df_chest[col_name] = file.readSignal(chn)

                self.chest_start_time = file.getStartdatetime()
                stop_time = self.chest_start_time + timedelta(seconds=file.getFileDuration())
                self.df_chest["Timestamp"] = pd.date_range(start=self.chest_start_time, end=stop_time,
                                                           periods=self.df_chest.shape[0])

                self.df_chest["hr_bpm"] = [i if i > 1 else None for i in self.df_chest["hr_bpm"]]
                self.df_chest["rr_rpm"] = [i if i > 1 else None for i in self.df_chest["rr_rpm"]]

        # ACCELEROMETER ----------------------------------------------------------------------------------------------
        if self.chest_acc_file is not None:

            print("-Importing chest ANNE accelerometer file...")

            if "csv" in self.chest_acc_file or "CSV" in self.chest_acc_file:
                self.chest_acc = pd.read_csv(self.chest_acc_file)

                # Calculates sample rate

                # Timestamp calculation
                # self.chest_accz_fs = 1000 / (self.chest_acc["time(ms)"].iloc[2] - self.chest_acc["time(ms)"].iloc[1])

                # Manually-calculated via spike signal
                # self.chest_accz_fs = 412.180729

                # Stated sample rate
                self.chest_accz_fs = 416

                self.chest_acc_fs = self.chest_accz_fs / 8

                # Calculates timestamps
                stop_time = self.chest_start_time + timedelta(seconds=self.chest_acc.shape[0] / self.chest_accz_fs)
                self.chest_acc["Timestamp"] = pd.date_range(start=self.chest_start_time, end=stop_time,
                                                            periods=self.chest_acc.shape[0])

                self.chest_acc.drop("time(ms)", axis=1)

            if "edf" in self.chest_acc_file or "EDF" in self.chest_acc_file:
                file = pyedflib.EdfReader(self.chest_acc_file)

                self.chest_acc = pd.DataFrame(columns=[i for i in file.getSignalLabels()])
                for chn, col_name in enumerate(file.getSignalLabels()):
                    self.chest_acc[col_name] = file.readSignal(chn)

                stop_time = self.chest_start_time + timedelta(seconds=file.getFileDuration())
                self.chest_acc["Timestamp"] = pd.date_range(start=self.chest_start_time, end=stop_time,
                                                            periods=self.chest_acc.shape[0])

        # ECG ---------------------------------------------------------------------------------------------------------
        if self.chest_ecg_file is not None:

            print("-Importing chest ANNE ECG file...")

            if "csv" in self.chest_ecg_file or "CSV" in self.chest_ecg_file:

                self.chest_ecg = pd.read_csv(self.chest_ecg_file)

                # Calculates sample rate
                # self.chest_ecg_fs = 1000 / (self.chest_ecg["time(ms)"].iloc[1] - self.chest_ecg["time(ms)"].iloc[0])
                self.chest_ecg_fs = 512

                # Calculates timestamps
                stop_time = self.chest_start_time + timedelta(seconds=self.chest_ecg.shape[0] / self.chest_ecg_fs)
                self.chest_ecg["Timestamp"] = pd.date_range(start=self.chest_start_time, end=stop_time,
                                                            periods=self.chest_ecg.shape[0])

                self.chest_ecg.drop("time(ms)", axis=1)

            if "edf" in self.chest_ecg_file or "EDF" in self.chest_ecg_file:
                file = pyedflib.EdfReader(self.chest_ecg_file)

                self.chest_ecg = pd.DataFrame(columns=[i for i in file.getSignalLabels()])
                for chn, col_name in enumerate(file.getSignalLabels()):
                    self.chest_ecg[col_name] = file.readSignal(chn)

                self.chest_ecg_fs = 512

                stop_time = self.chest_start_time + timedelta(seconds=file.getFileDuration())
                self.chest_ecg["Timestamp"] = pd.date_range(start=self.chest_start_time, end=stop_time,
                                                            periods=self.chest_ecg.shape[0])

            self.filter_ecg_data(filter_type="bandpass", low_f=.67, high_f=25)

        # =============================================== LIMB ANNE DATA ==============================================

        # VITAL INFO --------------------------------------------------------------------------------------------------
        if self.limb_out_vital_file is not None:

            print("-Importing limb ANNE out_vital file...")

            if "csv" in self.limb_out_vital_file or "CSV" in self.limb_out_vital_file:

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

                # Converts 0s to Nones for some data
                self.df_limb["pr_bpm"] = [i if i > 0 else None for i in self.df_limb["pr_bpm"]]
                self.df_limb["spO2_perc"] = [i if i > 0 else None for i in self.df_limb["spO2_perc"]]

                # Removes redundant timestamp data
                self.df_limb.drop("time_ms", axis=1)
                self.df_limb.drop("time_s", axis=1)
                self.df_limb.drop("epoch_ms", axis=1)

            if "edf" in self.limb_out_vital_file or "EDF" in self.limb_out_vital_file:
                file = pyedflib.EdfReader(self.limb_out_vital_file)

                self.df_limb = pd.DataFrame(columns=[i for i in file.getSignalLabels()])
                for chn, col_name in enumerate(file.getSignalLabels()):
                    self.df_limb[col_name] = file.readSignal(chn)

                self.limb_start_time = file.getStartdatetime()

                self.df_limb["pr_bpm"] = [i if i > 1 else None for i in self.df_limb["pr_bpm"]]
                self.df_limb["spO2_perc"] = [i if i > 1 else None for i in self.df_limb["spO2_perc"]]

            # Calculates timestamps
            stop_time = self.limb_start_time + timedelta(seconds=self.df_limb.shape[0] / 5)
            self.df_limb["Timestamp"] = pd.date_range(start=self.limb_start_time, end=stop_time,
                                                      periods=self.df_limb.shape[0])

        # PPG ---------------------------------------------------------------------------------------------------------
        if self.limb_ppg_file is not None:

            print("-Importing limb ANNE PPG file...")

            if "csv" in self.limb_ppg_file or "CSV" in self.limb_ppg_file:
                self.limb_ppg = pd.read_csv(self.limb_ppg_file)

                # Calculates sample rate
                # self.limb_ppg_fs = 1000 / (self.limb_ppg["time(ms)"].iloc[1] - self.limb_ppg["time(ms)"].iloc[0])
                self.limb_ppg_fs = 128

                self.limb_ppg.drop("time(ms)", axis=1)

            if "edf" in self.limb_ppg_file or "EDF" in self.limb_ppg_file:
                file = pyedflib.EdfReader(self.limb_ppg_file)

                self.limb_ppg = pd.DataFrame(columns=[i for i in file.getSignalLabels()])
                for chn, col_name in enumerate(file.getSignalLabels()):
                    self.limb_ppg[col_name] = file.readSignal(chn)

            # Calculates timestamps
            stop_time = self.limb_start_time + timedelta(seconds=self.limb_ppg.shape[0] / self.limb_ppg_fs)

            self.limb_ppg["Timestamp"] = pd.date_range(start=self.limb_start_time, end=stop_time,
                                                       periods=self.limb_ppg.shape[0])

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

            # Requires at least 1/3 of data points in epoch to be valid data else None
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

        # timestamps = anne.chest_acc["Timestamp"].iloc[::8 * int(epoch_len * self.chest_acc_fs)]
        timestamps = df["Timestamp"][::int(epoch_len * self.chest_acc_fs)]

        data = pd.DataFrame(list(zip(timestamps, svm)), columns=["Timestamp", "SVM"])

        print("Complete.")

        return data

    def epoch_limb_data(self, epoch_len=15):
        """Epochs limb ANNE pulse rate and oxygen saturation data.
           If data available, crops epochs to match chest ANNE data.
        """

        print("-Epoching ANNE limb data...")

        if self.epoch_hr is not None:
            epoch1 = self.epoch_hr.loc[self.epoch_hr["Timestamp"] >=
                                       self.df_limb["Timestamp"].iloc[0]].iloc[0]["Timestamp"]
        if anne.epoch_hr is None:
            epoch1 = self.df_limb.iloc[0]["Timestamp"]

        df_limb_crop = self.df_limb.loc[self.df_limb["Timestamp"] >= epoch1]

        pr = []
        spo2 = []
        timestamp = self.df_limb["Timestamp"][::epoch_len * 5]

        for index in range(0, self.df_limb.shape[0], epoch_len * 5):

            # Requires at least 1/3 of data points in epoch to be valid data else None
            if len([i for i in self.df_limb.iloc[index:index + epoch_len * 5]["pr_bpm"]]) >= int(epoch_len / 3):
                pr.append(np.mean([i for i in self.df_limb.iloc[index:index + epoch_len * 5]["pr_bpm"] if
                                   i is not None]))
            if len([i for i in self.df_limb.iloc[index:index + epoch_len * 5]["pr_bpm"]]) < int(epoch_len / 3):
                pr.append(None)

            if len([i for i in self.df_limb.iloc[index:index + epoch_len * 5]["spO2_perc"]]) >= int(epoch_len / 3):
                spo2.append(np.mean([i for i in self.df_limb.iloc[index:index + epoch_len * 5]["spO2_perc"] if
                                     i is not None]))
            if len([i for i in self.df_limb.iloc[index:index + epoch_len * 5]["spO2_perc"]]) < int(epoch_len / 3):
                spo2.append(None)

        df = pd.DataFrame(list(zip(timestamp, pr, spo2)), columns=["Timestamp", "pr_bpm", "spO2_perc"])

        print("Complete.")

        return df

    def write_chestvitalout_edf(self):
        """Writes chest_out_vital_file to edf"""

        print("-Writing {} to edf format...".format(self.chest_out_vital_file))

        channel_names = ["hr_bpm", "hr_sqi", "ecg_leadon", "ecg_valid", "rr_rpm", "apnea_s", "rr_sqi", "accx_g",
                         "accy_g",
                         "accz_g", "chesttemp_c", "hr_alarm", "rr_alarm", "spo2_alarm", "chesttemp_alarm",
                         "limbtemp_alarm",
                         "apnea_alarm", "exception", "chest_off", "limb_off"]

        self.df_chest["hr_bpm"] = [i if not np.isnan(i) else 0 for i in self.df_chest["hr_bpm"]]
        self.df_chest["rr_rpm"] = [i if not np.isnan(i) else 0 for i in self.df_chest["rr_rpm"]]

        data = np.array(self.df_chest[channel_names]).transpose()
        signal_headers = pyedflib.highlevel.make_signal_headers(channel_names, sample_rate=5)
        header = pyedflib.highlevel.make_header(startdate=self.df_chest.iloc[0]["Timestamp"])
        pyedflib.highlevel.write_edf(self.chest_out_vital_file.split(".")[0] + ".edf", data, signal_headers, header)

        print("Complete. File saved to "
              "{}".format(self.chest_out_vital_file[:-len(self.chest_out_vital_file.split("/")[-1])]))

    def write_chestecg_edf(self):
        """Writes chest_ecg_file to edf"""

        print("-Writing {} to edf format...".format(self.chest_ecg_file))

        channel_names = ["ecg", "lead_off"]

        data = np.array(self.chest_ecg[channel_names]).transpose()

        signal_headers = pyedflib.highlevel.make_signal_headers(channel_names, sample_rate=self.chest_ecg_fs)
        header = pyedflib.highlevel.make_header(startdate=self.chest_ecg.iloc[0]["Timestamp"])
        pyedflib.highlevel.write_edf(self.chest_ecg_file.split(".")[0] + ".edf", data, signal_headers, header)

        print("Complete. File saved to "
              "{}".format(self.chest_ecg_file[:-len(self.chest_ecg_file.split("/")[-1])]))

    def write_chestacc_edf(self):
        """Writes chest_acc_file to edf"""

        print("-Writing {} to edf format...".format(self.chest_acc_file))

        channel_names = ["x", "y", "z"]

        data = np.array(self.chest_acc[channel_names]).transpose()

        signal_headers = pyedflib.highlevel.make_signal_headers(channel_names, sample_rate=self.chest_accz_fs)
        header = pyedflib.highlevel.make_header(startdate=self.chest_acc.iloc[0]["Timestamp"])
        pyedflib.highlevel.write_edf(self.chest_acc_file.split(".")[0] + ".edf", data, signal_headers, header)

        print("Complete. File saved to "
              "{}".format(self.chest_acc_file[:-len(self.chest_acc_file.split("/")[-1])]))

    def write_limbvitalout_edf(self):
        """Writes limb_out_vital_file to edf"""

        print("-Writing {} to edf format...".format(self.limb_out_vital_file))

        channel_names = ["spO2_perc", "pr_bpm", "pi_perc", "spo2_sqi", "ppg_attach", "ppg_valid", "limb_temp",
                         "hr_alarm", "rr_alarm", "spo2_alarm", "chesttemp_alarm", "limbtemp_alarm", "apnea_alarm",
                         "exception", "chest_off", "limb_off"]

        self.df_limb["pr_bpm"] = [i if not np.isnan(i) else 0 for i in self.df_limb["pr_bpm"]]
        self.df_limb["spO2_perc"] = [i if not np.isnan(i) else 0 for i in self.df_limb["spO2_perc"]]

        data = np.array(self.df_limb[channel_names]).transpose()
        signal_headers = pyedflib.highlevel.make_signal_headers(channel_names, sample_rate=5)
        header = pyedflib.highlevel.make_header(startdate=self.df_limb.iloc[0]["Timestamp"])
        pyedflib.highlevel.write_edf(self.limb_out_vital_file.split(".")[0] + ".edf", data, signal_headers, header)

        print("Complete. File saved to "
              "{}".format(self.limb_out_vital_file[:-len(self.chest_out_vital_file.split("/")[-1])]))

    def write_limbppg_edf(self):
        """Writes limb_ppg_file to edf"""

        print("-Writing {} to edf format...".format(self.limb_ppg_file))

        channel_names = ["red", "ir", "detached"]

        data = np.array(self.limb_ppg[channel_names]/10000).transpose()

        signal_headers = pyedflib.highlevel.make_signal_headers(channel_names, sample_rate=self.limb_ppg_fs)
        header = pyedflib.highlevel.make_header(startdate=self.limb_ppg.iloc[0]["Timestamp"])
        pyedflib.highlevel.write_edf(edf_file=self.limb_ppg_file.split(".")[0] + ".edf",
                                     signals=data, signal_headers=signal_headers, header=header, digital=False)

        """output_file = pyedflib.EdfWriter(file_name=self.limb_ppg_file.split(".")[0] + ".edf",
                                         n_channels=2, file_type=1)

        output_file.setDigitalMaximum(edfsignal=0, digital_maximum=32767)
        output_file.setDigitalMinimum(edfsignal=0, digital_minimum=-32767)
        output_file.setDigitalMaximum(edfsignal=1, digital_maximum=32767)
        output_file.setDigitalMinimum(edfsignal=1, digital_minimum=-32767)

        output_file.setSamplefrequency(edfsignal=0, samplefrequency=self.limb_ppg_fs)
        output_file.setSamplefrequency(edfsignal=1, samplefrequency=self.limb_ppg_fs)

        output_file.setSignalHeaders(header)

        output_file.setLabel(edfsignal=0, label=channel_names[0])
        output_file.setLabel(edfsignal=1, label=channel_names[1])

        output_file.writeDigitalSamples(data)"""

        print("Complete. File saved to "
              "{}".format(self.limb_ppg_file[:-len(self.limb_ppg_file.split("/")[-1])]))


def crop_data(bf_file=None, lankle_ga_file=None, rankle_ga_file=None,
               lwrist_ga_file=None, rwrist_ga_file=None):
    """Function that crops ANNE or Bittium data to align at start of collection."""

    # Default start indexes
    bf_index = 0
    ra_index = 0
    la_index = 0
    lw_index = 0
    rw_index = 0

    # Gets info from all available files
    bf_start, bf_end, bf_fs, bf_dur = ImportEDF.check_file(filepath=bf_file, print_summary=False)
    ra_start, ra_end, ra_fs, ra_dur = ImportEDF.check_file(filepath=rankle_ga_file, print_summary=False)
    la_start, la_end, la_fs, la_dur = ImportEDF.check_file(filepath=lankle_ga_file, print_summary=False)
    lw_start, lw_end, lw_fs, lw_dur = ImportEDF.check_file(filepath=lwrist_ga_file, print_summary=False)
    rw_start, rw_end, rw_fs, rw_dur = ImportEDF.check_file(filepath=rwrist_ga_file, print_summary=False)

    try:
        chest_start = anne.df_chest["Timestamp"].iloc[0]
    except NameError:
        chest_start = None

    start_times = {"BittiumFaros": bf_start, "ChestANNE": chest_start,
                   "RAnkle": ra_start, "LAnkle": la_start,
                   "LWrist": lw_start, "RWrist": rw_start}

    # Timestamp of device that started last
    crop_time = max([i for i in start_times.values() if i is not None])

    # Individual start indexes ---------------------------------------------------------------------------------------
    if bf_start is not None and bf_start != crop_time:
        delta_t = (crop_time - bf_start).total_seconds()
        bf_index = int(delta_t * bf_fs)

    if ra_start is not None and ra_start != crop_time:
        delta_t = (crop_time - ra_start).total_seconds()
        ra_index = int(delta_t * ra_fs)

    if la_start is not None and la_start != crop_time:
        delta_t = (crop_time - la_start).total_seconds()
        la_index = int(delta_t * la_fs)

    if lw_start is not None and lw_start != crop_time:
        delta_t = (crop_time - lw_start).total_seconds()
        lw_index = int(delta_t * lw_fs)

    if rw_start is not None and rw_start != crop_time:
        delta_t = (crop_time - rw_start).total_seconds()
        rw_index = int(delta_t * rw_fs)

    if chest_start != crop_time:
        anne.chest_ecg = anne.chest_ecg.loc[anne.chest_ecg["Timestamp"] >= crop_time]
        anne.chest_acc = anne.chest_acc.loc[anne.chest_acc["Timestamp"] >= crop_time]

    output_dict = {"BittiumFaros": bf_index, "RAnkle": ra_index, "LAnkle": la_index,
                   "LWrist": lw_index, "RWrist": rw_index}

    return output_dict


# =========================================================== SET UP ==================================================

# bittium_file = "C:/Users/ksweber/Desktop/007_ANNEValidation/007_Stingray.EDF"
bittium_file = None
la_file = "C:/Users/ksweber/Desktop/007_ANNEValidation/007_Test_GENEActiv_LA_Accelerometer.edf"
lw_file = "C:/Users/ksweber/Desktop/007_ANNEValidation/007_Test_GENEActiv_LW_Accelerometer.edf"
rw_file = None
ra_file = None

anne = ANNE(subj_id="007",
            chest_acc_file="C:/Users/ksweber/Desktop/007_ANNEValidation/ChestC1515_accl.edf",
            chest_ecg_file="C:/Users/ksweber/Desktop/007_ANNEValidation/ChestC1515_ecg.edf",
            chest_out_vital_file="C:/Users/ksweber/Desktop/007_ANNEValidation/ChestC1515_out_vital.edf",
            limb_ppg_file="C:/Users/ksweber/Desktop/007_ANNEValidation/Limb1307_ppg.edf",
            limb_out_vital_file="C:/Users/ksweber/Desktop/007_ANNEValidation/Limb1307_out_vital.edf",
            log_file="C:/Users/ksweber/Desktop/ANNE_Validation_Logs.xlsx")
anne.import_data()

offset_dict = crop_data(bf_file=bittium_file, lwrist_ga_file=lw_file, rwrist_ga_file=rw_file,
                        lankle_ga_file=la_file, rankle_ga_file=ra_file)

"""
bf = ECG.ECG(subject_id=anne.subj_id, filepath=bittium_file,
             output_dir=None, processed_folder=None,
             processed_file=None, ecg_downsample=1,
             age=26,
             start_offset=offset_dict["BittiumFaros"], end_offset=0,
             rest_hr_window=60, n_epochs_rest=30,
             epoch_len=15, load_accel=True,
             filter_data=True, low_f=.67, high_f=25, f_type="bandpass",
             load_raw=True, from_processed=False)
"""

anne.epoch_hr = anne.epoch_chest_hr(epoch_len=15)
anne.epoch_acc = anne.epoch_chest_acc(epoch_len=15)
anne.epoch_limb = anne.epoch_limb_data(epoch_len=15)

# Filters chest accelerometer data
# anne.filter_acc_data(filter_type='bandpass', low_f=0.05, high_f=10)

# Plots accel data. Able to plot just raw or raw and filtered. Able to adjust sample rate.
# anne.plot_acc(sample_rate=25, show_filtered=True)


# ==================================================== DATA VISUALIZATION =============================================

"""Compares 15-second averaged HR between chest ANNE and Bittium Faros"""
# epoch_hr_blandaltman()


class DataViewer:

    def __init__(self, anne_obj=None, bf_obj=None,
                 lwrist_obj=None, rwrist_obj=None, lankle_obj=None, rankle_obj=None,
                 fig_width=10, fig_height=6):

        self.anne = anne_obj
        self.bf = bf_obj
        self.lwrist = lwrist_obj
        self.rwrist = rwrist_obj
        self.lankle = lankle_obj
        self.rankle = rankle_obj

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
        self.reset_button = None

    def plot_data(self):

        plt.close("all")
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(self.fig_width, self.fig_height))
        plt.subplots_adjust(right=.8, hspace=.24)
        ax4.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

        """============================================== Heart Rate ==============================================="""
        ax1.set_title("Heart Rate/ECG Data")

        ax1.set_ylabel("bpm")

        if (self.hr_data_dict["Raw"] or self.hr_data_dict["Filt."]) and not self.hr_data_dict["HR"]:
            ax1.set_ylabel("Voltage (scaled)")

        # ANNE Chest
        if self.hr_plot_dict["ANNE Chest"] and self.anne.df_chest is not None:
            if self.hr_data_dict["HR"]:
                ax1.plot(self.anne.df_chest["Timestamp"], self.anne.df_chest["hr_bpm"],
                         color='red', label='ANNE Chest')
            if self.hr_data_dict["Raw"]:
                ax1.plot(self.anne.chest_ecg["Timestamp"][::4],
                         preprocessing.scale(self.anne.chest_ecg["ecg"][::4]),
                         color='red', label='ChestANNE_Raw')
            if self.hr_data_dict["Filt."]:
                ax1.plot(self.anne.chest_ecg["Timestamp"][::4],
                         preprocessing.scale(self.anne.chest_ecg["ecg_filt"][::4]),
                         color='red' if not self.hr_data_dict["Raw"] else 'black', label='ChestAnne_Filt')

        # ANNE Limb
        if self.hr_plot_dict["ANNE Limb"] and self.hr_data_dict["HR"] and self.anne.df_limb is not None:
            ax1.plot(self.anne.df_limb["Timestamp"], self.anne.df_limb["pr_bpm"],
                     color='dodgerblue', label='ANNE Limb')

        # Bittium Faros
        if self.hr_plot_dict["BittiumFaros"] and self.bf is not None:
            if self.hr_data_dict["HR"]:
                ax1.plot(self.bf.epoch_timestamps, self.bf.valid_hr, color='black', label='BF')
            if self.hr_data_dict["Raw"]:
                ax1.plot(self.bf.timestamps[::2], preprocessing.scale(self.bf.raw[::2]),
                         color='black', label="BF_Raw")
            if self.hr_data_dict["Filt."]:
                ax1.plot(self.bf.timestamps[::2], preprocessing.scale(self.bf.filtered[::2]),
                         color='black' if not self.hr_data_dict["Raw"] else 'red', label="BF_Filt")

        if True in self.hr_plot_dict.values():
            ax1.legend(loc='upper left')

        rax1 = plt.axes([.81, .72, .115, .16])
        rax1_2 = plt.axes([.925, .72, .065, .16])
        self.check1 = CheckButtons(rax1, ("ANNE Chest", "ANNE Limb", "BittiumFaros"),
                                   (False, False, False))
        self.check1_data = CheckButtons(rax1_2, ("HR", "Raw", "Filt."), (False, False, False))

        """============================================= Accelerometer ============================================="""
        ax2.set_title("Accelerometer Data")
        ax2.set_ylabel("G")

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
        if self.accel_plot_dict["BittiumFaros"] and self.bf is not None:
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
        ax3.set_ylim(-1, 40)

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
        ax4.set_ylabel("Mysterious Units")

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
            ax4.set_ylim(0, 105)
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

        rax_reload = plt.axes([0.84, 0.05, 0.074, .043])
        # self.reload_button = Button(rax_reload, 'Reload', color='chartreuse')
        self.reload_button = Button(rax_reload, 'Reload', color='#7AC141')
        self.reload_button.on_clicked(self.get_values)

        rax_events = plt.axes([0.84, 0.005, 0.074, .043])
        # self.events_button = Button(rax_events, 'Events', color='mediumturquoise')
        self.events_button = Button(rax_events, 'Events', color='#8476B6')
        self.events_button.on_clicked(self.set_events)

        rax_help = plt.axes([0.917, .005, 0.074, .043])
        # self.help_button = Button(rax_help, 'Help', color='crimson')
        self.help_button = Button(rax_help, 'Help', color='#EF333A')
        self.help_button.on_clicked(self.print_desc)

        rax_reset = plt.axes([0.917, 0.05, 0.074, .043])
        # self.reset_button = Button(rax_reset, 'Reset', color='slateblue')
        self.reset_button = Button(rax_reset, 'Reset', color='#F57E21')
        self.reset_button.on_clicked(self.reset_plot)

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

    def reset_plot(self, event):

        self.hr_plot_dict = {"ANNE Chest": False, "ANNE Limb": False, "BittiumFaros": False}
        self.accel_plot_dict = {"ANNE Chest": False, "ANNE Limb": False, "BittiumFaros": False,
                                "WristGA": False, "AnkleGA": False}
        self.accel_axis_dict = {"x": False, "y": False, "z": False, "SVM": False}
        self.temp_plot_dict = {"ANNE Chest": False, "ANNE Limb": False, "WristGA": False, "AnnkleGA": False}
        self.misc_plot_dict = {"ANNE Limb ppg": False, "ANNE Limb sO2": False,
                               "ANNE Chest Resp.": False, "ECG Validity": False}

        self.show_events = False

        self.plot_data()

    def compare_hr_data(self, plot_type='scatter', device1="ANNE Chest", device2="Bittium Faros"):

        print("\nGenerating {} plot to compare {} and {} HR...".format(plot_type, device1, device2))

        if device1 == "ANNE Chest":
            df1 = [i if not np.isnan(i) else None for i in self.anne.epoch_hr["hr_bpm"]]
        if device2 == "ANNE Chest":
            df2 = [i if not np.isnan(i) else None for i in self.anne.epoch_hr["hr_bpm"]]

        if device1 == "ANNE Limb":
            df1 = [i if not np.isnan(i) else None for i in self.anne.epoch_limb["pr_bpm"]]
        if device2 == "ANNE Limb":
            df2 = [i if not np.isnan(i) else None for i in self.anne.epoch_limb["pr_bpm"]]

        if device1 == "Bittium Faros":
            df1 = self.bf.valid_hr
        if device2 == "Bittium Faros":
            df2 = self.bf.valid_hr

        if df1 is None or df2 is None:
            print("-Not enough data.")
            return None

        if plot_type == "blandaltman" or plot_type == "bland-altman":
            means = []
            diffs = []
            for d1, d2 in zip(df1, df2):
                if d1 is not None and d2 is not None:
                    means.append((d1 + d2) / 2)
                    diffs.append(d2 - d1)

            loa = np.std(diffs) * 1.96
            bias = np.mean(diffs)

            fig, ax = plt.subplots(1, figsize=(self.fig_width, self.fig_height))
            ax.scatter(x=means, y=diffs, color='black', s=5)

            ax.axhline(bias + loa, color='red', linestyle='dashed', label='Upper LOA ({}bpm)'.format(round(bias + loa, 1)))
            ax.axhline(bias, color='black', linestyle='dashed', label='Bias ({}bpm)'.format(round(bias, 1)))
            ax.axhline(bias - loa, color='red', linestyle='dashed', label='Lower LOA ({}bpm)'.format(round(bias - loa, 1)))

            ax.fill_between(x=[min(means), max(means)], y1=plt.ylim()[0], y2=bias - loa, color='red', alpha=.25)
            ax.fill_between(x=[min(means), max(means)], y1=bias + loa, y2=plt.ylim()[1], color='red', alpha=.25)
            ax.fill_between(x=[min(means), max(means)], y1=bias - loa, y2=bias + loa, color='green', alpha=.25)

            ax.set_xlabel("Mean HR")
            ax.set_ylabel("Difference ({} - {})".format(device2, device1))
            plt.legend()

            perc_same = len([i for i in diffs if (bias - loa) <= i <= (bias + loa)]) / len(diffs) * 100
            ax.set_title("Bland-Altman Comparison ({}% agreement)".format(round(perc_same, 1)))

        if plot_type == "scatter":

            fig, ax = plt.subplots(1, figsize=(self.fig_width, self.fig_height))

            ax.scatter(df1[:min([len(df1), len(df2)])], df2[:min([len(df1), len(df2)])], color='black', s=6)
            min_val = min(min([i for i in df1 if i is not None]), min([i for i in df2 if i is not None]))
            max_val = max(max([i for i in df1 if i is not None]), max([i for i in df2 if i is not None]))

            ax.plot(np.arange(.9*min_val, 1.1*max_val), np.arange(.9*min_val, 1.1*max_val),
                    color='green', linestyle='dashed')
            ax.set_xlabel(device1)
            ax.set_ylabel(device2)
            ax.set_title("HR Comparison: {} and {}".format(device1, device2))

    @staticmethod
    def print_desc(event):

        print("\n========================= PLOT OPERATION AND DESCRIPTION ========================= ")

        print("\nOPERATION:")
        print("-Check off the box for all the data you wish to see. Then click the 'reload' button.")
        print("-To show or hide the events from the ANNE validation protocol, click the 'Events' button.")

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


test = DataViewer(anne_obj=anne, bf_obj=None, fig_width=12, fig_height=9)
test.plot_data()

# TODO
# Something is up with ANNE chest accel sample rate --> data "drifts" through collection
# Adjust sampling rates
# chest_acc edf will be wrong until this is done

# chest_start_times if not reading in all data files from ANNE

# Epoching limb ANNE data to match epochs from BF/Chest ANNE

# Finish bland_altman plotting -> change variables using arguments

# write_limb_ppg has to scale data to fit in EDF max/min values?
# Do units matter? I don't even know what they are

"""
# Checking sample rate stuff

anne_s = [i/anne.chest_acc_fs for i in range(int(anne.chest_acc.shape[0]/8))]
bf_s = [i/bf.accel_sample_rate for i in range(len(bf.accel_x))]

plt.plot(anne_s[:len(anne.chest_acc["x"][::8])], anne.chest_acc["x"][::8][:len(anne_s)], color='red', label="ANNE")
plt.plot(bf_s, [i*-1 for i in bf.accel_x], color='black', label="BF")
plt.legend(loc='upper left')
plt.title("ANNE acc_z sample rate = " + str(anne.chest_accz_fs) + "Hz")
plt.ylabel("G")
plt.xlabel("Seconds")
"""
