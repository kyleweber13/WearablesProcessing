import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import Filtering
from datetime import datetime
from datetime import timedelta
import numpy as np
import ECG
import ImportEDF


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

    def plot_outcome_measure(self, outcome_measure):

        units_dict = {"hr_bpm": "BPM", "hr_sqi": "Fraction", "ecg_leadon": "Binary (1 = ON)",
                      "ecg_valid": "Binary (1 = Yes)", "rr_rpm": "Breaths/min", "apnea_s": "Seconds",
                      "rr_sqi": "Fraction", "accx_g": "G's", "accy_g": "G's", "accz_g": "G's",
                      "chesttemp_c": "Degrees Celcius", "hr_alarm": "Binary (1 = Yes)", "rr_alarm": "Binary (1 = Yes)",
                      "spo2_alarm": "Binary (1 = Yes)", "chesttemp_alarm": "Binary (1 = Yes)",
                      "limbtemp_alarm": "Binary (1 = Yes)", "apnea_alarm": "Binary (1 = Yes)",
                      "exception": "Binary (1 = Yes)", "chest_off": "Binary (1 = Yes)", "limb_off": "Binary (1 = Yes)"}

        if len(outcome_measure) == 1 or type(outcome_measure) == str:
            if outcome_measure[0] not in units_dict.keys():
                print("-Invalid outcome measure ({}). "
                      "Select from the list below and try again:".format(outcome_measure))
                print([i for i in self.df_chest.keys()])
                return None

            fig, ax = plt.subplots(1, figsize=(12, 8))
            plt.subplots_adjust(bottom=.1)
            ax.set_title(outcome_measure)
            ax.plot(self.df_chest["Timestamp"], self.df_chest[outcome_measure], color='black')
            ax.set_ylabel(units_dict[outcome_measure[0]])
            ax.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45, fontsize=8)

        if len(outcome_measure) == 2:
            if outcome_measure[0] not in units_dict.keys():
                print("-Invalid outcome measure ({}). "
                      "Select from the list below and try again:".format(outcome_measure[0]))
                print([i for i in self.df_chest.keys()])
                return None

            if outcome_measure[1] not in units_dict.keys():
                print("-Invalid outcome measure ({}). "
                      "Select from the list below and try again:".format(outcome_measure[1]))
                print([i for i in self.df_chest.keys()])
                return None

            fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(12, 8))
            plt.subplots_adjust(bottom=.1)

            ax1.set_title(outcome_measure[0])
            ax1.plot(self.df_chest["Timestamp"], self.df_chest[outcome_measure[0]], color='black')
            ax1.set_ylabel(units_dict[outcome_measure[0]])

            ax2.set_title(outcome_measure[1])
            ax2.plot(self.df_chest["Timestamp"], self.df_chest[outcome_measure[1]], color='dodgerblue')
            ax2.set_ylabel(units_dict[outcome_measure[1]])

            ax2.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45, fontsize=8)

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

        # anne.chest_ecg = anne.chest_ecg.iloc[anne_chest_ecg:]
        # anne.chest_acc = anne.chest_acc.iloc[anne_chest_acc:]
        # anne.df_chest = anne.df_chest.iloc[anne_chest_vital:]
        anne.chest_ecg = anne.chest_ecg.loc[anne.chest_ecg["Timestamp"] >= bf_start]
        anne.chest_acc = anne.chest_acc.loc[anne.chest_acc["Timestamp"] >= bf_start]

    return bf_index


# =========================================================== SET UP ==================================================

bittium_file = "C:/Users/ksweber/Desktop/007_Test_Chest_C1515/007_Stingray.EDF"

anne = ANNE(subj_id="007",
            chest_acc_file="C:/Users/ksweber/Desktop/007_Test_Chest_C1515/accl.csv",
            chest_ecg_file="C:/Users/ksweber/Desktop/007_Test_Chest_C1515/ecg.csv",
            chest_out_vital_file="C:/Users/ksweber/Desktop/007_Test_Chest_C1515/out_vital.csv",
            limb_ppg_file="C:/Users/ksweber/Desktop/007_Test_Limb_L1307/ppg.csv",
            limb_out_vital_file="C:/Users/ksweber/Desktop/007_Test_Limb_L1307/out_vital.csv",
            log_file="C:/Users/ksweber/Desktop/ANNE_Validation_Logs.xlsx")
anne.import_data()

bittium_offset = crop_data(bf_file=bittium_file)


bf = ECG.ECG(subject_id=anne.subj_id, filepath=bittium_file,
             output_dir=None, processed_folder=None,
             processed_file=None, ecg_downsample=1,
             age=26, start_offset=bittium_offset, end_offset=0,
             rest_hr_window=60, n_epochs_rest=30,
             epoch_len=15, load_accel=True,
             filter_data=False, low_f=1, high_f=30, f_type="bandpass",
             load_raw=True, from_processed=False)


anne.epoch_hr = anne.epoch_chest_hr(epoch_len=15)
anne.epoch_acc = anne.epoch_chest_acc(epoch_len=15)

# Able to plot one or two variables
# anne.plot_outcome_measure(['hr_bpm'])

# Filters ecg data
# anne.filter_ecg_data(filter_type='bandpass', low_f=.67, high_f=25)

# Plots ecg data. Able to plot just raw or raw and filtered. Able to adjust sample rate.
# anne.plot_ecg(show_filtered=True, sample_rate=125)

# Filters chest accelerometer data
# anne.filter_acc_data(filter_type='bandpass', low_f=0.05, high_f=10)

# Plots accel data. Able to plot just raw or raw and filtered. Able to adjust sample rate.
# anne.plot_acc(sample_rate=25, show_filtered=False)

"""

# ECG
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(12, 7))
ax1.plot(anne.chest_ecg["Timestamp"][::5], anne.chest_ecg['ecg'][::5], color='red')
ax1.set_title("ANNE Chest ECG")
ax2.plot(anne.df_limb["Timestamp"], anne.df_limb["spO2_perc"], color='dodgerblue')
ax2.set_title("ANNE Limb SPO2")
ax3.plot(bf.timestamps[::3], bf.raw[::3], color='black')
ax3.set_title("BF Stingray ECG")

# ACC
fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(12, 7))
ax1.plot(anne.df_chest["Timestamp"], anne.df_chest['accx_g'], color='red')
ax1.set_title("Chest ANNE Acc_x")
ax2.plot(bf.timestamps[::10], bf.accel_x, color='black')
ax2.set_title("BF Stingray Acc_x")

"""


def compare_epoch_hr(plot_type="time series"):

    if plot_type == "time series" or plot_type == "timeseries":
        fig, ax = plt.subplots(1, figsize=(12, 7))

        ax.plot(bf.epoch_timestamps, bf.valid_hr, label='Bittium', color='black')
        ax.axvline(x=bf.epoch_timestamps[-1], linestyle='dashed', color='black', label="End BF")

        ax.plot(anne.epoch_hr["Timestamp"], anne.epoch_hr["hr_bpm"], label='ANNE', color='red')
        ax.axvline(x=anne.epoch_hr["Timestamp"].iloc[-1], linestyle='dashed', color='red', label="End ANNE")

        ax.legend(loc='best')
        ax.set_ylabel("HR (bpm)")
        plt.title("15-second avg HR comparison")

        ax.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

    if plot_type == "blandaltman":

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


def plot_hr_and_movement():
    """Plots HR and ANNE chest accelerometer SVM."""

    fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(12, 7))
    ax1.set_title("HR and Movement Data")

    ax1.plot(anne.epoch_hr["Timestamp"], anne.epoch_hr["hr_bpm"], color='red')
    ax1.set_ylabel("HR (bpm)")

    ax2.plot(anne.epoch_acc["Timestamp"], anne.epoch_acc["SVM"], color='dodgerblue')
    ax2.set_ylabel("SVM")

    ax2.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=45, fontsize=8)


# ==================================================== DATA VISUALIZATION =============================================

"""Adds events to an open plot"""
# anne.plot_events()

"""Compares 15-second averaged HR between chest ANNE and Bittium Faros"""
# plot_type: 'timeseries' or 'blandaltman'
# compare_epoch_hr(plot_type="blandaltman")

"""Plots ANNE chest HR and accelerometer SVM"""
# Can use to see if movement affects ECG quality (missing HR data)
# plot_hr_and_movement()

# TODO
# Plot to compare validity
