import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ECG
import ImportEDF
from datetime import timedelta
import statistics
import matplotlib.dates as mdates


class Data:

    def __init__(self, lw_file=None, rw_file=None, ecg_file=None, gait_log=None,
                 epoch_len=15, gait_thresh_min=10, dom_hand="Right", pad_window=5):

        self.lw_file = lw_file
        self.rw_file = rw_file
        self.ecg_file = ecg_file
        self.gaitlog_file = gait_log
        self.epoch_len = epoch_len
        self.accel_fs = 75
        self.gait_thresh = gait_thresh_min
        self.dom_hand = dom_hand.capitalize()
        self.pad_window = pad_window

        self.crop_dict = None
        self.lw = None
        self.rw = None
        self.ecg = None

        self.df_svm = None
        self.epoch_hr = pd.DataFrame(list(zip([], [], [], [])), columns=["Timestamp", "HR", "%HRR", "HRR_Intensity"])

        self.gait_log = None
        self.gait_log_long = None

        self.rest_hr = 0

        self.cutpoints = None

        self.df_walks = None

    def check_sync(self):
        """Checks start times for all given files. Makes sure all devices start at same time."""

        lw_crop_index = 0
        rw_crop_index = 0
        ecg_crop_index = 0

        lw_starttime, lw_endtime, lw_fs, lw_duration = ImportEDF.check_file(filepath=self.lw_file, print_summary=False)

        rw_starttime, rw_endtime, rw_fs, rw_duration = ImportEDF.check_file(filepath=self.rw_file, print_summary=False)

        ecg_starttime, ecg_endtime, ecg_fs, ecg_duration = ImportEDF.check_file(filepath=self.ecg_file,
                                                                                print_summary=False)

        crop_time = max([lw_starttime, rw_starttime, ecg_starttime])

        if lw_starttime < crop_time:
            lw_crop_index = int((crop_time - lw_starttime).total_seconds() * lw_fs)
        if rw_starttime < crop_time:
            rw_crop_index = int((crop_time - rw_starttime).total_seconds() * rw_fs)
        if ecg_starttime < crop_time:
            ecg_crop_index = int((crop_time - ecg_starttime).total_seconds() * ecg_fs)

        self.crop_dict = {"LW": lw_crop_index, "RW": rw_crop_index, "ECG": ecg_crop_index}

        if lw_fs != rw_fs:
            print("\n-Accelerometer sampling rates do not match. Errors will ensue.")
        if lw_fs == rw_fs:
            self.accel_fs = lw_fs

    def import_data(self):
        """Imports wrist and ECG data"""

        self.lw = ImportEDF.GENEActiv(filepath=self.lw_file, load_raw=True, start_offset=self.crop_dict["LW"])
        self.rw = ImportEDF.GENEActiv(filepath=self.rw_file, load_raw=True, start_offset=self.crop_dict["RW"])
        self.ecg = ImportEDF.Bittium(filepath=self.ecg_file, load_accel=False, start_offset=self.crop_dict["ECG"],
                                     epoch_len=self.epoch_len)

    def scale_cutpoints(self):
        """Scales accelerometer cutpoints based on epoch length and sampling rate"""

        nd_light = 255 * self.accel_fs / 100 * self.epoch_len / 60
        nd_mod = 588 * self.accel_fs / 100 * self.epoch_len / 60

        d_light = 375 * self.accel_fs / 100 * self.epoch_len / 60
        d_mod = 555 * self.accel_fs / 100 * self.epoch_len / 60

        self.cutpoints = {"ND_Light": nd_light, "ND_Mod": nd_mod, "D_Light": d_light, "D_Mod": d_mod}

    def epoch1s_accels(self):
        """Epochs accelerometer data into one-second epochs"""

        print("\n-Epoching accelerometer data into 1-second epochs...")
        lw_svm = []
        rw_svm = []

        for i in range(0, min(len(self.lw.vm), len(self.rw.vm)), self.accel_fs):

            lw_svm.append(sum(self.lw.vm[i:i+self.accel_fs]))
            rw_svm.append(sum(self.rw.vm[i:i+self.accel_fs]))

        coll_dur = (pd.to_datetime(self.lw.timestamps[-1], "%Y-%m-%dT%H:%M:%S.%f") -
                    pd.to_datetime(self.lw.timestamps[0],"%Y-%m-%dT%H:%M:%S.%f")).total_seconds()

        epoch_stamps = pd.date_range(start=self.lw.timestamps[0], freq="1S", periods=coll_dur)

        self.df_svm = pd.DataFrame(list(zip(epoch_stamps, lw_svm, rw_svm)), columns=["Timestamp", "LW_SVM", "RW_SVM"])

    def import_gait_log(self, threshold=None):
        """Imports gait long and stores long walks as separate dataframe

            :argument
            -threshold: thresholds for a 'long walk' in minutes
        """

        if threshold is not None:
            self.gait_thresh = threshold

        if threshold is None:
            threshold = self.gait_thresh

        print("\nImporting gait log...")
        self.gait_log = pd.read_csv(self.gaitlog_file).iloc[:, 1:]
        self.gait_log = self.gait_log[["gait_bout_num", "start_timestamp", "end_timestamp",
                                       "bout_length_sec", "total_cycle_count"]]

        self.gait_log["start_timestamp"] = pd.to_datetime(self.gait_log["start_timestamp"])
        self.gait_log["end_timestamp"] = pd.to_datetime(self.gait_log["end_timestamp"])

        self.gait_log_long = self.gait_log.loc[self.gait_log["bout_length_sec"] >= threshold * 60].reset_index()

        if self.gait_log_long.shape[0] == 0:
            print("-No bouts lasting longer than {} minutes were found.".format(threshold))
        if self.gait_log_long.shape[0] >= 1:
            print("-Found {} bouts lasting longer than {} minutes.".format(self.gait_log_long.shape[0], threshold))

    def process_gait_bouts(self, pad_len_min=None):
        """Processes data only during long walks.

            :argument
            -pad_len_min: 'pads' data with pre/post walk data, number of minutes
        """

        if pad_len_min is not None:
            self.pad_window = pad_len_min
        if pad_len_min is None:
            pad_len_min = self.pad_window

        print("\nProcessing gait bouts longer than {} minutes...".format(self.gait_thresh))
        print("-Including {} seconds pre/post bout.".format(pad_len_min * 60))

        lw_svm = []
        rw_svm = []
        walk_nums = []
        timestamps = []
        into_bouts = []
        event = []
        epoch_hr = []

        for bout in self.gait_log_long.itertuples():

            bout_start = int((bout.start_timestamp + timedelta(seconds=-pad_len_min*60) -
                              self.ecg.timestamps[0]).total_seconds() * self.ecg.sample_rate)

            bout_end = int((bout.end_timestamp + timedelta(seconds=pad_len_min*60) -
                            self.ecg.timestamps[0]).total_seconds() * self.ecg.sample_rate)

            for i in range(bout_start, bout_end, int(self.ecg.sample_rate * self.epoch_len)):
                qc = ECG.CheckQuality(raw_data=self.ecg.raw, start_index=i, template_data='filtered',
                                      voltage_thresh=250, epoch_len=self.epoch_len, sample_rate=self.ecg.sample_rate)

                if qc.valid_period:
                    epoch_hr.append(qc.hr)
                if not qc.valid_period:
                    epoch_hr.append(None)

            df = self.df_svm.loc[(self.df_svm["Timestamp"] >=
                                  bout.start_timestamp + timedelta(seconds=-pad_len_min*60)) &
                                 (self.df_svm["Timestamp"] < bout.end_timestamp +
                                  timedelta(seconds=pad_len_min*60))]

            stamps = pd.date_range(start=bout.start_timestamp + timedelta(seconds=-self.pad_window * 60),
                                   end=bout.end_timestamp + timedelta(seconds=self.pad_window * 60 + self.epoch_len),
                                   freq="{}S".format(self.epoch_len))

            for start, end in zip(stamps[:], stamps[1:]):
                d = df.loc[(df["Timestamp"] >= start) & (df["Timestamp"] < end)]
                lw_svm.append(sum(d["LW_SVM"]))
                rw_svm.append(sum(d["RW_SVM"]))
                walk_nums.append(bout.Index + 1)
                timestamps.append(start)

                """into_bout = (start - bout.start_timestamp).total_seconds()
                if start <= bout.end_timestamp:
                    into_bouts.append(into_bout)
                if start > bout.end_timestamp:
                    into_bouts.append("+" + str((round((start - bout.end_timestamp).total_seconds(), 1))))"""
                into_bout = (start - bout.start_timestamp).total_seconds()

                if start <= bout.start_timestamp:
                    event.append("Pre")
                    into_bouts.append(into_bout)
                if bout.start_timestamp < start <= bout.end_timestamp:
                    event.append("Bout")
                    into_bouts.append(into_bout)
                if start > bout.end_timestamp:
                    event.append("Post")
                    into_bouts.append(round((start-bout.end_timestamp).total_seconds(), 1))

        epoch = pd.DataFrame(list(zip(timestamps, walk_nums, [round(i, 2) for i in lw_svm],
                                      [round(i, 2) for i in rw_svm],
                                      [round(i, 1) if i is not None else None for i in epoch_hr], into_bouts, event)),
                             columns=["Timestamp", "Walk_num", "LW_SVM", "RW_SVM", "HR", "Time_into_bout", "Event"])

        self.df_walks = epoch

    def calculate_wrist_intensity(self):
        """Calculates wrist intensity using cutpoints."""

        lw = []
        rw = []

        if self.dom_hand == "Right":

            for epoch in self.df_walks.itertuples():
                if epoch.LW_SVM < self.cutpoints["ND_Light"]:
                    lw.append("Sedentary")
                if self.cutpoints["ND_Light"] <= epoch.LW_SVM < self.cutpoints["ND_Mod"]:
                    lw.append("Light")
                if self.cutpoints["ND_Mod"] <= epoch.LW_SVM:
                    lw.append("Moderate")

                if epoch.RW_SVM < self.cutpoints["D_Light"]:
                    rw.append("Sedentary")
                if self.cutpoints["D_Light"] <= epoch.RW_SVM < self.cutpoints["D_Mod"]:
                    rw.append("Light")
                if self.cutpoints["D_Mod"] <= epoch.RW_SVM:
                    rw.append("Moderate")

        if self.dom_hand == "Left":

            for epoch in self.df_walks.itertuples():
                if epoch.LW_SVM < self.cutpoints["D_Light"]:
                    lw.append("Sedentary")
                if self.cutpoints["D_Light"] <= epoch.LW_SVM < self.cutpoints["D_Mod"]:
                    lw.append("Light")
                if self.cutpoints["D_Mod"] <= epoch.LW_SVM:
                    lw.append("Moderate")

                if epoch.RW_SVM < self.cutpoints["ND_Light"]:
                    rw.append("Sedentary")
                if self.cutpoints["ND_Light"] <= epoch.RW_SVM < self.cutpoints["ND_Mod"]:
                    rw.append("Light")
                if self.cutpoints["ND_Mod"] <= epoch.RW_SVM:
                    rw.append("Moderate")

        self.df_walks["LW_Intensity"] = lw
        self.df_walks["RW_Intensity"] = rw

    def find_resting_hr(self, rest_hr=None, window_size=60, n_windows=30, sleep_log=None):
        """Function that calculates resting HR based on inputs.
           Able to input a resting HR value to skip ECG processing.

        :argument
        -rest_hr: int/float if already calculated; will skip ECG processing
            -If None, will perform ECG processing
        -sleep_log: pathway to sleep log
            -If None, will (obviously) not remove sleep periods --> lower resting HR
        -window_size: size of window over which rolling average is calculated, seconds
        -n_windows: number of epochs over which resting HR is averaged (lowest n_windows number of epochs)
        -sleep_status: data from class Sleep that corresponds to asleep/awake epochs
        """

        # Sets integer for window length based on window_size and epoch_len
        window_len = int(window_size / self.epoch_len)

        if rest_hr is not None:
            self.rest_hr = rest_hr

        if rest_hr is None:

            # Runs QC algorithm and calculates epoch HRs --------------------------------------------------------------
            print("\nRunning ECG quality check algorithm on whole data file to find resting HR...")

            markers = np.arange(0, len(self.ecg.raw) * 1.1, len(self.ecg.raw)/10)
            marker_ind = 0

            epoch_hr = []
            for i in range(0, len(self.ecg.raw), int(self.ecg.sample_rate * self.epoch_len)):
                qc = ECG.CheckQuality(raw_data=self.ecg.raw, start_index=i, template_data='filtered',
                                      voltage_thresh=250, epoch_len=self.epoch_len, sample_rate=self.ecg.sample_rate)

                if i >= markers[marker_ind]:
                    print("{}% complete".format(marker_ind*10))
                    marker_ind += 1

                if qc.valid_period:
                    epoch_hr.append(qc.hr)
                if not qc.valid_period:
                    epoch_hr.append(None)

            print("100% complete")

            self.epoch_hr = pd.DataFrame(list(zip(pd.date_range(start=self.ecg.timestamps[0], end=self.ecg.timestamps[-1],
                                                             freq="{}S".format(self.epoch_len)),
                                                  [round(i, 1) for i in epoch_hr])),
                                         columns=["Timestamp", "HR"])

            # Calculates resting HR -----------------------------------------------------------------------------------
            try:
                rolling_avg = [statistics.mean(epoch_hr[i:i + window_len]) if None not in epoch_hr[i:i + window_len]
                               else None for i in range(len(epoch_hr))]
            except statistics.StatisticsError:
                print("No data points found.")
                rolling_avg = []

            # Calculates resting HR during waking hours if sleep_log available --------
            if sleep_log is not None:
                print("\nCalculating resting HR from periods of wakefulness...")

                # Flags sleep epochs from log
                df_sleep = pd.read_excel("/Users/kyleweber/Desktop/Sleep.xlsx")
                df_sleep = df_sleep.loc[df_sleep["Subject"] == subj_id]

                sleep_ind = [int((row.Sleep - x.lw.timestamps[0]).total_seconds() / x.epoch_len) for row in
                             df_sleep.itertuples()]
                wake_ind = [int((row.Wake - x.lw.timestamps[0]).total_seconds() / x.epoch_len) for row in
                            df_sleep.itertuples()]

                sleep_list = np.zeros(int(len(x.lw.timestamps) / x.lw.sample_rate / x.epoch_len))

                for s, w in zip(sleep_ind, wake_ind):
                    sleep_list[s:w] = 1

                awake_hr = [rolling_avg[i] for i in range(0, min([len(sleep_list), len(rolling_avg)]))
                            if sleep_list[i] == 0 and rolling_avg[i] is not None]

                sorted_hr = sorted(awake_hr)

                if len(sorted_hr) < n_windows:
                    resting_hr = "N/A"

                if len(sorted_hr) >= n_windows:
                    resting_hr = round(sum(sorted_hr[:n_windows]) / n_windows, 1)

                print("Resting HR (average of {} lowest {}-second periods while awake) is {} bpm.".format(n_windows,
                                                                                                          window_size,
                                                                                                          resting_hr))

            # Calculates resting HR during all hours if sleep_log not available --------
            if sleep_log is None:
                # print("\n" + "Calculating resting HR from periods of all data (sleep data not available)...")

                awake_hr = None

                valid_hr = [i for i in rolling_avg if i is not None]

                sorted_hr = sorted(valid_hr)

                resting_hr = round(sum(sorted_hr[:n_windows]) / n_windows, 1)

                print("Resting HR (sleep not removed; average of {} lowest "
                      "{}-second periods) is {} bpm.".format(n_windows, window_size, resting_hr))

            self.rest_hr = resting_hr

    def calculate_hrr(self, age=30):
        """Uses predicted max HR and measured resting HR to quantify %HRR data.

            :argument
            -age: participant age in years
        """

        print("\nCalculating %HRR data...")

        hrr = (208 - .7 * age - self.rest_hr)

        # HR during walks ---------------------------------------------------------------------------------------------
        hrr_list = [round((hr - self.rest_hr) / hrr * 100, 1) if not np.isnan(hr) else None
                    for hr in self.df_walks["HR"]]
        self.df_walks["%HRR"] = hrr_list

        hrr_intensity = []

        for hr in hrr_list:
            if hr is None:
                hrr_intensity.append(None)
            if hr is not None:
                if hr < 30:
                    hrr_intensity.append("Sedentary")
                if 30 <= hr < 40:
                    hrr_intensity.append("Light")
                if hr >= 40:
                    hrr_intensity.append("Moderate")

        self.df_walks["HRR_Intensity"] = hrr_intensity

        # HR during all data ------------------------------------------------------------------------------------------
        hrr_list = [round((hr - self.rest_hr) / hrr * 100, 1) if not np.isnan(hr) else None
                    for hr in self.epoch_hr["HR"]]
        self.epoch_hr["%HRR"] = hrr_list

        hrr_intensity = []

        for hr in hrr_list:
            if hr is None:
                hrr_intensity.append(None)
            if hr is not None:
                if hr < 30:
                    hrr_intensity.append("Sedentary")
                if 30 <= hr < 40:
                    hrr_intensity.append("Light")
                if hr >= 40:
                    hrr_intensity.append("Moderate")

        self.epoch_hr["HRR_Intensity"] = hrr_intensity

    def plot_all_data(self, start=None, stop=None, save_image=False, image_path=None):
        """Able to plot specified sections of data. Shades long walking bouts. Able to save.

            :argument
            -start/stop: timestamp or None to crop data
            -save_image: boolean. If True, saves image to path specified by image_path
            -image_path: save location and filename of image
        """

        if self.epoch_hr.shape[0] == 0:
            df_hr = self.df_walks
        if self.epoch_hr.shape[0] > 0:
            df_hr = self.epoch_hr

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 6))

        ax1.plot(self.df_svm["Timestamp"], self.df_svm["LW_SVM"], color='black')
        ax1.set_title("Left Wrist")
        ax1.set_ylabel("Counts")

        ax2.plot(self.df_svm["Timestamp"], self.df_svm["RW_SVM"], color='black')
        ax2.set_title("Right Wrist")
        ax2.set_ylabel("Counts")

        ax3.plot(df_hr["Timestamp"], df_hr['%HRR'], color='black')
        ax3.set_ylabel("%HRR")
        ax3.set_title("Heart Rate")

        ax4 = ax3.twinx()
        ax4.plot(df_hr['Timestamp'], df_hr["HR"], linestyle="")
        ax4.set_ylabel("HR (bpm)")
        ax4.axhline(self.rest_hr, linestyle='dashed', color='red')

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        ax4.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=7)

        for row in self.gait_log_long.itertuples():
            # Shades long walking bouts in green
            ax1.fill_between(x=[row.start_timestamp, row.end_timestamp], y1=0, y2=max(self.df_svm["LW_SVM"]),
                             color='green', alpha=.5)
            ax2.fill_between(x=[row.start_timestamp, row.end_timestamp], y1=0, y2=max(self.df_svm["RW_SVM"]),
                             color='green', alpha=.5)
            ax3.fill_between(x=[row.start_timestamp, row.end_timestamp],
                             y1=0, y2=max([i for i in df_hr["%HRR"] if not np.isnan(i)]), color='green', alpha=.5)

            # Shades padded area in red
            ax1.fill_between(x=[row.start_timestamp + timedelta(seconds=-self.pad_window * 60), row.start_timestamp],
                             y1=0, y2=max(self.df_svm["LW_SVM"]), color='lightgrey', alpha=.5)
            ax1.fill_between(x=[row.end_timestamp, row.end_timestamp + timedelta(seconds=self.pad_window * 60)],
                             y1=0, y2=max(self.df_svm["LW_SVM"]), color='orange', alpha=.5)

            ax2.fill_between(x=[row.start_timestamp + timedelta(seconds=-self.pad_window * 60), row.start_timestamp],
                             y1=0, y2=max(self.df_svm["RW_SVM"]), color='lightgrey', alpha=.5)
            ax2.fill_between(x=[row.end_timestamp, row.end_timestamp + timedelta(seconds=self.pad_window * 60)],
                             y1=0, y2=max(self.df_svm["RW_SVM"]), color='orange', alpha=.5)

            ax3.fill_between(x=[row.start_timestamp + timedelta(seconds=-self.pad_window * 60), row.start_timestamp],
                             y1=0, y2=max([i for i in df_hr["%HRR"] if not np.isnan(i)]), color='lightgrey', alpha=.5)
            ax3.fill_between(x=[row.end_timestamp, row.end_timestamp + timedelta(seconds=self.pad_window * 60)],
                             y1=0, y2=max([i for i in df_hr["%HRR"] if not np.isnan(i)]), color='orange', alpha=.5)

        if start is None and stop is None:
            pass
        if start is not None and stop is not None:
            ax4.set_xlim(start, stop)

            ax1.set_ylim(ax1.get_ylim()[0], max(self.df_svm.loc[(self.df_svm["Timestamp"] >= start) &
                                                                (self.df_svm["Timestamp"] < stop)]["LW_SVM"])*1.1)
            ax2.set_ylim(ax2.get_ylim()[0], max(self.df_svm.loc[(self.df_svm["Timestamp"] >= start) &
                                                                (self.df_svm["Timestamp"] < stop)]["RW_SVM"])*1.1)

        if save_image:
            plt.savefig(image_path)

    def generate_longwalk_images(self, image_path="/Users/kyleweber/Desktop/{}_LongWalk{}.png"):
        """Loops through long walks and calls self.plot_longwalk_data for specified region of data. Saves images.

        :argument
        -image_path: pathway where image(s) are saved. Include {} for walk index.
        """

        for row in self.gait_log_long.itertuples():
            plt.close('all')
            self.plot_all_data(start=row.start_timestamp + timedelta(minutes=-1.1 * self.pad_window),
                               stop=row.end_timestamp + timedelta(minutes=1.1 * self.pad_window),
                               save_image=True,
                               image_path=image_path.format(subj_id,row.Index + 1))
            plt.close('all')

    def save_data(self, pathway=""):
        """Saves relevant data to Excel files.

            :argument
            -pathway: where files are saved
        """

        print("\nSaving relevant data to {}..".format(pathway))

        self.df_walks.to_excel(pathway + "{}_LongWalk_Data.xlsx".format(subj_id), index=False)

        self.gait_log_long.to_excel(pathway + "{}_LongWalks_Log.xlsx".format(subj_id), index=False)


subj_id = 3034
x = Data(lw_file="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_GA_LWrist_Accelerometer.EDF".format(str(subj_id)),
         rw_file="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_GA_LAnkle_Accelerometer.EDF".format(str(subj_id)),
         ecg_file="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_BF.EDF".format(str(subj_id)),
         gait_log="/Users/kyleweber/Desktop/summary_data OND07 WTL {} sample.csv".format(str(subj_id)),
         epoch_len=15, gait_thresh_min=3, pad_window=5)

x.check_sync()
x.import_data()
x.epoch1s_accels()
x.scale_cutpoints()
x.import_gait_log(threshold=3)
x.process_gait_bouts(pad_len_min=3)
x.calculate_wrist_intensity()
x.find_resting_hr(rest_hr=57.8, window_size=60, n_windows=30, sleep_log="/Users/kyleweber/Desktop/Sleep.xlsx")
# x.find_resting_hr(rest_hr=None, window_size=60, n_windows=30, sleep_log="/Users/kyleweber/Desktop/Sleep.xlsx")
x.calculate_hrr(age=34)

# x.plot_all_data(save_image=False)
# x.generate_longwalk_images(image_path="/Users/kyleweber/Desktop/Kin 472 - Nidhi/{}_LongWalk{}.png")
# x.save_data(pathway="/Users/kyleweber/Desktop/Kin 472 - Nidhi/")
