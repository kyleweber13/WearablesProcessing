import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import ECG
import ImportEDF
from datetime import timedelta
import statistics
import matplotlib.dates as mdates


class Data:

    def __init__(self, wrist_file=None, ankle_file=None, ecg_file=None,
                 activity_log=None, sleep_log=None, epoch_len=15):

        self.wrist_file = wrist_file
        self.ankle_file = ankle_file
        self.ecg_file = ecg_file
        self.activity_file = activity_log
        self.sleep_file = sleep_log
        self.epoch_len = epoch_len
        self.accel_fs = 75

        self.crop_dict = None
        self.wrist = None
        self.ankle = None
        self.ecg = None

        self.df_svm = None
        self.epoch_hr = pd.DataFrame(list(zip([], [], [], [])), columns=["Timestamp", "HR", "%HRR", "HRR_Intensity"])

        self.rest_hr = 0
        self.pad_pre = 0
        self.pad_post = 0

        self.cutpoints = None

        self.df_active = None
        self.df_events = None

    def check_sync(self):
        """Checks start times for all given files. Makes sure all devices start at same time."""

        lw_crop_index = 0
        rw_crop_index = 0
        ecg_crop_index = 0

        wrist_starttime, wrist_endtime, wrist_fs, wrist_duration = ImportEDF.check_file(filepath=self.wrist_file,
                                                                                        print_summary=False)

        ankle_starttime, ankle_endtime, ankle_fs, ankle_duration = ImportEDF.check_file(filepath=self.ankle_file,
                                                                                        print_summary=False)

        ecg_starttime, ecg_endtime, ecg_fs, ecg_duration = ImportEDF.check_file(filepath=self.ecg_file,
                                                                                print_summary=False)

        crop_time = max([wrist_starttime, ankle_starttime, ecg_starttime])

        if wrist_starttime < crop_time:
            lw_crop_index = int((crop_time - wrist_starttime).total_seconds() * wrist_fs)
        if ankle_starttime < crop_time:
            rw_crop_index = int((crop_time - ankle_starttime).total_seconds() * ankle_fs)
        if ecg_starttime < crop_time:
            ecg_crop_index = int((crop_time - ecg_starttime).total_seconds() * ecg_fs)

        self.crop_dict = {"LW": lw_crop_index, "RW": rw_crop_index, "ECG": ecg_crop_index}

        if wrist_fs != ankle_fs:
            print("\n-Accelerometer sampling rates do not match. Errors will ensue.")
        if wrist_fs == ankle_fs:
            self.accel_fs = wrist_fs

    def import_data(self):
        """Imports wrist and ECG data"""

        self.wrist = ImportEDF.GENEActiv(filepath=self.wrist_file, load_raw=True, start_offset=self.crop_dict["LW"])
        self.ankle = ImportEDF.GENEActiv(filepath=self.ankle_file, load_raw=True, start_offset=self.crop_dict["RW"])
        self.ecg = ImportEDF.Bittium(filepath=self.ecg_file, load_accel=False, start_offset=self.crop_dict["ECG"],
                                     epoch_len=self.epoch_len)

    def scale_cutpoints(self):
        """Scales Duncan et al. (2019) accelerometer cutpoints based on epoch length and sampling rate"""

        ndwrist_light = 4.9 * self.accel_fs / 80 * self.epoch_len
        ndwrist_mvpa = 12.0 * self.accel_fs / 80 * self.epoch_len

        ankle_light = 4.5 * self.accel_fs / 80 * self.epoch_len
        ankle_mvpa = 129.2 * self.accel_fs / 80 * self.epoch_len

        self.cutpoints = {"NDWrist_Light": ndwrist_light, "NDWrist_MVPA": ndwrist_mvpa,
                          "Ankle_Light": ankle_light, "Ankle_MVPA": ankle_mvpa}

    def epoch1s_accels(self):
        """Epochs accelerometer data into one-second epochs"""

        print("\n-Epoching accelerometer data into 1-second epochs...")
        wrist_svm = []
        ankle_svm = []

        for i in range(0, min(len(self.wrist.vm), len(self.ankle.vm)), self.accel_fs):

            wrist_svm.append(sum(self.wrist.vm[i:i+self.accel_fs]))
            ankle_svm.append(sum(self.ankle.vm[i:i+self.accel_fs]))

        coll_dur = (pd.to_datetime(self.wrist.timestamps[-1], "%Y-%m-%dT%H:%M:%S.%f") -
                    pd.to_datetime(self.wrist.timestamps[0],"%Y-%m-%dT%H:%M:%S.%f")).total_seconds()

        epoch_stamps = pd.date_range(start=self.wrist.timestamps[0], freq="1S", periods=coll_dur)

        self.df_svm = pd.DataFrame(list(zip(epoch_stamps, wrist_svm, ankle_svm)),
                                   columns=["Timestamp", "Wrist_SVM", "Ankle_SVM"])

    def import_activity_log(self):
        """Imports activity log."""

        print("\nImporting activity log...")
        self.df_events = pd.read_excel(self.activity_file)

        self.df_events["Start"] = pd.to_datetime(self.df_events["Start"])
        self.df_events["Stop"] = pd.to_datetime(self.df_events["Stop"])

        if self.df_events.shape[0] == 0:
            print("-No activities found.")
        if self.df_events.shape[0] >= 1:
            print("-Found {} activity events.".format(self.df_events.shape[0]))

    def process_activity_bouts(self, pad_pre=0, pad_post=0):
        """Processes data only during activities.

            :argument
            -pad_pre: seconds to include prior to activity
            -pad_post: seconds to include post activity
        """

        print("-Including {} seconds pre-actvity and {} seconds post-activity.".format(pad_pre, pad_post))

        self.pad_pre, self.pad_post = pad_pre, pad_post

        wrist_svm = []
        ankle_svm = []
        timestamps = []
        into_bouts = []
        events = []
        epoch_hr = []
        desc = []

        for bout in self.df_events.itertuples():

            bout_start = int((bout.Start + timedelta(seconds=-pad_pre) -
                              self.ecg.timestamps[0]).total_seconds() * self.ecg.sample_rate)

            bout_end = int((bout.Stop + timedelta(seconds=pad_post) -
                            self.ecg.timestamps[0]).total_seconds() * self.ecg.sample_rate)

            for i in range(bout_start, bout_end, int(self.ecg.sample_rate * self.epoch_len)):
                qc = ECG.CheckQuality(raw_data=self.ecg.raw, start_index=i, template_data='filtered',
                                      voltage_thresh=500, epoch_len=self.epoch_len, sample_rate=self.ecg.sample_rate)

                if qc.valid_period:
                    epoch_hr.append(qc.hr)
                if not qc.valid_period:
                    epoch_hr.append(None)

            df = self.df_svm.loc[(self.df_svm["Timestamp"] >= bout.Start + timedelta(seconds=-pad_pre)) &
                                 (self.df_svm["Timestamp"] < bout.Stop +timedelta(seconds=pad_post))]

            stamps = pd.date_range(start=bout.Start + timedelta(seconds=-pad_pre),
                                   end=bout.Stop + timedelta(seconds=pad_post + self.epoch_len),
                                   freq="{}S".format(self.epoch_len))

            for start, end in zip(stamps[:], stamps[1:]):
                d = df.loc[(df["Timestamp"] >= start) & (df["Timestamp"] < end)]
                wrist_svm.append(sum(d["Wrist_SVM"]))
                ankle_svm.append(sum(d["Ankle_SVM"]))
                timestamps.append(start)
                events.append(bout.Event)

                into_bout = (start - bout.Start).total_seconds()

                if start <= bout.Start:
                    into_bouts.append(into_bout)
                    desc.append("Pre")
                if bout.Start < start <= bout.Stop:
                    into_bouts.append(into_bout)
                    desc.append("Bout")
                if start > bout.Stop:
                    into_bouts.append(round((start-bout.Stop).total_seconds(), 1))
                    desc.append("Recovery")

        epoch = pd.DataFrame(list(zip(timestamps,
                                      [round(i, 2) for i in wrist_svm],
                                      [round(i, 2) for i in ankle_svm],
                                      [round(i, 1) if i is not None else None for i in epoch_hr],
                                      desc, into_bouts,
                                      events)),
                             columns=["Timestamp", "Wrist_SVM", "Ankle_SVM", "HR", "Timing",
                                      "Time_into_bout", "Activity"])

        self.df_active = epoch[["Timestamp", "Activity", "Timing", "Time_into_bout",
                                "Wrist_SVM", "Ankle_SVM", "HR"]]

    def calculate_accel_intensity(self):
        """Calculates wrist and ankle intensity using Duncan et al. (2019) cutpoints."""

        wrist = []
        ankle = []

        for epoch in self.df_active.itertuples():
            if epoch.Wrist_SVM < self.cutpoints["NDWrist_Light"]:
                wrist.append("Sedentary")
            if self.cutpoints["NDWrist_Light"] <= epoch.Wrist_SVM < self.cutpoints["NDWrist_MVPA"]:
                wrist.append("Light")
            if self.cutpoints["NDWrist_MVPA"] <= epoch.Wrist_SVM:
                wrist.append("Moderate")

            if epoch.Ankle_SVM < self.cutpoints["Ankle_Light"]:
                ankle.append("Sedentary")
            if self.cutpoints["Ankle_Light"] <= epoch.Ankle_SVM < self.cutpoints["Ankle_MVPA"]:
                ankle.append("Light")
            if self.cutpoints["Ankle_MVPA"] <= epoch.Ankle_SVM:
                ankle.append("Moderate")

        self.df_active["Wrist_Intensity"] = wrist
        self.df_active["Ankle_Intensity"] = ankle

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
            print("\n-Setting resting HR to {} bpm.".format(rest_hr))
            self.rest_hr = rest_hr

        if rest_hr is None:

            # Runs QC algorithm and calculates epoch HRs --------------------------------------------------------------
            print("\nRunning ECG quality check algorithm on whole data file to find resting HR...")

            markers = np.arange(0, len(self.ecg.raw) * 1.1, len(self.ecg.raw)/10)
            marker_ind = 0

            epoch_hr = []
            for i in range(0, len(self.ecg.raw), int(self.ecg.sample_rate * self.epoch_len)):
                qc = ECG.CheckQuality(raw_data=self.ecg.raw, start_index=i, template_data='filtered',
                                      voltage_thresh=500, epoch_len=self.epoch_len, sample_rate=self.ecg.sample_rate)

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
                                                  [round(i, 1) if i is not None else None for i in epoch_hr])),
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
                df_sleep = pd.read_excel("/Users/kyleweber/Desktop/Student Supervision/Kin 472 - Megan/Sleep.xlsx")
                df_sleep = df_sleep.loc[df_sleep["Subject"] == subj_id]

                sleep_ind = [int((row.Sleep - self.wrist.timestamps[0]).total_seconds() / self.epoch_len) for row in
                             df_sleep.itertuples()]
                wake_ind = [int((row.Wake - self.wrist.timestamps[0]).total_seconds() / self.epoch_len) for row in
                            df_sleep.itertuples()]

                sleep_list = np.zeros(int(len(self.wrist.timestamps) / self.wrist.sample_rate / self.epoch_len))

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

        # HR during activities ----------------------------------------------------------------------------------------
        hrr_list = [round((hr - self.rest_hr) / hrr * 100, 1) if not np.isnan(hr) else None
                    for hr in self.df_active["HR"]]
        self.df_active["%HRR"] = hrr_list

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

        self.df_active["HRR_Intensity"] = hrr_intensity

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
            df_hr = self.df_active
        if self.epoch_hr.shape[0] > 0:
            df_hr = self.epoch_hr

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 6))

        ax1.plot(self.df_svm["Timestamp"], self.df_svm["Wrist_SVM"], color='black')
        ax1.set_title("Wrist")
        ax1.set_ylabel("Counts")

        ax2.plot(self.df_svm["Timestamp"], self.df_svm["Ankle_SVM"], color='black')
        ax2.set_title("Ankle")
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

        for row in self.df_events.itertuples():
            # Shades activities in green
            ax1.fill_between(x=[row.Start, row.Stop], y1=0, y2=max(self.df_svm["Wrist_SVM"]),
                             color='green', alpha=.5)
            ax2.fill_between(x=[row.Start, row.Stop], y1=0, y2=max(self.df_svm["Ankle_SVM"]),
                             color='green', alpha=.5)
            ax3.fill_between(x=[row.Start, row.Stop],
                             y1=0, y2=max([i for i in df_hr["%HRR"] if not np.isnan(i)]), color='green', alpha=.5)

            # Shades padded area in red
            ax1.fill_between(x=[row.Start + timedelta(seconds=-self.pad_pre), row.Start],
                             y1=0, y2=max(self.df_svm["Wrist_SVM"]), color='grey', alpha=.5)
            ax1.fill_between(x=[row.Stop, row.Stop + timedelta(seconds=self.pad_post)],
                             y1=0, y2=max(self.df_svm["Wrist_SVM"]), color='orange', alpha=.5)

            ax2.fill_between(x=[row.Start + timedelta(seconds=-self.pad_pre), row.Start],
                             y1=0, y2=max(self.df_svm["Ankle_SVM"]), color='grey', alpha=.5)
            ax2.fill_between(x=[row.Stop, row.Stop + timedelta(seconds=self.pad_post)],
                             y1=0, y2=max(self.df_svm["Ankle_SVM"]), color='orange', alpha=.5)

            ax3.fill_between(x=[row.Start + timedelta(seconds=-self.pad_pre), row.Start],
                             y1=0, y2=max([i for i in df_hr["%HRR"] if not np.isnan(i)]), color='grey', alpha=.5)
            ax3.fill_between(x=[row.Stop, row.Stop + timedelta(seconds=self.pad_post)],
                             y1=0, y2=max([i for i in df_hr["%HRR"] if not np.isnan(i)]), color='orange', alpha=.5)

        if start is None and stop is None:
            pass
        if start is not None and stop is not None:
            ax4.set_xlim(start, stop)

            ax1.set_ylim(ax1.get_ylim()[0], max(self.df_svm.loc[(self.df_svm["Timestamp"] >= start) &
                                                                (self.df_svm["Timestamp"] < stop)]["Wrist_SVM"])*1.1)
            ax2.set_ylim(ax2.get_ylim()[0], max(self.df_svm.loc[(self.df_svm["Timestamp"] >= start) &
                                                                (self.df_svm["Timestamp"] < stop)]["Ankle_SVM"])*1.1)

        if save_image:
            plt.savefig(image_path)

    def generate_activity_images(self, image_path="/Users/kyleweber/Desktop/Student Supervision/{}_Event{}_{}.png"):
        """Loops through long walks and calls self.plot_longwalk_data for specified region of data. Saves images.

        :argument
        -image_path: pathway where image(s) are saved. Include {} for walk index.
        """

        for row in self.df_events.itertuples():
            plt.close('all')
            self.plot_all_data(start=row.Start + timedelta(seconds=-1.1 * self.pad_pre),
                               stop=row.Stop + timedelta(seconds=1.1 * self.pad_post),
                               save_image=True,
                               image_path=image_path.format(subj_id, row.Index + 1, row.Event))
            plt.close('all')

    def save_data(self, pathway=""):
        """Saves relevant data to Excel files.

            :argument
            -pathway: where files are saved
        """

        print("\nSaving relevant data to {}..".format(pathway))

        self.df_active.to_excel(pathway + "{}_Activity_Data.xlsx".format(subj_id), index=False)


class DataVisualizer:

    def __init__(self, output_folder):

        self.output_folder = output_folder
        self.timestamps = []
        self.ax1 = None

        self.gen_plot()

    def gen_plot(self):

        fig, (self.ax1, ax2) = plt.subplots(2, figsize=(10, 6), sharex='col')
        plt.subplots_adjust(bottom=.15)

        self.ax1.plot(x.df_svm["Timestamp"], x.df_svm["Wrist_SVM"], color='dodgerblue', label='Wrist')
        self.ax1.set_ylabel("SVM")

        ax2.plot(x.df_svm["Timestamp"], x.df_svm["Ankle_SVM"], color='red', label='Ankle')
        ax2.set_ylabel("SVM")

        xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
        ax2.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

    def get_timestamps(self, event):

        start = mdates.num2date(self.ax1.get_xlim()[0]).strftime("%Y-%m-%d %H:%M:%S")
        stop = mdates.num2date(self.ax1.get_xlim()[1]).strftime("%Y-%m-%d %H:%M:%S")
        self.timestamps.append([start, stop])

        print("Added {} and {} to start/stop list.".format(start, stop))

    def save_data(self, event):

        df = pd.DataFrame(np.array(self.timestamps), columns=["Start", "Stop"])
        df.to_excel(self.output_folder, index=True)

        print("\nSaving data to {}".format(self.output_folder))

        plt.close('all')


viz = DataVisualizer(output_folder="/Users/kyleweber/Desktop/Test.xlsx")
ax_store = plt.axes([.9, .075, .07, .05])
store_button = Button(ax_store, "Add")
store_button.on_clicked(viz.get_timestamps)

ax_save = plt.axes([.9, .02, .07, .05])
save_button = Button(ax_save, "Done")
save_button.on_clicked(viz.save_data)

plt.show()


subj_id = 3034
x = Data(wrist_file="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_GA_LWrist_Accelerometer.EDF".format(str(subj_id)),
         ankle_file="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_GA_LAnkle_Accelerometer.EDF".format(str(subj_id)),
         ecg_file="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_BF.EDF".format(str(subj_id)),
         activity_log="/Users/kyleweber/Desktop/Student Supervision/Kin 472 - Megan/3034_LongWalks_Log.xlsx",
         epoch_len=15)


x.check_sync()
x.import_data()
x.import_activity_log()
x.epoch1s_accels()
x.scale_cutpoints()

x.process_activity_bouts(pad_pre=15, pad_post=300)
x.calculate_accel_intensity()

# x.find_resting_hr(rest_hr=57.8, window_size=60, n_windows=30)
x.find_resting_hr(rest_hr=None, window_size=60, n_windows=30,
                  sleep_log="/Users/kyleweber/Desktop/Student Superivision/Kin 472 - Megan/Sleep.xlsx")
x.calculate_hrr(age=34)

# x.plot_all_data(save_image=False)
x.generate_activity_images(image_path="/Users/kyleweber/Desktop/Student Supervision/Kin 472 - Megan/{}_Event{}_{}.png")
x.save_data(pathway="/Users/kyleweber/Desktop/Student Supervision/Kin 472 - Megan/")
