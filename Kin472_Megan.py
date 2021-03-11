import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import ECG
import ImportEDF
from datetime import timedelta
import matplotlib.dates as mdates
import os
from OndriAtHome.BintoEDFConversion.ga_to_edf import ga_to_edf
import datetime
from Filtering import filter_signal

collection = "1"
activity = "Run"
data_folder = "/Users/kyleweber/Desktop/Student Supervision/Kin 472 - Megan/Data/Converted/"
save_folder = data_folder
log_file = data_folder + "Collection {}/EventLog{}_{}.xlsx".format(collection, collection, activity)
wrist_edf = data_folder + "Collection {}/{}_GENEActiv_Accelerometer_0{}_LW.edf".format(collection, activity, collection)
ankle_edf = data_folder + "Collection {}/{}_GENEActiv_Accelerometer_0{}_LA.edf".format(collection, activity, collection)
lead_file = data_folder + "Collection {}/3Lead{}{}.edf".format(collection, activity, collection)
ff_file = data_folder + "Collection {}/FastFix{}{}.edf".format(collection, activity, collection)


class Data:

    def __init__(self, wrist_file=None, ankle_file=None, ecg_3lead_file=None, ecg_ff_file=None,
                 activity_log=None, epoch_len=15, pad_pre=0, pad_post=0):

        self.wrist_file = wrist_file
        self.ankle_file = ankle_file
        self.ecg_lead_file = ecg_3lead_file
        self.ecg_ff_file = ecg_ff_file
        self.activity_file = activity_log
        self.epoch_len = epoch_len
        self.accel_fs = 75

        self.pad_pre = pad_pre
        self.pad_post = pad_post

        self.crop_dict = None
        self.wrist = None
        self.ankle = None
        self.ecg_lead = None
        self.ecg_ff = None

        self.df_svm = None
        self.epoch_hr = pd.DataFrame(list(zip([], [], [], [], [], [], [])),
                                     columns=["Timestamp", "HR_Lead", "%HRR_Lead", "HRR_Intensity_Lead",
                                              "HR_FF", "%HRR_Lead", "HRR_Intensity_FF"])

        self.rest_hr = 0

        self.cutpoints = None

        self.df_active = None
        self.df_events = None

    def check_sync(self):
        """Checks start times for all given files. Makes sure all devices start at same time."""

        lw_crop_index = 0
        rw_crop_index = 0
        lead_crop_index = 0
        ff_crop_index = 0

        wrist_starttime, wrist_endtime, wrist_fs, wrist_duration = ImportEDF.check_file(filepath=self.wrist_file,
                                                                                        print_summary=False)

        ankle_starttime, ankle_endtime, ankle_fs, ankle_duration = ImportEDF.check_file(filepath=self.ankle_file,
                                                                                        print_summary=False)

        l_starttime, l_endtime, l_fs, l_duration = ImportEDF.check_file(filepath=self.ecg_lead_file,
                                                                        print_summary=False)

        ff_starttime, ff_endtime, ff_fs, ff_duration = ImportEDF.check_file(filepath=self.ecg_ff_file,
                                                                            print_summary=False)

        crop_time = max([i for i in [wrist_starttime, ankle_starttime, l_starttime, ff_starttime] if i is not None])

        if wrist_starttime is not None:
            if wrist_starttime < crop_time:
                lw_crop_index = int((crop_time - wrist_starttime).total_seconds() * wrist_fs)
        if ankle_starttime is not None:
            if ankle_starttime < crop_time:
                rw_crop_index = int((crop_time - ankle_starttime).total_seconds() * ankle_fs)
        if l_starttime is not None:
            if l_starttime < crop_time:
                lead_crop_index = int((crop_time - l_starttime).total_seconds() * l_fs)
        if ff_starttime is not None:
            if ff_starttime < crop_time:
                ff_crop_index = int((crop_time - ff_starttime).total_seconds() * ff_fs)

        self.crop_dict = {"LW": lw_crop_index, "RW": rw_crop_index,
                          "ECG_Lead": lead_crop_index, "ECG_FF": ff_crop_index}

        if wrist_fs is not None and ankle_fs is not None:
            if wrist_fs != ankle_fs:
                print("\n-Accelerometer sampling rates do not match. Errors will ensue.")
            if wrist_fs == ankle_fs:
                self.accel_fs = wrist_fs

    def import_data(self):
        """Imports wrist and ECG data"""

        if self.wrist_file is not None:
            self.wrist = ImportEDF.GENEActiv(filepath=self.wrist_file,
                                             load_raw=True, start_offset=self.crop_dict["LW"])
        if self.ankle_file is not None:
            self.ankle = ImportEDF.GENEActiv(filepath=self.ankle_file,
                                             load_raw=True, start_offset=self.crop_dict["RW"])
        if self.ecg_lead_file is not None:
            self.ecg_lead = ImportEDF.Bittium(filepath=self.ecg_lead_file, load_accel=False,
                                              start_offset=self.crop_dict["ECG_Lead"],
                                              epoch_len=self.epoch_len)
            self.ecg_lead.electrode_type = "Three Lead"

        if self.ecg_ff_file is not None:
            self.ecg_ff = ImportEDF.Bittium(filepath=self.ecg_ff_file, load_accel=False,
                                            start_offset=self.crop_dict["ECG_FF"],
                                            epoch_len=self.epoch_len)
            self.ecg_ff.electrode_type = "FastFix"

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
        """Imports activity log.
        Input timestamps are HH:MM:SS into collection period -> converts to real timestamps if necessary OR
        actual timestamps (no conversion)
        """

        print("\nImporting activity log...")
        self.df_events = pd.read_excel(self.activity_file)

        if self.df_events.shape[0] != 0:
            if type(self.df_events["Start"].iloc[0]) is datetime.time:
                start_stamp = self.df_svm['Timestamp'].iloc[0]

                self.df_events["Start"] = [str(i) for i in self.df_events["Start"]]
                self.df_events["Stop"] = [str(i) for i in self.df_events["Stop"]]

                starts = []
                stops = []
                for row in self.df_events.itertuples():
                    start = start_stamp + \
                            timedelta(hours=int(row.Start.split(":")[0])) + \
                            timedelta(minutes=int(row.Start.split(":")[1])) + \
                            timedelta(seconds=int(row.Start.split(":")[2]))
                    starts.append(start)

                    stop = start_stamp + \
                           timedelta(hours=int(row.Stop.split(":")[0])) + \
                           timedelta(minutes=int(row.Stop.split(":")[1])) + \
                           timedelta(seconds=int(row.Stop.split(":")[2]))
                    stops.append(stop)

                self.df_events["Start"] = pd.to_datetime(starts)
                self.df_events["Stop"] = pd.to_datetime(stops)

            if type(self.df_events["Start"].iloc[0]) is not datetime.date:
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
        epoch_hr_lead = []
        epoch_hr_ff = []
        desc = []

        for bout in self.df_events.itertuples():

            """3-lead ECG"""
            bout_start = int((bout.Start + timedelta(seconds=-pad_pre) -
                              self.ecg_lead.timestamps[0]).total_seconds() * self.ecg_lead.sample_rate)

            bout_end = int((bout.Stop + timedelta(seconds=pad_post) -
                            self.ecg_lead.timestamps[0]).total_seconds() * self.ecg_lead.sample_rate)

            if bout.Start + timedelta(seconds=-pad_pre) <= self.ecg_lead.timestamps[0]:
                bout_start = int((bout.Start - self.ecg_lead.timestamps[0]).total_seconds() * self.ecg_lead.sample_rate)
            if bout.Stop + timedelta(seconds=pad_post) >= self.ecg_lead.timestamps[-1]:
                bout_end = int((bout.Stop - self.ecg_lead.timestamps[0]).total_seconds() * self.ecg_lead.sample_rate)

            for i in range(bout_start, bout_end, int(self.ecg_lead.sample_rate * self.epoch_len)):
                qc = ECG.CheckQuality(raw_data=self.ecg_lead.raw, start_index=i, template_data='filtered',
                                      voltage_thresh=250, epoch_len=self.epoch_len,
                                      sample_rate=self.ecg_lead.sample_rate)

                if qc.valid_period:
                    epoch_hr_lead.append(qc.hr)
                if not qc.valid_period:
                    epoch_hr_lead.append(None)

            """FastFix ECG"""
            bout_start = int((bout.Start + timedelta(seconds=-pad_pre) -
                              self.ecg_ff.timestamps[0]).total_seconds() * self.ecg_ff.sample_rate)

            bout_end = int((bout.Stop + timedelta(seconds=pad_post) -
                            self.ecg_ff.timestamps[0]).total_seconds() * self.ecg_ff.sample_rate)

            if bout.Stop + timedelta(seconds=pad_post) >= self.ecg_ff.timestamps[-1]:
                bout_end = len(self.ecg_ff.timestamps)

            for i in range(bout_start, bout_end, int(self.ecg_ff.sample_rate * self.epoch_len)):
                qc = ECG.CheckQuality(raw_data=self.ecg_ff.raw, start_index=i, template_data='filtered',
                                      voltage_thresh=250, epoch_len=self.epoch_len,
                                      sample_rate=self.ecg_ff.sample_rate)

                if qc.valid_period:
                    epoch_hr_ff.append(qc.hr)
                if not qc.valid_period:
                    epoch_hr_ff.append(None)

            df = self.df_svm.loc[(self.df_svm["Timestamp"] >= bout.Start + timedelta(seconds=-pad_pre)) &
                                 (self.df_svm["Timestamp"] < bout.Stop + timedelta(seconds=pad_post))]

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
                                      [round(i, 1) if i is not None else None for i in epoch_hr_lead],
                                      [round(i, 1) if i is not None else None for i in epoch_hr_ff],
                                      desc, into_bouts, events)),
                             columns=["Timestamp", "Wrist_SVM", "Ankle_SVM", "HR_Lead", "HR_FF", "Timing",
                                      "Time_into_bout", "Activity"])

        self.df_active = epoch[["Timestamp", "Activity", "Timing", "Time_into_bout",
                                "Wrist_SVM", "Ankle_SVM", "HR_Lead", "HR_FF"]]

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

    def find_resting_hr(self, rest_hr=None, ecg_type="lead"):
        """Function that calculates resting HR based on inputs.
           Able to input a resting HR value to skip ECG processing.

        :argument
        -rest_hr: int/float if already calculated; will skip ECG processing
            -If None, will perform ECG processing
        """

        if ecg_type.capitalize() == "Lead":
            ecg = self.ecg_lead
            ecg_name = "_Lead"
        if ecg_type == "ff" or ecg_type == "fastfix":
            ecg = self.ecg_ff
            ecg_name = "_FF"

        def hr_from_raw(start_ind=0, stop_ind=None):

            # Runs QC algorithm and calculates epoch HRs ----------------------------------------------------------
            print("\nRunning ECG quality check algorithm to find resting HR...")

            if stop_ind == 0:
                stop_ind = len(ecg.raw)

            markers = np.arange(start_ind, stop_ind * 1.1, (stop_ind - start_ind) / 10)
            marker_ind = 0

            epoch_hr = []
            for i in range(start_ind, stop_ind, int(ecg.sample_rate * self.epoch_len)):
                qc = ECG.CheckQuality(raw_data=ecg.raw, start_index=i, template_data='filtered',
                                      voltage_thresh=250, epoch_len=self.epoch_len,
                                      sample_rate=ecg.sample_rate)

                if i >= markers[marker_ind]:
                    print("{}% complete".format(marker_ind * 10))
                    marker_ind += 1

                if qc.valid_period:
                    epoch_hr.append(qc.hr)
                if not qc.valid_period:
                    epoch_hr.append(None)

            print("100% complete")

            if self.epoch_hr is None:
                self.epoch_hr = pd.DataFrame(list(zip(pd.date_range(start=ecg.timestamps[0],
                                                                    end=ecg.timestamps[-1],
                                                                    freq="{}S".format(self.epoch_len)),
                                                      [round(i, 1) if i is not None else None for i in epoch_hr])),
                                             columns=["Timestamp", "HR" + ecg_name])

            if self.epoch_hr is not None:
                self.epoch_hr["HR" + ecg_name] = [round(i, 1) if i is not None else None for i in epoch_hr]

            valid_hr = [i for i in epoch_hr if i is not None]

            resting_hr = round(sum(valid_hr) / len(valid_hr), 1)

            print("Average HR from designated period is {} bpm.".format(resting_hr))

            self.rest_hr = resting_hr

        if rest_hr is not None:
            print("\n-Setting resting HR to {} bpm.".format(rest_hr))
            self.rest_hr = rest_hr

        if rest_hr is None:
            if "Resting HR" not in [i for i in self.df_events["Event"]]:
                print("\nNo resting HR event found in event file.")
                print("Please ensure there is an event called 'Resting HR' and try again.")
                return None

            if "Resting HR" in [i for i in self.df_events["Event"]]:
                start_index = int((self.df_events[self.df_events["Event"] == "Resting HR"]["Start"].iloc[0] -
                                   ecg.timestamps[0]).total_seconds() * ecg.sample_rate)
                stop_index = int((self.df_events[self.df_events["Event"] == "Resting HR"]["Stop"].iloc[0] -
                                  ecg.timestamps[0]).total_seconds() * ecg.sample_rate)

                hr_from_raw(start_ind=start_index, stop_ind=stop_index)

    def calculate_hrr(self, age=30, ecg_type="lead"):
        """Uses predicted max HR and measured resting HR to quantify %HRR data.

            :argument
            -age: participant age in years
        """

        if ecg_type.capitalize() == "Lead":
            ecg = "_Lead"
        if ecg_type == "ff":
            ecg = "_FF"

        print("\nCalculating %HRR data (Resting HR = {}, age = {} years)...".format(self.rest_hr, age))

        hrr = (208 - .7 * age - self.rest_hr)

        # HR during activities ----------------------------------------------------------------------------------------
        hrr_list = [round((hr - self.rest_hr) / hrr * 100, 1) if not np.isnan(hr) else None
                    for hr in self.df_active["HR" + ecg]]

        hrr_list2 = []
        for i in hrr_list:
            if i is None:
                hrr_list2.append(None)
            if i is not None:
                if i < 0:
                    hrr_list2.append(0)
                if i >= 0:
                    hrr_list2.append(i)

        hrr_list = hrr_list2

        self.df_active["%HRR" + ecg] = hrr_list

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

        self.df_active["HRR_Intensity" + ecg] = hrr_intensity

        # HR during all data ------------------------------------------------------------------------------------------
        hrr_list = [round((hr - self.rest_hr) / hrr * 100, 1) if not np.isnan(hr) else None
                    for hr in self.epoch_hr["HR" + ecg]]
        self.epoch_hr["%HRR" + ecg] = hrr_list

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

        self.epoch_hr["HRR_Intensity" + ecg] = hrr_intensity

        print("Complete.")

    def plot_all_data(self, start=None, stop=None, save_image=False, image_path=None, hr_type='HR'):
        """Able to plot specified sections of data. Shades long walking bouts. Able to save.

            :argument
            -start/stop: timestamp or None to crop data
            -save_image: boolean. If True, saves image to path specified by image_path
            -image_path: save location and filename of image
            -hr_type: "HR" or "%HRR"
        """

        n_plots = len([i for i in [self.wrist_file, self.ankle_file] if i is not None]) + \
                  1 if not None in [i for i in [self.ecg_lead, self.ecg_ff] if i is not None] else 0

        fig, axes = plt.subplots(n_plots, sharex='col', figsize=(10, 6))

        if self.wrist_file is not None:
            axes[0].plot(self.df_svm["Timestamp"], self.df_svm["Wrist_SVM"], color='black')
            axes[0].set_title("Wrist")
            axes[0].set_ylabel("Counts")

        if self.ankle_file is not None:
            axes[1].plot(self.df_svm["Timestamp"], self.df_svm["Ankle_SVM"], color='black')
            axes[1].set_title("Ankle")
            axes[1].set_ylabel("Counts")

        if self.ecg_lead is not None:
            axes[2].plot(self.df_active["Timestamp"], self.df_active[hr_type + "_Lead"],
                         color='black', label="Lead", linestyle="", marker="o", markersize=4)
            axes[2].set_ylabel(hr_type)
            axes[2].set_title("Heart Rate")

            axes[2].axhline(self.rest_hr, linestyle='dashed', color='green')
            axes[2].legend()

        if self.ecg_ff is not None:
            axes[2].plot(self.df_active["Timestamp"], self.df_active[hr_type + "_FF"],
                         color='red', label="FastFix", linestyle="", marker="o", markersize=4)
            axes[2].set_ylabel(hr_type)
            axes[2].set_title("Heart Rate")

            axes[2].legend()

        xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
        axes[-1].xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=7)

        for row in self.df_events.itertuples():
            # Shades activities in green

            if self.wrist_file is not None:
                axes[0].fill_between(x=[row.Start, row.Stop], y1=0, y2=max(self.df_svm["Wrist_SVM"]),
                                     color='green', alpha=.5)
            if self.ankle_file is not None:
                axes[1].fill_between(x=[row.Start, row.Stop], y1=0, y2=max(self.df_svm["Ankle_SVM"]),
                                     color='green', alpha=.5)
            if self.ecg_lead is not None or self.ecg_ff is not None:
                axes[2].fill_between(x=[row.Start, row.Stop],
                                     y1=0, y2=max([i for i in self.df_active[hr_type + "_Lead"] if not np.isnan(i)]),
                                     color='green', alpha=.5)

            # Shades padded area in red
            if self.wrist_file is not None:
                axes[0].fill_between(x=[row.Start + timedelta(seconds=-self.pad_pre), row.Start],
                                     y1=0, y2=max(self.df_svm["Wrist_SVM"]), color='grey', alpha=.5)
                axes[0].fill_between(x=[row.Stop, row.Stop + timedelta(seconds=self.pad_post)],
                                     y1=0, y2=max(self.df_svm["Wrist_SVM"]), color='orange', alpha=.5)

            if self.ankle_file is not None:
                axes[1].fill_between(x=[row.Start + timedelta(seconds=-self.pad_pre), row.Start],
                                     y1=0, y2=max(self.df_svm["Ankle_SVM"]), color='grey', alpha=.5)
                axes[1].fill_between(x=[row.Stop, row.Stop + timedelta(seconds=self.pad_post)],
                                     y1=0, y2=max(self.df_svm["Ankle_SVM"]), color='orange', alpha=.5)

            if self.ecg_lead is not None or self.ecg_ff is not None:
                axes[2].fill_between(x=[row.Start + timedelta(seconds=-self.pad_pre), row.Start],
                                     y1=0, y2=max([i for i in self.df_active[hr_type + "_Lead"] if not np.isnan(i)]),
                                     color='grey', alpha=.5)
                axes[2].fill_between(x=[row.Stop, row.Stop + timedelta(seconds=self.pad_post)],
                                     y1=0, y2=max([i for i in self.df_active[hr_type + "_Lead"] if not np.isnan(i)]),
                                     color='orange', alpha=.5)

        if start is None and stop is None:
            pass
        if start is not None and stop is not None:
            axes[-1].set_xlim(start, stop)

            if self.wrist_file is not None:
                axes[0].set_ylim(axes[0].get_ylim()[0],
                                 max(self.df_svm.loc[(self.df_svm["Timestamp"] >= start) &
                                                     (self.df_svm["Timestamp"] < stop)]["Wrist_SVM"])*1.1)
            if self.ankle_file is not None:
                axes[1].set_ylim(axes[1].get_ylim()[0],
                                 max(self.df_svm.loc[(self.df_svm["Timestamp"] >= start) &
                                                     (self.df_svm["Timestamp"] < stop)]["Ankle_SVM"])*1.1)

        if save_image:
            plt.savefig(image_path)

    def generate_activity_images(self, image_path):
        """Loops through long walks and calls self.plot_longwalk_data for specified region of data. Saves images.

        :argument
        -image_path: pathway where image(s) are saved. Include {} for walk index.
        """

        print("\nSaving images of data for all events...")

        for row in self.df_events.itertuples():
            plt.close('all')
            self.plot_all_data(start=row.Start + timedelta(seconds=-1.1 * self.pad_pre),
                               stop=row.Stop + timedelta(seconds=1.1 * self.pad_post),
                               save_image=True,
                               image_path=image_path.format(row.Index + 1, row.Event))
            plt.close('all')

        print("Complete.")

    def save_data(self, pathway=""):
        """Saves relevant data to Excel files.

            :argument
            -pathway: where files are saved
        """

        print("\nSaving relevant data to {}..".format(pathway))

        self.df_active.to_excel(pathway + "Activity_Data.xlsx", index=False)


class DataVisualizer:

    def __init__(self, data_obj=None, epoched_file=None, output_filename=None):

        self.data = data_obj
        self.df_epoch = None

        if epoched_file is not None:
            self.df_epoch = pd.read_csv(epoched_file)
            self.df_epoch["Timestamp"] = pd.to_datetime(self.df_epoch["Timestamp"])
            self.df_epoch["Timestamp"] = pd.date_range(start=self.df_epoch["Timestamp"].iloc[0],
                                                       freq="1S", periods=self.df_epoch.shape[0])
            self.df_epoch.columns = ["Timestamp", "Wrist_SVM", "Wrist_AVM", "Ankle_SVM", "Ankle_AVM", "WristIntensity"]

        self.filename = output_filename
        self.timestamps = []
        self.ax1 = None
        self.ax2 = None

        self.gen_plot()

    def gen_plot(self):

        fig, (self.ax1, self.ax2) = plt.subplots(2, figsize=(10, 6), sharex='col')
        plt.subplots_adjust(bottom=.15)
        plt.suptitle("Use controls to show full activity on screen and hit 'Add' to save timestamps")

        if self.data is not None:
            self.ax1.plot(self.data.df_svm["Timestamp"], self.data.df_svm["Wrist_SVM"],
                          color='dodgerblue', label='Wrist')
            self.ax2.plot(self.data.df_svm["Timestamp"], self.data.df_svm["Ankle_SVM"],
                          color='red', label='Ankle')
        if self.data is None and self.df_epoch is not None:
            self.ax1.plot(self.df_epoch["Timestamp"], self.df_epoch["Wrist_SVM"],
                          color='dodgerblue', label='Wrist')
            self.ax2.plot(self.df_epoch["Timestamp"], self.df_epoch["Ankle_SVM"],
                          color='red', label='Ankle')

        self.ax1.set_ylabel("SVM")
        self.ax2.set_ylabel("SVM")

        xfmt = mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S")
        self.ax2.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

    def get_timestamps(self, event):

        start = mdates.num2date(self.ax1.get_xlim()[0]).strftime("%Y-%m-%d %H:%M:%S")
        stop = mdates.num2date(self.ax1.get_xlim()[1]).strftime("%Y-%m-%d %H:%M:%S")
        self.timestamps.append([start, stop])

        print("Added {} and {} to start/stop list.".format(start, stop))

    def save_data(self, event):

        if os.path.exists(self.filename):
            self.filename = self.filename.split(".")[0] + "Version2.xlsx"
            self.data.activity_file = self.filename

        df = pd.DataFrame(np.array(self.timestamps), columns=["Start", "Stop"])
        df.insert(loc=0, column="Event", value=["NoName{}".format(i) for i in range(df.shape[0])])
        df.to_excel(self.filename, index=False)

        print("\nSaving data to {}".format(self.filename))

        plt.close('all')


# =====================================================================================================================
# ============================================== STEP 0: DATA CONVERSION ==============================================
# =====================================================================================================================

"""
# Full pathways to all GENEActiv files. Include '.bin'.
# Filename must follow the structure used below with the underscores
wrist_bin = "/Users/kyleweber/Desktop/Student Supervision/Kin 472 - Megan/Data/HIIT_GA_002_01_LW.bin"
ankle_bin = "/Users/kyleweber/Desktop/Student Supervision/Kin 472 - Megan/Data/HIIT_GA_002_01_LA.bin"

convert_files = [wrist_bin, ankle_bin]

for file in convert_files:
    ga_to_edf(input_file_path=file,
              accelerometer_dir=save_folder,
              temperature_dir="", light_dir="",
              button_dir="", device_dir="",
              device_edf=False, correct_drift=True, quiet=False)
"""

# =====================================================================================================================
# ============================================ STEP 1: DATA VISUALIZATION =============================================
# =====================================================================================================================

# RUN THIS FIRST ---------------------------------------------------


"""
d = Data(wrist_file=wrist_edf,
         ankle_file=ankle_edf,
         ecg_3lead_file=lead_file, ecg_ff_file=ff_file,
         activity_log=log_file,
         epoch_len=15, pad_pre=15, pad_post=15)

d.check_sync()
d.import_data()
d.epoch1s_accels()
"""

"""
d = Data(wrist_file=wrist_edf,
         ankle_file=ankle_edf,
         ecg_3lead_file=lead_file, ecg_ff_file=ff_file,
         activity_log=log_file,
         epoch_len=15, pad_pre=15, pad_post=15)

viz = DataVisualizer(data_obj=d, epoched_file=None,
                     output_filename="/Users/kyleweber/Desktop/Out.xlsx")
ax_store = plt.axes([.9, .075, .07, .05])
store_button = Button(ax_store, "Add")
store_button.on_clicked(viz.get_timestamps)

ax_save = plt.axes([.9, .02, .07, .05])
save_button = Button(ax_save, "Done")
save_button.on_clicked(viz.save_data)

plt.show()
"""

# THEN RUN THIS -----------------------------------------------------


# d.import_activity_log()
# d.generate_activity_images(image_path=save_folder + "Event{}_{}.png")


# =====================================================================================================================
# ============================================== STEP 2: DATA PROCESSING ==============================================
# =====================================================================================================================


# wrist_file, ankle_file, and ecg_file are EDFs
# activity_log is an Excel file (.xlsx)
x = Data(wrist_file=wrist_edf,
         ankle_file=ankle_edf,
         ecg_3lead_file=lead_file, ecg_ff_file=ff_file,
         activity_log=log_file,
         epoch_len=15)

x.check_sync()
x.import_data()
x.epoch1s_accels()
x.import_activity_log()

# x.scale_cutpoints()

# x.process_activity_bouts(pad_pre=15, pad_post=15)
# x.calculate_accel_intensity()

# x.find_resting_hr(rest_hr=60, ecg_type="Lead")
# x.calculate_hrr(age=22, ecg_type='Lead')

# x.find_resting_hr(rest_hr=60, ecg_type="ff")
# x.calculate_hrr(age=22, ecg_type='ff')

# x.plot_all_data(x, save_image=False)
# x.generate_activity_images(save_folder + "Event{}_{}.png")
# x.save_data(pathway=save_folder)


def check_ecg_peaks(data, roll_avg=10, epoch_len=15, show_plot=True):

    # Runs QC check and epoch HR on valid epochs ---------------------------------------------------------------------
    epoch_hr = []
    for i in range(0, len(data.raw) - int(data.sample_rate * epoch_len), int(data.sample_rate * epoch_len)):
        qc = ECG.CheckQuality(raw_data=data.raw, start_index=i, template_data='filtered',
                              voltage_thresh=250, epoch_len=epoch_len,
                              sample_rate=data.sample_rate)

        if qc.valid_period:
            epoch_hr.append(qc.hr)
        if not qc.valid_period:
            epoch_hr.append(None)

    # Finds all peaks using wavelet transformation --------------------------------------------------------------------
    e = ECG.DetectAllPeaks(data=data.raw, sample_rate=data.sample_rate, algorithm="wavelet")
    e.detect_peaks()

    # Calculates beat-to-beat HR without QC check ---------------------------------------------------------------------
    rr = []
    inst_hr = []
    for p1, p2 in zip(e.r_peaks[:], e.r_peaks[1:]):
        r = (p2-p1)/data.sample_rate
        rr.append(r)
        inst_hr.append(60/r)

    # Calculates average HR in epochs without QC check ----------------------------------------------------------------
    df_inst = pd.DataFrame([[data.timestamps[i] for i in e.r_peaks[:-1]], inst_hr]).transpose()
    df_inst.columns = ["Timestamp", "InstHR"]

    event_name = []

    for row in df_inst.itertuples():
        label_found = False

        for event in x.df_events.itertuples():

            if event.Start <= row.Timestamp < event.Stop:
                event_name.append(event.Event)
                label_found = True
                break

        if not label_found:
            event_name.append("Nothing")

    df_inst["Event"] = event_name

    epochs = pd.date_range(start=df_inst.iloc[0]["Timestamp"].round("{}S".format(epoch_len)),
                           end=df_inst.iloc[-1]["Timestamp"].round("{}S".format(epoch_len)),
                           freq='{}S'.format(epoch_len))

    avg_hr = []
    for epoch1, epoch2 in zip(epochs[:], epochs[1:]):
        d = df_inst.loc[(df_inst["Timestamp"] >= epoch1) & (df_inst["Timestamp"] < epoch2)]
        avg_hr.append(d["InstHR"].mean())
    epochs = epochs[:-1]

    df_avg = pd.DataFrame([epochs, avg_hr]).transpose()
    df_avg.columns = ["Timestamp", "AvgHR"]

    event_name = []

    for row in df_avg.itertuples():
        label_found = False

        for event in x.df_events.itertuples():

            if event.Start <= row.Timestamp < event.Stop:
                event_name.append(event.Event)
                label_found = True
                break

        if not label_found:
            event_name.append("Nothing")

    df_avg["Event"] = event_name

    # Epoch (valid only) df
    df_epoch = pd.DataFrame([data.timestamps[::int(data.sample_rate * epoch_len)][:len(epoch_hr)],
                             epoch_hr]).transpose()
    df_epoch.columns = ["Timestamp", "AvgHR"]

    event_name = []

    for row in df_epoch.itertuples():
        label_found = False

        for event in x.df_events.itertuples():

            if event.Start <= row.Timestamp < event.Stop:
                event_name.append(event.Event)
                label_found = True
                break

        if not label_found:
            event_name.append("Nothing")

    df_epoch["Event"] = event_name

    # PLOTTING --------------------------------------------------------------------------------------------------------

    if show_plot:
        fig, axes = plt.subplots(3, sharex='col', figsize=(10, 7))
        plt.subplots_adjust(hspace=.33)
        plt.suptitle("{}{} - {} Data".format(activity, collection, data.electrode_type))

        axes[0].plot(data.timestamps, filter_signal(data.raw, filter_type="highpass", high_f=.25,
                                                    sample_f=data.sample_rate),
                     color='red', label="HP Filter")
        axes[0].set_title("ECG")
        axes[0].set_ylabel("Voltage", color='red')
        axes[0].tick_params(axis="y", colors='red')

        ax2 = axes[0].twinx()

        try:
            ax2.plot(data.timestamps, e.filt_squared[:len(data.timestamps)], color='black', label="Processed")
            ax2.plot([data.timestamps[i] for i in e.r_peaks], [e.filt_squared[i] for i in e.r_peaks],
                     linestyle="", marker="o", color='green', markersize=5, label="Peaks")
        except TypeError:
            pass
        ax2.set_ylabel("Meaningless unit")
        ax2.set_yticks([])
        axes[0].legend(loc='upper left')
        ax2.legend(loc='upper right')

        axes[1].set_ylabel("HR (bpm)")
        axes[1].set_title("Beat-by-beat HR")
        axes[1].plot([data.timestamps[i] for i in e.r_peaks[:-1]], inst_hr, color='dodgerblue', label="Beat-to-beat",
                     marker="s", markerfacecolor='dodgerblue', markersize=3, linewidth=2)

        axes[2].set_title("Epoched HR")

        axes[2].plot([data.timestamps[i] for i in e.r_peaks[:-roll_avg - 1]],
                     [sum(inst_hr[i:i+roll_avg])/roll_avg for i in range(len(inst_hr)-roll_avg)], color='black',
                     label="{} beat avg".format(roll_avg), marker="^", markerfacecolor='black', markersize=3)

        for i in range(len(epoch_hr)):
            axes[2].plot([data.timestamps[int(i*data.sample_rate * epoch_len)],
                          data.timestamps[int(i*data.sample_rate * epoch_len)]],
                         [epoch_hr[i], epoch_hr[i]], color='#e67300')

        axes[2].plot(data.timestamps[::int(data.sample_rate * epoch_len)][:len(epoch_hr)], epoch_hr, markersize=3,
                     color='limegreen', label="{}s epoch (valid)".format(epoch_len), marker="o", markerfacecolor="limegreen")

        axes[2].plot(df_avg["Timestamp"], df_avg["AvgHR"], markersize=4,
                     color='darkgrey', label="{}s epoch (all)".format(epoch_len), marker="x", markerfacecolor="darkgray")

        axes[2].legend()
        axes[2].set_ylabel("HR (bpm)")

        xfmt = mdates.DateFormatter("%H:%M:%S")
        axes[-1].xaxis.set_major_formatter(xfmt)

    return e, df_inst, df_avg, df_epoch


ff_ecg, ff_df_inst, ff_df_avg, ff_df_epoch = check_ecg_peaks(data=x.ecg_ff, roll_avg=10,
                                                             epoch_len=15, show_plot=False)
lead_ecg, lead_df_inst, lead_df_avg, lead_df_epoch = check_ecg_peaks(data=x.ecg_lead, roll_avg=10,
                                                                     epoch_len=15, show_plot=False)


def compare_electrodes(label_events=True):

    fig, axes = plt.subplots(3, sharex='col', figsize=(10, 7))
    plt.subplots_adjust(hspace=.3)
    plt.suptitle("{}{} - Electrode Comparison".format(activity, collection))

    axes[0].plot(ff_df_inst["Timestamp"], ff_df_inst["InstHR"], color='red', label="FF")
    axes[0].plot(lead_df_inst["Timestamp"], lead_df_inst["InstHR"], color='dodgerblue', label="3L")
    axes[0].legend()
    axes[0].set_title("Beat-by-beat HR (All Peaks)")
    axes[0].set_ylabel("HR")

    axes[1].plot(ff_df_avg.dropna()["Timestamp"], ff_df_avg.dropna()["AvgHR"], color='red', label="FF")
    axes[1].plot(lead_df_avg.dropna()["Timestamp"], lead_df_avg.dropna()["AvgHR"], color='dodgerblue', label="3L")
    axes[1].set_title("Rolling Avg (Valid/Invalid)")
    axes[1].set_ylabel("HR")

    axes[2].plot(ff_df_epoch.dropna()["Timestamp"], ff_df_epoch.dropna()["AvgHR"],
                 color='red', marker="o", markersize=4)
    axes[2].plot(lead_df_epoch.dropna()["Timestamp"], lead_df_epoch.dropna()["AvgHR"],
                 color='dodgerblue', marker="s", markersize=3)
    axes[2].set_title("Epoch HR (Valid Only)")
    axes[2].set_ylabel("HR")

    xfmt = mdates.DateFormatter("%H:%M:%S")
    axes[-1].xaxis.set_major_formatter(xfmt)

    ylims0 = axes[0].get_ylim()
    ylims1 = axes[1].get_ylim()
    ylims2 = axes[2].get_ylim()

    for row in x.df_events.itertuples():

        axes[0].fill_between(x=[row.Start, row.Stop], y1=ylims0[0], y2=ylims0[1],
                             color='dimgrey' if row.Index % 2 == 0 else 'darkgray', alpha=.5)
        axes[1].fill_between(x=[row.Start, row.Stop], y1=ylims1[0], y2=ylims1[1],
                             color='dimgrey' if row.Index % 2 == 0 else 'darkgray', alpha=.5)
        axes[2].fill_between(x=[row.Start, row.Stop], y1=ylims2[0], y2=ylims2[1],
                             color='dimgrey' if row.Index % 2 == 0 else 'darkgray', alpha=.5)

        if label_events:
            axes[0].text(x=row.Start + (row.Stop - row.Start) / 10,
                         y=ylims0[0]*.9, s=row.Event)


compare_electrodes(label_events=False)


# TODO
# Figure out how to deal with bad data
# Processing events
