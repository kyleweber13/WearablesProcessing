import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ECG
import ImportEDF


class Data:

    def __init__(self, lw_file=None, rw_file=None, ecg_file=None, gait_log=None,
                 epoch_len=15, gait_thresh_min=10, dom_hand="Right"):

        self.lw_file = lw_file
        self.rw_file = rw_file
        self.ecg_file = ecg_file
        self.gaitlog_file = gait_log
        self.epoch_len = epoch_len
        self.accel_fs = 75
        self.gait_thresh = gait_thresh_min
        self.dom_hand = dom_hand.capitalize()

        self.crop_dict = None
        self.lw = None
        self.rw = None
        self.ecg = None

        self.df_svm = None

        self.gait_log = None
        self.gait_log_long = None

        self.cutpoints = None

        self.df_walks = None

    def check_sync(self):

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

        self.lw = ImportEDF.GENEActiv(filepath=self.lw_file, load_raw=True, start_offset=self.crop_dict["LW"])
        self.rw = ImportEDF.GENEActiv(filepath=self.rw_file, load_raw=True, start_offset=self.crop_dict["RW"])
        self.ecg = ImportEDF.Bittium(filepath=self.ecg_file, load_accel=False, start_offset=self.crop_dict["ECG"],
                                     epoch_len=self.epoch_len)

    def scale_cutpoints(self):

        nd_light = 255 * self.accel_fs / 100 * self.epoch_len / 60
        nd_mod = 588 * self.accel_fs / 100 * self.epoch_len / 60

        d_light = 375 * self.accel_fs / 100 * self.epoch_len / 60
        d_mod = 555 * self.accel_fs / 100 * self.epoch_len / 60

        self.cutpoints = {"ND_Light": nd_light, "ND_Mod": nd_mod, "D_Light": d_light, "D_Mod": d_mod}

    def epoch1s_accels(self):

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

    def import_gait_log(self, threshold):

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

    def process_gait_bouts(self):

        print("\nProcessing gait bouts longer than {} minutes...".format(self.gait_thresh))

        lw_svm = []
        rw_svm = []
        walk_nums = []
        timestamps = []
        into_bouts = []
        epoch_hr = []

        for bout in self.gait_log_long.itertuples():

            bout_start = int((bout.start_timestamp - self.ecg.timestamps[0]).total_seconds() * self.ecg.sample_rate)
            bout_end = int((bout.end_timestamp - self.ecg.timestamps[0]).total_seconds() * self.ecg.sample_rate)

            for i in range(bout_start, bout_end, int(self.ecg.sample_rate * self.epoch_len)):
                qc = ECG.CheckQuality(ecg_object=self.ecg, start_index=i, template_data='filtered',
                                      voltage_thresh=250, epoch_len=self.epoch_len)

                if qc.valid_period:
                    epoch_hr.append(qc.hr)
                if not qc.valid_period:
                    epoch_hr.append(None)

            df = self.df_svm.loc[(self.df_svm["Timestamp"] >= bout.start_timestamp) &
                                 (self.df_svm["Timestamp"] <= bout.end_timestamp)]

            stamps = pd.date_range(start=bout.start_timestamp,
                                   end=bout.end_timestamp,
                                   freq="{}S".format(self.epoch_len))

            for start, end in zip(stamps[:], stamps[1:]):
                d = df.loc[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]
                lw_svm.append(sum(d["LW_SVM"]))
                rw_svm.append(sum(d["RW_SVM"]))
                walk_nums.append(bout.Index + 1)
                timestamps.append(start)
                into_bouts.append((start - bout.start_timestamp).total_seconds())

        epoch = pd.DataFrame(list(zip(timestamps, walk_nums, lw_svm, rw_svm, epoch_hr, into_bouts)),
                             columns=["Timestamp", "Walk_num", "LW_SVM", "RW_SVM", "HR", "Time_into_bout"])

        self.df_walks = epoch

    def calculate_wrist_intensity(self):

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

    def calculate_hrr(self, rest_hr=60, age=30):

        hrr = (208 - .7 * age - rest_hr)

        hrr_list = [(hr - rest_hr) / hrr * 100 if hr is not None else None for hr in self.df_walks["HR"]]
        self.df_walks["%HRR"] = hrr_list

        hrr_intensity = []

        for hr in hrr_list:
            if np.isnan(hr):
                hrr_intensity.append(None)
            if hr is not None:
                if hr < 30:
                    hrr_intensity.append("Sedentary")
                if 30 <= hr < 40:
                    hrr_intensity.append("Light")
                if hr >= 40:
                    hrr_intensity.append("Moderate")

        self.df_walks["HRR_Intensity"] = hrr_intensity


x = Data(lw_file="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_3034_01_GA_LWrist_Accelerometer.EDF",
         rw_file="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_3034_01_GA_LAnkle_Accelerometer.EDF",
         ecg_file="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_3034_01_BF.EDF",
         gait_log="/Users/kyleweber/Desktop/summary_data OND07 WTL 3034 sample.csv",
         epoch_len=15, gait_thresh_min=3)
x.check_sync()
x.import_data()
x.epoch1s_accels()
x.scale_cutpoints()
x.import_gait_log(threshold=x.gait_thresh)
x.process_gait_bouts()
x.calculate_wrist_intensity()
x.calculate_hrr(rest_hr=60, age=30)

# TODO
# Calculate real resting HR
