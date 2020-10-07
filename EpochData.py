from datetime import datetime
import math
import pandas as pd


class EpochAccel:

    def __init__(self, raw_data=None, raw_filename=None, proc_filepath=None, accel_type=None,
                 remove_baseline=False, from_processed=True,
                 processed_folder=None, accel_only=False, epoch_len=15):

        self.epoch_len = epoch_len
        self.remove_baseline = remove_baseline
        self.from_processed = from_processed
        self.accel_only = accel_only
        self.processed_folder = processed_folder
        self.raw_filename = raw_filename
        self.proc_filepath = proc_filepath

        if self.proc_filepath is not None:
            self.proc_filename = proc_filepath.split("/")[-1]
        if self.proc_filepath is None:
            self.proc_filename = self.raw_filename + "_IntensityData.csv"

        self.svm = []
        self.timestamps = None

        # GENEActiv: ankle only
        self.pred_speed = None
        self.pred_mets = None
        self.intensity_cat = None

        # Epoching from raw data
        if not self.from_processed and raw_data is not None:
            self.epoch_from_raw(raw_data=raw_data)

        # Loads epoched data from existing file
        if self.from_processed and accel_type == "wrist":
            self.epoch_from_processed_wrist()
        if self.from_processed and accel_type == "ankle":
            self.epoch_from_processed_ankle()

        # Removes bias from SVM by subtracting minimum value
        if self.remove_baseline and min(self.svm) != 0.0:
            print("\n" + "Removing bias from SVM calculations...")
            self.svm = [i - min(self.svm) for i in self.svm]
            print("Complete. Bias removed.")

    def epoch_from_raw(self, raw_data):
        """Epochs accelerometer data into specified epoch length using raw data."""

        # Calculates epochs if from_processed is False
        print("\n" + "Epoching using raw data...")

        self.timestamps = raw_data.timestamps[::self.epoch_len * raw_data.sample_rate]

        # Calculates gravity-subtracted vector magnitude
        raw_data.vm = [round(abs(math.sqrt(math.pow(raw_data.x[i], 2) + math.pow(raw_data.y[i], 2) +
                                           math.pow(raw_data.z[i], 2)) - 1), 5) for i in range(len(raw_data.x))]

        # Calculates activity counts
        for i in range(0, len(raw_data.vm), int(raw_data.sample_rate * self.epoch_len)):

            if i + self.epoch_len * raw_data.sample_rate > len(raw_data.vm):
                break

            vm_sum = sum(raw_data.vm[i:i + self.epoch_len * raw_data.sample_rate])

            # Bug handling: when we combine multiple EDF files they are zero-padded
            # When vector magnitude is calculated, it is 1
            # Any epoch where the values were all the epoch length * sampling rate (i.e. a VM of 1 for each data point)
            # becomes 0
            if vm_sum == self.epoch_len * raw_data.sample_rate:
                vm_sum = 0

            self.svm.append(round(vm_sum, 5))

        print("Epoching complete.")

    def epoch_from_processed(self):

        print("\n" + "Importing processed accelerometer from {}.".format(self.processed_folder))

        # Wrist accelerometer ----------------------------------------------------------------------------------------
        if "Wrist" in self.proc_filename:

            # Reads in correct file
            if self.proc_filename.split(".")[-1] == "csv":
                df_wrist = pd.read_csv(self.proc_filepath)

            if self.proc_filename.split(".")[-1] == "xlsx":
                df_wrist = pd.read_excel(self.proc_filepath)

            # Timestamp formatting: files have been generated using different packages over the project
            if type(df_wrist["Timestamp"].iloc[0]) is str:
                self.timestamps = pd.to_datetime(df_wrist["Timestamp"], format="%Y-%m-%d %H:%M:%S.%f")

            if type(df_wrist["Timestamp"].iloc[0]) is not str:
                self.timestamps = [datetime.strftime(i, "%Y-%m-%d %H:%M:%S.%f") for i in df_wrist["Timestamp"]]
                self.timestamps = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f") for i in self.timestamps]

            self.epoch_len = (self.timestamps[1] - self.timestamps[0]).seconds
            self.svm = [round(float(i), 2) for i in df_wrist["ActivityCount"]]

        # Ankle accelerometer ----------------------------------------------------------------------------------------

        if "Ankle" in self.proc_filename:
            if self.proc_filename.split(".")[-1] == "csv":
                df_ankle = pd.read_csv(self.proc_filepath)

            if self.proc_filename.split(".")[-1] == "xlsx":
                df_ankle = pd.read_excel(self.proc_filepath)

            df_ankle["Timestamp"] = pd.to_datetime(df_ankle["Timestamp"])

            self.timestamps = [datetime.strftime(i, "%Y-%m-%d %H:%M:%S.%f") for i in df_ankle["Timestamp"]]
            self.timestamps = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f") for i in self.timestamps]

            """
            # Timestamp formatting: files have been generated using different packages over the project
            if type(df_wrist["Timestamp"].iloc[0]) is str:
                self.timestamps = pd.to_datetime(df_wrist["Timestamp"], format="%Y-%m-%d %H:%M:%S.%f")

            if type(df_wrist["Timestamp"].iloc[0]) is not str:
                self.timestamps = [datetime.strftime(i, "%Y-%m-%d %H:%M:%S.%f") for i in df_wrist["Timestamp"]]
                self.timestamps = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f") for i in self.timestamps]
            """

            self.epoch_len = (self.timestamps[1] - self.timestamps[0]).seconds
            self.svm = [float(i) for i in df_ankle["ActivityCount"]]

            self.pred_mets = [float(i) for i in df_ankle["PredictedMETs"]]
            self.pred_speed = [float(i) for i in df_ankle["PredictedSpeed"]]
            self.intensity_cat = [int(i) for i in df_ankle["IntensityCategory"]]

        print("Complete.")

    def epoch_from_processed_wrist(self):

        df = pd.read_csv(self.proc_filepath, usecols=["Timestamps", "Wrist_SVM"])
        self.timestamps = [i for i in pd.to_datetime(df["Timestamps"])]
        self.svm = [i for i in df["Wrist_SVM"]]

    def epoch_from_processed_ankle(self):

        df = pd.read_csv(self.proc_filepath, usecols=["Timestamps", "Ankle_SVM", "Ankle_Intensity",
                                                      "Ankle_Speed", "Ankle_METs"])
        # self.epoch_len = int((self.timestamps.iloc[1] - self.timestamps.iloc[0]).seconds())
        self.timestamps = [i for i in pd.to_datetime(df["Timestamps"])]
        self.svm = [i for i in df["Ankle_SVM"]]
        self.pred_mets = [i for i in df["Ankle_METs"]]
        self.pred_speed = [i for i in df["Ankle_Speed"]]
        self.intensity_cat = [i for i in df["Ankle_Intensity"]]