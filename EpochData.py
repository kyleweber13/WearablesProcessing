from datetime import datetime
import math
import pandas as pd


class EpochAccel:

    def __init__(self, raw_data=None, remove_baseline=False, from_processed=True,
                 processed_folder=None, accel_only=False, epoch_len=15):

        self.epoch_len = epoch_len
        self.remove_baseline = remove_baseline
        self.from_processed = from_processed
        self.accel_only = accel_only
        self.processed_folder = processed_folder
        self.raw_filename = raw_data.filepath.split("/")[-1].split(".")[0]

        if not self.accel_only:
            self.processed_file = self.processed_folder + self.raw_filename + "_IntensityData.csv"
        if self.accel_only:
            self.processed_file = self.processed_folder + self.raw_filename + "_IntensityData_AccelOnly.csv"

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
        if self.from_processed:
            self.epoch_from_processed()

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

        print("\n" + "Importing accelerometer-only data processed from {}.".format(self.processed_folder))

        # Wrist accelerometer ----------------------------------------------------------------------------------------
        if "Wrist" in self.processed_file:

            if self.processed_file.split(".")[-1] == "csv":
                df_wrist = pd.read_csv(self.processed_file)

            if self.processed_file.split(".")[-1] == "xlsx":
                df_wrist = pd.read_excel(self.processed_file)

            df_wrist["Timestamp"] = pd.to_datetime(df_wrist["Timestamp"])

            self.timestamps = [datetime.strftime(i, "%Y-%m-%d %H:%M:%S.%f") for i in df_wrist["Timestamp"]]
            self.timestamps = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f") for i in self.timestamps]

            self.epoch_len = (self.timestamps[1] - self.timestamps[0]).seconds
            self.svm = [round(float(i), 2) for i in df_wrist["ActivityCount"]]

        # Ankle accelerometer ----------------------------------------------------------------------------------------

        if "Ankle" in self.processed_file:
            if self.processed_file.split(".")[-1] == "csv":
                df_ankle = pd.read_csv(self.processed_file)

            if self.processed_file.split(".")[-1] == "xlsx":
                df_ankle = pd.read_excel(self.processed_file)

            df_ankle["Timestamp"] = pd.to_datetime(df_ankle["Timestamp"])

            self.timestamps = [datetime.strftime(i, "%Y-%m-%d %H:%M:%S.%f") for i in df_ankle["Timestamp"]]
            self.timestamps = [datetime.strptime(i, "%Y-%m-%d %H:%M:%S.%f") for i in self.timestamps]

            self.epoch_len = (self.timestamps[1] - self.timestamps[0]).seconds
            self.svm = [float(i) for i in df_ankle["ActivityCount"]]

            self.pred_mets = [float(i) for i in df_ankle["PredictedMETs"]]
            self.pred_speed = [float(i) for i in df_ankle["PredictedSpeed"]]
            self.intensity_cat = [int(i) for i in df_ankle["IntensityCategory"]]

        print("Complete.")
