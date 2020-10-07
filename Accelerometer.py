import ImportEDF
import EpochData

import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
from datetime import datetime
import statistics as stats
import pandas as pd


# ====================================================================================================================
# ================================================ WRIST ACCELEROMETER ===============================================
# ====================================================================================================================


class Wrist:

    def __init__(self, subject_id=None, raw_filepath=None, proc_filepath=None, filename=None,
                 temperature_filepath=None,
                 output_dir=None, load_raw=False, accel_only=False,
                 epoch_len=15, start_offset=0, end_offset=0, ecg_object=None,
                 from_processed=True, processed_folder=None):

        print()
        print("======================================== WRIST ACCELEROMETER ========================================")

        self.subject_id = subject_id
        self.filepath = raw_filepath
        self.proc_filepath = proc_filepath
        self.temperature_filepath = temperature_filepath
        self.filename = filename
        self.output_dir = output_dir

        self.load_raw = load_raw
        self.accel_only = accel_only

        self.epoch_len = epoch_len
        self.start_offset = start_offset
        self.end_offset = end_offset

        self.ecg_object = ecg_object

        self.from_processed = from_processed
        self.processed_folder = processed_folder

        # Loads raw accelerometer data and generates timestamps
        self.raw = ImportEDF.GENEActiv(filepath=self.filepath,
                                       start_offset=self.start_offset, end_offset=self.end_offset,
                                       load_raw=self.load_raw)

        self.epoch = EpochData.EpochAccel(raw_data=self.raw,
                                          accel_type="wrist",
                                          raw_filename=self.filename,
                                          proc_filepath=self.proc_filepath,
                                          accel_only=self.accel_only, epoch_len=self.epoch_len,
                                          from_processed=self.from_processed, processed_folder=self.processed_folder)

        # Model
        self.model = WristModel(accel_object=self, ecg_object=self.ecg_object)

        # Temperature data
        self.temperature = ImportEDF.GENEActivTemperature(filepath=self.temperature_filepath)

        if self.load_raw:
            self.temperature.sample_rate = 1 / (300 / self.raw.sample_rate)
        if not self.load_raw:
            self.temperature.sample_rate = 0.25


class WristModel:

    def __init__(self, accel_object, ecg_object=None):
        """Contains activity data for the Wrist accelerometer. Calculates total amount of time spent in each intensity
           category. Reported as % of valid epochs (awake + device worn). These values are contained in the
           intensity_totals object.

           If an ECG object is given, epochs with invalid ECG are accounted for and reflected in the
           intensity_totals_valid object.

           Report that gets printed in console does not reflect ECG data.
        """

        self.accel_object = accel_object

        if ecg_object is not None:
            self.valid_ecg = ecg_object.epoch_validity
        if ecg_object is None:
            self.valid_ecg = None

        self.epoch_intensity = []
        self.epoch_intensity_valid = None
        self.intensity_totals = None
        self.intensity_totals_valid = None

        # Calculates intensity based on Powell et al. (2017) cut-points
        self.powell_cutpoints()

    def powell_cutpoints(self):
        """Function that applies Powell et al. (2017) cut-points to epoched accelerometer data. Also calculates
           total minutes spent at each of the 4 intensities in minutes and as a percent of collection.

           :param
           -accel_object: Data class object that contains accelerometer data (epoch)
           """

        scaling_factor = self.accel_object.epoch_len / 15

        print("\nPowell et al. cut-points are being scaled by a factor of "
              "{} to match epoch lengths.".format(round(scaling_factor, 2)))

        print("\n" + "Applying Powell et al. (2017) cut-points to the data...")

        # Conversion factor: epochs to minutes
        epoch_to_minutes = 60 / self.accel_object.epoch_len

        # Sample rate
        sample_rate = self.accel_object.raw.sample_rate

        # Epoch-by-epoch intensity
        for epoch in self.accel_object.epoch.svm:
            if epoch < 47 * scaling_factor * sample_rate / 30:
                self.epoch_intensity.append(0)
            if 47 * scaling_factor * sample_rate / 30 <= epoch < 64 * scaling_factor * sample_rate / 30:
                self.epoch_intensity.append(1)
            if 64 * scaling_factor * sample_rate / 30 <= epoch < 157 * scaling_factor * sample_rate / 30:
                self.epoch_intensity.append(2)
            if epoch >= 157 * scaling_factor * sample_rate / 30:
                self.epoch_intensity.append(3)

        if self.valid_ecg is not None:
            index_list = min([len(self.epoch_intensity), len(self.valid_ecg)])

            self.epoch_intensity_valid = [self.epoch_intensity[i] if self.valid_ecg[i] == 0
                                          else None for i in range(0, index_list)]

        # MODEL TOTALS IF NOT CORRECTED USING VALID ECG EPOCHS -------------------------------------------------------
        # Intensity data: totals
        # In minutes and %
        self.intensity_totals = {"Sedentary": self.epoch_intensity.count(0) / epoch_to_minutes,
                                 "Sedentary%": round(self.epoch_intensity.count(0) /
                                                     len(self.accel_object.epoch.svm), 3),
                                 "Light": (self.epoch_intensity.count(1)) / epoch_to_minutes,
                                 "Light%": round(self.epoch_intensity.count(1) /
                                                 len(self.accel_object.epoch.svm), 3),
                                 "Moderate": self.epoch_intensity.count(2) / epoch_to_minutes,
                                 "Moderate%": round(self.epoch_intensity.count(2) /
                                                    len(self.accel_object.epoch.svm), 3),
                                 "Vigorous": self.epoch_intensity.count(3) / epoch_to_minutes,
                                 "Vigorous%": round(self.epoch_intensity.count(3) /
                                                    len(self.accel_object.epoch.svm), 3)}

        # MODEL TOTALS IF CORRECTED USING VALID ECG EPOCHS -----------------------------------------------------------
        # Intensity data: totals
        # In minutes and %
        if self.valid_ecg is not None:
            n_valid_epochs = len(self.epoch_intensity_valid) - self.epoch_intensity_valid.count(None)

            if n_valid_epochs == 0:
                n_valid_epochs = len(self.epoch_intensity_valid)

            self.intensity_totals_valid = {"Sedentary": self.epoch_intensity_valid.count(0) / epoch_to_minutes,
                                           "Sedentary%": round(self.epoch_intensity_valid.count(0) / n_valid_epochs, 3),
                                           "Light": (self.epoch_intensity.count(1)) / epoch_to_minutes,
                                           "Light%": round(self.epoch_intensity_valid.count(1) / n_valid_epochs, 3),
                                           "Moderate": self.epoch_intensity.count(2) / epoch_to_minutes,
                                           "Moderate%": round(self.epoch_intensity_valid.count(2) / n_valid_epochs, 3),
                                           "Vigorous": self.epoch_intensity.count(3) / epoch_to_minutes,
                                           "Vigorous%": round(self.epoch_intensity_valid.count(3) / n_valid_epochs, 3)}

        print("Complete.")

        print("\n" + "WRIST MODEL SUMMARY")
        print("-Sedentary: {} minutes ({}%)".format(self.intensity_totals["Sedentary"],
                                                    round(self.intensity_totals["Sedentary%"]*100, 3)))

        print("-Light: {} minutes ({}%)".format(self.intensity_totals["Light"],
                                                round(self.intensity_totals["Light%"]*100, 3)))

        print("-Moderate: {} minutes ({}%)".format(self.intensity_totals["Moderate"],
                                                   round(self.intensity_totals["Moderate%"]*100, 3)))

        print("-Vigorous: {} minutes ({}%)".format(self.intensity_totals["Vigorous"],
                                                   round(self.intensity_totals["Vigorous%"]*100, 3)))

# ====================================================================================================================
# ================================================ ANKLE ACCELEROMETER ===============================================
# ====================================================================================================================


class Ankle:

    def __init__(self, subject_id=None, raw_filepath=None, proc_filepath=None,
                 filename=None, load_raw=False, accel_only=False,
                 output_dir=None, rvo2=None, age=None, bmi=1, epoch_len=15,
                 start_offset=0, end_offset=0,
                 remove_baseline=False, ecg_object=None,
                 from_processed=True, treadmill_log_file=None, treadmill_regression_file=None,
                 processed_folder=None, write_results=False):

        print()
        print("======================================== ANKLE ACCELEROMETER ========================================")

        self.subject_id = subject_id
        self.filepath = raw_filepath
        self.filename = filename
        self.proc_filepath = proc_filepath
        self.load_raw = load_raw
        self.accel_only = accel_only
        self.output_dir = output_dir

        self.rvo2 = rvo2
        self.age = age
        self.bmi = bmi

        self.epoch_len = epoch_len
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.remove_baseline = remove_baseline

        self.ecg_object = ecg_object

        self.from_processed = from_processed
        self.processed_folder = processed_folder
        self.treadmill_log_file = treadmill_log_file
        self.treadmill_regression_file = treadmill_regression_file
        self.write_results = write_results

        # Loads raw accelerometer data and generates timestamps
        self.raw = ImportEDF.GENEActiv(filepath=self.filepath,
                                       start_offset=self.start_offset, end_offset=self.end_offset,
                                       load_raw=self.load_raw)

        self.epoch = EpochData.EpochAccel(raw_data=self.raw,
                                          accel_type="ankle",
                                          raw_filename=self.filename,
                                          proc_filepath=self.proc_filepath,
                                          epoch_len=self.epoch_len,
                                          remove_baseline=self.remove_baseline, accel_only=self.accel_only,
                                          from_processed=self.from_processed, processed_folder=self.processed_folder)

        # Create Treadmill object
        self.treadmill = Treadmill(ankle_object=self)

        # Create AnkleModel object
        self.model = AnkleModel(ankle_object=self, bmi=self.bmi, write_results=self.write_results,
                                ecg_object=self.ecg_object)

    def write_model(self):

        if not self.accel_only:
            out_filename = self.output_dir + self.filename.split(".")[0].split("/")[-1] + "_IntensityData.csv"
        if self.accel_only:
            out_filename = self.output_dir + self.filename.split(".")[0].split("/")[-1] + \
                           "_IntensityData_AccelOnly.csv"

        # Writes epoch-by-epoch data to .csv
        with open(out_filename, "w") as output:
            writer = csv.writer(output, delimiter=",", lineterminator="\n")

            writer.writerow(["Timestamp", "ActivityCount", "PredictedSpeed", "PredictedMETs", "IntensityCategory"])
            writer.writerows(zip(self.model.epoch_timestamps, self.model.epoch_data,
                                 self.model.linear_speed, self.model.predicted_mets, self.model.epoch_intensity))

        print("\n" + "Complete. File {}".format(out_filename))


class Treadmill:

    def __init__(self, ankle_object):
        """Class that stores treadmill protocol information from tracking spreadsheet.
           Imports protocol start time and each walks' speed. Stores this information in a dictionary.
           Calculates the index from the raw data which corresponds to the protocol start time.

        :returns
        -treadmill_dict: information imported from treadmill protocol spreadsheet
        -walk_speeds: list of walk speeds (easier to use than values in treadmill_dict)
        """

        self.subject_id = ankle_object.subject_id
        self.log_file = ankle_object.treadmill_log_file
        self.epoch_data = ankle_object.epoch.svm
        self.epoch_timestamps = ankle_object.epoch.timestamps
        self.epoch_len = ankle_object.epoch_len
        self.filename = ankle_object.filename
        self.walk_indexes = []
        self.valid_data = False
        self.equation_found = False

        # Creates treadmill dictionary and walk speed data from spreadsheet data
        self.treadmill_dict, self.walk_speeds, self.walk_indexes, self.df = self.import_log()

        self.avg_walk_counts = self.calculate_average_tm_counts()

    def import_log(self):
        """Retrieves treadmill protocol information from spreadsheet for correct subject:
           -Protocol start time, walking speeds in m/s, data index that corresponds to start of protocol"""

        if self.log_file is not None:

            if ".csv" in self.log_file:
                log = pd.read_csv(self.log_file, usecols=['SUBJECT', 'DATE', 'START_TIME', '60%_SPEED',
                                                          '80%_SPEED', 'PREF_SPEED', '120%_SPEED', '140%_SPEED',
                                                          'Walk1Start', 'Walk1End', 'Walk2Start', 'Walk2End',
                                                          'Walk3Start', 'Walk3End', 'Walk4Start', 'Walk4End',
                                                          'Walk5Start', 'Walk5End',
                                                          'Walk1Counts', 'Walk2Counts', 'Walk3Counts',
                                                          'Walk4Counts', 'Walk5Counts',
                                                          'Slope', 'Y_int', 'r2'])
            if ".xlsx" in self.log_file:
                log = pd.read_excel(self.log_file, usecols=['SUBJECT', 'DATE', 'START_TIME', '60%_SPEED',
                                                            '80%_SPEED', 'PREF_SPEED', '120%_SPEED', '140%_SPEED',
                                                            'Walk1Start', 'Walk1End', 'Walk2Start', 'Walk2End',
                                                            'Walk3Start', 'Walk3End', 'Walk4Start', 'Walk4End',
                                                            'Walk5Start', 'Walk5End',
                                                            'Walk1Counts', 'Walk2Counts', 'Walk3Counts',
                                                            'Walk4Counts', 'Walk5Counts', 'Slope', 'Y_int', 'r2'])

            # df = log.loc[log["SUBJECT"] == self.ankle_object.subject_id]
            df = log.loc[log["SUBJECT"].str.contains(str(self.subject_id))]

            if df.shape[0] == 1:
                self.valid_data = True

                # Formats treadmill protocol start time
                date = df['DATE'].iloc[0][0:4] + "/" + str(df['DATE'].iloc[0][4:7]).title() + "/" + \
                       df["DATE"].iloc[0][7:] + " " + df["START_TIME"].iloc[0]
                date_formatted = (datetime.strptime(date, "%Y/%b/%d %H:%M"))

                # Stores data and treadmill speeds (m/s) as dictionary
                treadmill_dict = {"File": df['SUBJECT'].iloc[0], "ProtocolTime": date_formatted,
                                  "60%": float(df['60%_SPEED']), "80%": float(df["80%_SPEED"]),
                                  "100%": float(df["PREF_SPEED"]), "120%": float(df["120%_SPEED"]),
                                  "140%": float(df["140%_SPEED"]), "Slope": None, "Y_int": None, "r2": None}

                # Same information as above; easier to access
                walk_speeds = [treadmill_dict["60%"], treadmill_dict["80%"],
                               treadmill_dict["100%"], treadmill_dict["120%"], treadmill_dict["140%"]]

                try:
                    colnames = [i for i in df.columns]
                    walk_indexes = [df[col_name].iloc[0] for col_name in colnames[8:18]]
                    print("\n" + "Previous processed treadmill data found. Skipping processing.")
                except ValueError:
                    walk_indexes = []
                    print("\n" + "No previous treadmill processing found. ")
                    pass

                # Retrieves regression equation data if available
                values = df[["Walk1Counts", "Walk2Counts", "Walk3Counts", "Walk4Counts", "Walk5Counts",
                             "Slope", "Y_int", "r2"]].values

                if True not in np.isnan(values):
                    self.equation_found = True

                    treadmill_dict["Slope"] = df["Slope"].iloc[0]
                    treadmill_dict["Y_int"] = df["Y_int"].iloc[0]
                    treadmill_dict["r2"] = df["r2"].iloc[0]

        # Sets treadmill_dict, walk_indexes and walk_speeds to empty objects if no treadmill data found in log
        if self.log_file is None or df.shape[0] == 0:
            df = None

            treadmill_dict = {"File": "N/A", "ProtocolTime": "N/A",
                              "StartIndex": "N/A",
                              "60%": "N/A", "80%": "N/A", "100%": "N/A", "120%": "N/A", "140%": "N/A",
                              "Slope": None, "Y_int": None, "r2": None}
            walk_indexes = []
            walk_speeds = []

            print("\nParticipant did not perform individual treadmill protocol. Using group-level regression.")

        return treadmill_dict, walk_speeds, walk_indexes, df

    def plot_treadmill_protocol(self, ankle_object, show_highlights=True):
        """Plots raw and epoched data during treadmill protocol on subplots or
           just epoched data if raw not available."""

        # If raw data available ---------------------------------------------------------------------------------------
        if ankle_object.raw.timestamps is not None:

            print("\n" + "Plotting raw and epoched treadmill protocol data.")

            raw_start = ankle_object.treadmill.treadmill_dict["StartIndex"] * \
                        ankle_object.raw.sample_rate * ankle_object.epoch_len

            if type(raw_start) != int:
                # Sets raw start index to 10 minutes prior to start of protocol
                raw_start = (ankle_object.treadmill.walk_indexes[0] - 5 * ankle_object.epoch_len) * \
                            ankle_object.raw.sample_rate * ankle_object.epoch_len
                if raw_start < 0:
                    raw_start = 0

            # If StartIndex is N/A...
            try:
                epoch_start = int(raw_start / (ankle_object.raw.sample_rate * ankle_object.epoch_len))
            except TypeError:
                epoch_start = ankle_object.raw.sample_rate * ankle_object.epoch_len

            # X-axis coordinates that correspond to epoch number
            # One hour = 3600 seconds
            index_list = np.arange(raw_start, raw_start + 4800 * ankle_object.raw.sample_rate) / \
                         ankle_object.raw.sample_rate / ankle_object.epoch_len

            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex="col", figsize=(10, 7))

            ax1.set_title("{}: Treadmill Protocol".format(ankle_object.filename))

            # Plots one hour of raw data (3600 seconds)
            ax1.plot(index_list, ankle_object.raw.x[raw_start:raw_start + ankle_object.raw.sample_rate * 4800],
                     color="black", label="X")
            ax1.plot(index_list, ankle_object.raw.y[raw_start:raw_start + ankle_object.raw.sample_rate * 4800],
                     color="green", label="Y")
            ax1.plot(index_list, ankle_object.raw.z[raw_start:raw_start + ankle_object.raw.sample_rate * 4800],
                     color="red", label="Z")
            ax1.legend()
            ax1.set_ylabel("G's")

            ax2.plot(index_list, ankle_object.raw.vm[raw_start:raw_start + ankle_object.raw.sample_rate * 4800],
                     color="blue", label="Vector Mag.")
            ax2.legend()
            ax2.set_ylabel("G's")

            # Epoched data
            ax3.bar(index_list[::ankle_object.epoch_len * ankle_object.raw.sample_rate],
                    ankle_object.epoch.svm[epoch_start:epoch_start + int(4800 / ankle_object.epoch_len)],
                    width=1.0, edgecolor='black', color='grey', alpha=0.75, align="edge")
            ax3.set_ylabel("Counts")

            # Highlights treadmill walks
            if show_highlights:
                for start, stop in zip(self.walk_indexes[::2], self.walk_indexes[1::2]):
                    ax1.fill_betweenx(y=(0, max(ankle_object.epoch.svm[self.walk_indexes[0]-10:
                                                                       self.walk_indexes[0] +
                                                                       int(4800 / ankle_object.epoch_len)])),
                                      x1=start, x2=stop, color='green', alpha=0.35)

            plt.show()

        # If raw data not available -----------------------------------------------------------------------------------
        if ankle_object.raw.timestamps is None:

            print("\n" + "Plotting epoched treadmill protocol data. Raw data not available.")

            fig, ax1 = plt.subplots(1, figsize=(10, 7))

            ax1.bar(np.arange(self.walk_indexes[0]-10, self.walk_indexes[0] + int(4800 / ankle_object.epoch_len), 1),
                    ankle_object.epoch.svm[self.walk_indexes[0]-10:
                                           self.walk_indexes[0] + int(4800 / ankle_object.epoch_len)],
                    width=1.0, edgecolor='black', color='grey', alpha=0.75, align="edge")
            ax1.set_ylabel("Counts")
            ax1.set_xlabel("Epoch Number")
            ax1.set_title("Participant {}: Treadmill Protocol - Epoched Data".format(ankle_object.subject_id))

            # Highlights treadmill walks
            if show_highlights:
                for start, stop in zip(self.walk_indexes[::2], self.walk_indexes[1::2]):
                    ax1.fill_betweenx(y=(0, max(ankle_object.epoch.svm[self.walk_indexes[0]-10:
                                                                       self.walk_indexes[0] +
                                                                       int(4800 / ankle_object.epoch_len)])),
                                      x1=start, x2=stop, color='green', alpha=0.35)

            plt.show()

    def calculate_average_tm_counts(self):
        """Calculates average counts per epoch from the ankle accelerometer.

        :returns
        -avg_walk_count: I bet you can figure this one out on your own
        """

        if not self.equation_found:
            try:
                avg_walk_count = [round(stats.mean(self.epoch_data[self.walk_indexes[index]:
                                                                   self.walk_indexes[index+1]]), 2)
                                  for index in np.arange(0, len(self.walk_indexes), 2)]

            except IndexError:
                avg_walk_count = [0, 0, 0, 0, 0]

        if self.equation_found:
            avg_walk_count = [self.df["Walk1Counts"], self.df["Walk2Counts"], self.df["Walk3Counts"],
                              self.df["Walk4Counts"], self.df["Walk5Counts"]]

        return avg_walk_count


class AnkleModel:

    def __init__(self, ankle_object, bmi=1, write_results=False, ecg_object=None):
        """Class that stores ankle model data. Performs regression analysis on activity counts vs. gait speed if
           participant performed the treadmill protocol. Uses group-level regression otherwise.

        Predicts gait speed and METs from activity counts using ACSM equation that predicts VO2 from gait speed.

        :arguments
        -ankle_object: AnkleAccel class instance
        -treadmill_object: Treadmill class instance
        -output_dir: pathway to folder where data is to be saved
        """

        self.epoch_data = ankle_object.epoch.svm
        self.epoch_len = ankle_object.epoch_len
        self.accel_only = ankle_object.accel_only
        self.epoch_scale = 1
        self.epoch_timestamps = ankle_object.epoch.timestamps
        self.subject_id = ankle_object.subject_id
        self.filepath = ankle_object.filepath
        self.filename = ankle_object.filename
        self.bmi = bmi
        self.rvo2 = ankle_object.rvo2
        self.tm_object = ankle_object.treadmill
        self.walk_indexes = None
        self.write_results = write_results

        self.regression_df = None
        self.standard_error = None
        self.r2 = None

        if ecg_object is not None:
            self.valid_ecg = ecg_object.epoch_validity
        if ecg_object is None:
            self.valid_ecg = None

        self.output_dir = ankle_object.output_dir

        if self.tm_object.valid_data:

            # Index multiplier for different epoch lengths since treadmill data processed with 15-second epochs
            self.walk_indexes = self.scale_epoch_indexes()

            # Adds average count data to self.tm_object since it has to be run in a weird order
            self.calculate_average_tm_counts()

        self.regression_dict, self.linear_speed = self.calculate_regression()

        self.predicted_mets, self.epoch_intensity, \
            self.intensity_totals = self.calculate_intensity(self.linear_speed)

    def scale_epoch_indexes(self):
        """Scales treadmill walk indexes if epoch length is not 15 seconds. Returns new list."""

        if self.epoch_len != 15:
            self.epoch_scale = int(np.floor(15 / self.epoch_len))

            walk_indexes = [i * self.epoch_scale for i in self.tm_object.walk_indexes]

        if self.epoch_len == 15:
            walk_indexes = self.tm_object.walk_indexes

        return walk_indexes

    def calculate_average_tm_counts(self):
        """Calculates average activity count total for each treadmill walk."""

        if self.walk_indexes[-1] <= len(self.epoch_data):
            self.tm_object.avg_walk_counts = [round(stats.mean(self.epoch_data[self.walk_indexes[index]:
                                                                               self.walk_indexes[index + 1]]), 2)
                                              for index in np.arange(0, 10, 2)]
        if self.walk_indexes[-1] > len(self.epoch_data):
            self.tm_object.avg_walk_counts = [None for i in range(5)]
            self.tm_object.valid_data = False

    def calculate_regression(self):

        # INDIVIDUAL REGRESSION ---------------------------------------------------------------------------------------
        if self.tm_object.valid_data:
            regression_type = "Individual"

            if not self.tm_object.equation_found:

                # Reshapes data to work with
                counts = np.array(self.tm_object.avg_walk_counts).reshape(-1, 1)
                speed = np.array(self.tm_object.walk_speeds).reshape(-1, 1)  # m/s

                # Linear regression using sklearn
                lm = linear_model.LinearRegression()
                model = lm.fit(counts, speed)
                y_intercept = lm.intercept_[0]
                counts_coef = lm.coef_[0][0]
                bmi_coef = 0
                self.r2 = round(lm.score(counts, speed), 5)

            if self.tm_object.equation_found:
                y_intercept = self.tm_object.treadmill_dict["Y_int"]
                counts_coef = self.tm_object.treadmill_dict["Slope"]
                self.r2 = self.tm_object.treadmill_dict["r2"]

            # Threshold corresponding to a 5-second walk at preferred speed
            meaningful_threshold = round(self.tm_object.avg_walk_counts[2] / (self.epoch_len / 5), 2)

        # GROUP-LEVEL REGRESSION --------------------------------------------------------------------------------------
        if not self.tm_object.valid_data:
            y_intercept = 0.37979
            counts_coef = 0.00132
            bmi_coef = 0
            self.r2 = None

            # Threshold corresponding to a 5-second walk at average walking pace (assume 1.4 m/s)
            meaningful_threshold = round((1.4 - y_intercept) / counts_coef / (self.epoch_len / 5), 2)

            regression_type = "Group"

        # Calculates count and speed limits for different intensity levels
        light_speed = ((1.5 * self.rvo2 - self.rvo2) / 0.1) / 60  # m/s

        # TEMP VALUE
        light_counts = round((light_speed - y_intercept) / counts_coef, 1)
        mod_speed = ((3 * self.rvo2 - self.rvo2) / 0.1) / 60
        mod_counts = round((mod_speed - y_intercept) / counts_coef, 1)
        vig_speed = ((6 * self.rvo2 - self.rvo2) / 0.1) / 60
        vig_counts = round((vig_speed - y_intercept) / counts_coef, 1)

        # ESTIMATING SPEED --------------------------------------------------------------------------------------------

        # Predicts speed using linear regression
        linear_speed = [svm * counts_coef + y_intercept for svm in self.epoch_data]

        # Creates a list of predicted speeds where any speed below the sedentary threshold is set to 0 m/s
        above_sed_thresh = []

        # Sets threshold to either meaningful_threshold OR light_counts based on which is greater
        if meaningful_threshold >= light_counts:
            meaningful_threshold = meaningful_threshold
        if meaningful_threshold < light_counts:
            meaningful_threshold = light_counts

        for speed, counts in zip(linear_speed, self.epoch_data):
            if counts >= meaningful_threshold:
                above_sed_thresh.append(speed)
            if counts < meaningful_threshold:
                above_sed_thresh.append(0)

        linear_reg_dict = {"Regression Type": regression_type,
                           "a": counts_coef, "b": y_intercept, "r2": self.r2,
                           "Light speed": round(light_speed, 3), "Light counts": light_counts,
                           "Moderate speed": round(mod_speed, 3), "Moderate counts": mod_counts,
                           "Vigorous speed": round(vig_speed, 3), "Vigorous counts": vig_counts,
                           "Meaningful threshold": meaningful_threshold}

        return linear_reg_dict, above_sed_thresh

    def counts_to_speed(self, count, print_output=True):

        speed = self.regression_dict["a"] * count + self.regression_dict["b"]

        if print_output:
            print("-Predicted speed for {} counts is {} m/s.".format(count, round(speed, 3)))

        return speed

    def plot_regression(self):
        """Plots measured results and results predicted from regression."""

        if not self.tm_object.valid_data:
            print("\nNo valid treadmill data; cannot generate plot.")
            return None

        # Variables from each regression type ------------------------------------------------------------------------
        min_value = np.floor(min(self.epoch_data))
        max_value = np.ceil(max(self.epoch_data))

        dict = self.regression_dict
        curve_data = [round(i * self.regression_dict["a"] + self.regression_dict["b"], 3)
                      for i in np.arange(0, max_value)]
        predicted_max = max_value * self.regression_dict["a"] + self.regression_dict["b"]

        # Threshold below which counts are considered noise (100% preferred speed / 3)
        meaningful_thresh = self.regression_dict["Meaningful threshold"]

        # Uses regression to calculate speed equivalent at meaningful threshold
        # No physiological meaning if it is derived from meaningful threshold instead of light counts
        light_speed = self.regression_dict["Meaningful threshold"] * self.regression_dict["a"] + self.regression_dict["b"]

        min_value = 0

        # Plot --------------------------------------------------------------------------------------------------------

        plt.figure(figsize=(10, 7))

        # Measured (true) values
        plt.plot(self.tm_object.avg_walk_counts, self.tm_object.walk_speeds, label='Treadmill Protocol',
                 markerfacecolor='white', markeredgecolor='black', color='black', marker="o")

        # Predicted values: count range between min and max svm
        plt.plot(np.arange(min_value, max_value), curve_data,
                 label='Regression line (r^2 = {})'.format(dict["r2"]), color='#1993C5', linestyle='dashed')

        # Fills in regions for different intensities
        plt.fill_between(x=[0, meaningful_thresh], y1=0, y2=light_speed,
                         color='grey', alpha=0.5, label="Sedentary")

        plt.fill_between(x=[meaningful_thresh, dict["Moderate counts"]],
                         y1=light_speed, y2=dict["Moderate speed"],
                         color='green', alpha=0.5, label="Light")

        plt.fill_between(x=[dict["Moderate counts"], dict["Vigorous counts"]],
                         y1=dict["Moderate speed"], y2=dict["Vigorous speed"],
                         color='orange', alpha=0.5, label="Moderate")

        plt.fill_between(x=[dict["Vigorous counts"], max_value],
                         y1=dict["Vigorous speed"],
                         y2=predicted_max,
                         color='red', alpha=0.5, label="Vigorous")

        # Lines on axes
        plt.axhline(y=0, color='black')
        plt.axvline(x=0, color='black')

        plt.xlim(0, max(self.epoch_data))
        plt.ylim(0, max(self.linear_speed))

        plt.legend(loc='upper left')
        plt.ylabel("Gait speed (m/s)")
        plt.xlabel("Counts")
        plt.title("Participant #{}: Treadmill Protocol - " 
                  "Counts vs. Gait Speed ({} regression)".format(self.subject_id,
                                                                 self.regression_dict["Regression Type"]))
        plt.show()

    def calculate_intensity(self, predicted_speed):

        # Converts m/s to m/min
        m_min = [i * 60 for i in predicted_speed]

        # Uses ACSM equation to predict METs from predicted gait speed
        mets = []

        for epoch_speed in m_min:
            if epoch_speed <= 100:
                mets.append((self.rvo2 + .1 * epoch_speed) / self.rvo2)
            if epoch_speed > 100:
                mets.append((self.rvo2 + .2 * epoch_speed) / self.rvo2)

        # Calculates epoch-by-epoch intensity
        # <1.5 METs = sedentary, 1.5-2.99 METs = light, 3.00-5.99 METs = moderate, >= 6.0 METS = vigorous
        intensity = []

        for met in mets:
            if met < 1.5:
                intensity.append(0)
            if 1.5 <= met < 3.0:
                intensity.append(1)
            if 3.0 <= met < 6.0:
                intensity.append(2)
            if met >= 6.0:
                intensity.append(3)

        # Calculates time spent in each intensity category
        intensity_totals = {"Sedentary": intensity.count(0) / (60 / self.epoch_len),
                            "Sedentary%": round(intensity.count(0) / len(self.epoch_data), 3),
                            "Light": intensity.count(1) / (60 / self.epoch_len),
                            "Light%": round(intensity.count(1) / len(self.epoch_data), 3),
                            "Moderate": intensity.count(2) / (60 / self.epoch_len),
                            "Moderate%": round(intensity.count(2) / len(self.epoch_data), 3),
                            "Vigorous": intensity.count(3) / (60 / self.epoch_len),
                            "Vigorous%": round(intensity.count(3) / len(self.epoch_data), 3)}

        print("\n" + "ANKLE MODEL SUMMARY")
        print("Sedentary: {} minutes ({}%)".format(intensity_totals["Sedentary"],
                                                   round(intensity_totals["Sedentary%"] * 100, 3)))

        print("Light: {} minutes ({}%)".format(intensity_totals["Light"],
                                               round(intensity_totals["Light%"] * 100, 3)))

        print("Moderate: {} minutes ({}%)".format(intensity_totals["Moderate"],
                                                  round(intensity_totals["Moderate%"] * 100, 3)))

        print("Vigorous: {} minutes ({}%)".format(intensity_totals["Vigorous"],
                                                  round(intensity_totals["Vigorous%"] * 100, 3)))

        return mets, intensity, intensity_totals

    def plot_results(self):
        """Plots predicted speed, predicted METs, and predicted intensity categorization on 3 subplots"""

        print("\n" + "Plotting ankle model data...")

        # X-axis datestamp formating
        xfmt = mdates.DateFormatter("%a, %I:%M %p")
        locator = mdates.HourLocator(byhour=[0, 12], interval=1)

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(10, 6))
        ax1.set_title("Participant #{}: Ankle Model Data".format(self.subject_id))

        # Counts
        ax1.plot(self.epoch_timestamps[:len(self.linear_speed)], self.epoch_data[:len(self.linear_speed)],
                 color='black')
        ax1.set_ylabel("Counts")

        # Predicted speed (m/s)
        ax2.plot(self.epoch_timestamps[:len(self.linear_speed)], self.linear_speed, color='black')
        ax2.set_ylabel("Predicted Speed (m/s)")

        # Predicted METs
        ax3.plot(self.epoch_timestamps[:len(self.predicted_mets)], self.predicted_mets, color='black')
        ax3.axhline(y=1.5, linestyle='dashed', color='green')
        ax3.axhline(y=3.0, linestyle='dashed', color='orange')
        ax3.axhline(y=6.0, linestyle='dashed', color='red')
        ax3.set_ylabel("METs")

        # Intensity category
        ax4.plot(self.epoch_timestamps[:len(self.epoch_intensity)], self.epoch_intensity, color='black')
        ax4.set_ylabel("Intensity Category")

        ax4.xaxis.set_major_formatter(xfmt)
        ax4.xaxis.set_major_locator(locator)
        plt.xticks(rotation=45, fontsize=6)
