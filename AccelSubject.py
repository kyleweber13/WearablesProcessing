import pyedflib
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from Accelerometer import AccelerometerCondensed
import pandas as pd
import os
import numpy as np
import Filtering

xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")


class Subject:

    def __init__(self, subj_id=None,
                 lw_filepath=None, lw_temp_filepath=None,
                 la_filepath=None, la_temp_filepath=None,
                 rw_filepath=None, rw_temp_filepath=None,
                 ra_filepath=None, ra_temp_filepath=None,
                 lw_start_offset=0, la_start_offset=0, rw_start_offset=0, ra_start_offset=0,
                 crop_starts=False,
                 processed_filepath=None, from_processed=False,
                 load_raw=True, output_dir=None, epoch_len=15,
                 write_epoched_data=False, write_intensity_data=False, overwrite_output=False):
        """Class to read in EDF-formatted wrist and ankle accelerometer files.

        :argument
        -subj_id: used for file naming, str
            -format example: "OND07_WTL_3013"
        -wrist_filepath: full pathway to wrist .edf accelerometer file
        -wrist_temp_filepath: full pathway to wrist .edf temperature file
        -ankle_filepath: full pathway to ankle .edf accelerometer file
        -ankle_temp_filepath: full pathway to wrist .edf temperature file

        -wrist_start_offset, ankle_start_offset: datapoints to skip at start of collection
        -crop_starts: boolean whether to crop files to start at the same time

        -processed_filepath: full pathway to .csv file created using Subject.create_epoch_df()
        -load_raw: whether to load raw data; boolean
        -from_processed: whether to load file specified using processed_filepath; boolean
        -output_dir: full pathway to where files get written
        -epoch_len: epoch length in seconds, int
        -write_epoched_data: whether to write df_epoch to .csv; boolean
        -write_intensity_data: whether to write df_daily and activity_totals to .csv's; boolean
        -overwrite_output: whether to automatically overwrite existing df_epoch file; boolean
            -If False, user will be prompted to manually overwrite existing file.
        """

        self.subj_id = subj_id

        """Left wrist filename(s)"""
        if lw_filepath is not None:
            self.lw_filepath = lw_filepath.format(subj_id.split("_")[-1])
            self.lw_exists = True if os.path.exists(lw_filepath.format(subj_id.split("_")[-1])) else False
        if lw_filepath is None:
            self.lw_filepath = None
            self.lw_exists = False

        if lw_temp_filepath is not None:
            self.lw_temp_filepath = lw_temp_filepath.format(subj_id.split("_")[-1])
            self.lw_temp_exists = True if os.path.exists(lw_temp_filepath.format(subj_id.split("_")[-1])) else False
        if lw_temp_filepath is None:
            self.lw_temp_filepath = None
            self.lw_temp_exists = False

        """Right wrist filename(s)"""
        if rw_filepath is not None:
            self.rw_filepath = rw_filepath.format(subj_id.split("_")[-1])
            self.rw_exists = True if os.path.exists(rw_filepath.format(subj_id.split("_")[-1])) else False
        if rw_filepath is None:
            self.rw_filepath = None
            self.rw_exists = False

        if rw_temp_filepath is not None:
            self.rw_temp_filepath = rw_temp_filepath.format(subj_id.split("_")[-1])
            self.rw_temp_exists = True if os.path.exists(rw_temp_filepath.format(subj_id.split("_")[-1])) else False
        if rw_temp_filepath is None:
            self.rw_temp_filepath = None
            self.rw_temp_exists = False

        """Left ankle filename(s)"""
        if la_filepath is not None:
            self.la_filepath = la_filepath.format(subj_id.split("_")[-1])
            self.la_exists = True if os.path.exists(la_filepath.format(subj_id.split("_")[-1])) else False
        if la_filepath is None:
            self.la_filepath = None
            self.la_exists = False

        if la_temp_filepath is not None:
            self.la_temp_filepath = la_temp_filepath.format(subj_id.split("_")[-1])
            self.la_temp_exists = True if os.path.exists(la_temp_filepath.format(subj_id.split("_")[-1])) else False
        if la_temp_filepath is None:
            self.la_temp_filepath = None

        """Right ankle filename(s)"""
        if ra_filepath is not None:
            self.ra_filepath = ra_filepath.format(subj_id.split("_")[-1])
            self.ra_exists = True if os.path.exists(ra_filepath.format(subj_id.split("_")[-1])) else False
        if ra_filepath is None:
            self.ra_filepath = None
            self.ra_exists = False
        if ra_temp_filepath is not None:
            self.ra_temp_filepath = ra_temp_filepath.format(subj_id.split("_")[-1])
            self.ra_temp_exists = True if os.path.exists(ra_temp_filepath.format(subj_id.split("_")[-1])) else False
        if ra_temp_filepath is None:
            self.ra_temp_filepath = None
            self.ra_temp_exists = False

        self.lw_offset = lw_start_offset
        self.rw_offset = rw_start_offset
        self.la_offset = la_start_offset
        self.ra_offset = ra_start_offset

        self.processed_filepath = processed_filepath
        self.output_dir = output_dir
        self.load_raw = load_raw
        self.from_processed = from_processed
        self.epoch_len = epoch_len
        self.write_epoched = write_epoched_data
        self.write_intensity_data = write_intensity_data
        self.overwrite_output = overwrite_output

        self.activity_totals = {"Sedentary": 0, "Light": 0, "Moderate": 0, "Vigorous": 0, "MVPA": 0}
        self.df_daily = pd.DataFrame(columns=["Date", "Sedentary", "Light", "Moderate", "Vigorous", "MVPA"])

        # ================================================== RUNS METHODS =============================================

        # Prints summary of what data will be imported
        self.print_summary()

        if self.load_raw:
            if crop_starts:
                self.sync_starts()

            if self.lw_filepath is not None:
                self.lw, self.lw_cutpoint_dict = self.create_wrist_obj(filepath=self.lw_filepath,
                                                                       temp_filepath=self.lw_temp_filepath,
                                                                       offset_index=self.lw_offset)

                self.lw_svm, self.lw_avm = self.epoch_accel(acc_type="wrist",
                                                            fs=self.lw.sample_rate,
                                                            vm_data=self.lw.accel_vm)

            """if self.wrist_filepath is None:
                self.wrist = None
                self.cutpoint_dict = None
                self.wrist_svm = None
                self.wrist_avm = None

            if self.ankle_filepath is not None:
                self.ankle = self.create_ankle_obj()
                self.ankle_svm, self.ankle_avm = self.epoch_accel(acc_type="ankle",
                                                                  fs=self.ankle.sample_rate,
                                                                  vm_data=self.ankle.accel_vm)

            self.df_epoch = self.create_epoch_df(write_df=self.write_epoched)

            if self.wrist_filepath is not None:
                self.calculate_wrist_intensity()

        if self.from_processed:
            self.df_epoch = self.import_processed_df()"""

        print("\n=====================================================================================================")
        print("Processing complete.")

    def print_summary(self):
        """Prints summary of what data will be read in."""

        print("======================================================================================================")
        print("\nData import summary:")

        if self.lw_filepath is not None:
            print("-Importing left wrist file: {}".format(self.lw_filepath))
        if self.lw_filepath is None:
            print("-No left wrist file will be imported.")

        if self.rw_filepath is not None:
            print("-Importing right wrist file: {}".format(self.rw_filepath))
        if self.rw_filepath is None:
            print("-No right wrist file will be imported.")

        if self.la_filepath is not None:
            print("-Importing left ankle file: {}".format(self.la_filepath))
        if self.la_filepath is None:
            print("-No left ankle file will be imported.")

        if self.ra_filepath is not None:
            print("-Importing right ankle file: {}".format(self.ra_filepath))
        if self.ra_filepath is None:
            print("-No right ankle file will be imported.")

        if self.load_raw:
            print("\n-Raw data will be imported.")
        if not self.load_raw:
            print("\n-Raw data will not be imported.")

        if not self.from_processed:
            print("-Data will not be read from processed.")
        if self.from_processed:
            print("-Data will be read from processed.")

        print()
        print("======================================================================================================")

    @staticmethod
    def check_file(filepath, print_summary=True):
        """Checks EDF header info to retrieve start time and sample rate.
           Used for cropping ankle and wrist accelerometers data.

        :returns
        -start time: timestamp
        -sampling rate: Hz (int)
        """

        if filepath is None or not os.path.exists(filepath):
            return None, None

        edf_file = pyedflib.EdfReader(filepath)

        duration = edf_file.getFileDuration()
        start_time = edf_file.getStartdatetime()
        end_time = start_time + timedelta(seconds=edf_file.getFileDuration())

        if print_summary:
            print("\n", filepath)
            print("-Sample rate: {}Hz".format(edf_file.getSampleFrequency(0)))
            print("-Start time: ", start_time)
            print("-End time:", end_time)
            print("-Duration: {} hours".format(round(duration / 3600, 2)))

        return start_time, edf_file.getSampleFrequency(0)
    
    def sync_starts(self):

        print("\nChecking file start times to sync devices...")

        if self.la_exists + self.ra_exists + self.lw_exists + self.rw_exists > 1:
            print("-Multiple files found. Cropping start times...")

            start_dict = {"LA_start": None, "LA_fs": 1, "RA_start": None, "RA_fs": 1,
                          "LW_start": None, "LW_fs": 1, "RW_start": None, "RW_fs": 1}

            start_dict["LA_start"], start_dict["LA_fs"] = self.check_file(filepath=self.la_filepath, print_summary=False)
            start_dict["RA_start"], start_dict["RA_fs"] = self.check_file(filepath=self.ra_filepath, print_summary=False)
            start_dict["LW_start"], start_dict["LW_fs"] = self.check_file(filepath=self.lw_filepath, print_summary=False)
            start_dict["RW_start"], start_dict["RW_fs"] = self.check_file(filepath=self.rw_filepath, print_summary=False)

            last_start = max([start_dict["LA_start"], start_dict["RA_start"],
                             start_dict["LW_start"], start_dict["RW_start"]])

            if self.la_exists and start_dict["LA_start"] > last_start:
                self.la_offset = int((last_start - start_dict["LA_start"]).total_seconds() * start_dict["LA_fs"])
                print("    -Left ankle offset = ", str(self.la_offset))

            if self.ra_exists and start_dict["RA_start"] > last_start:
                self.ra_offset = int((last_start - start_dict["RA_start"]).total_seconds() * start_dict["RA_fs"])
                print("    -Right ankle offset = ", str(self.ra_offset))

            if self.lw_exists and start_dict["LW_start"] > last_start:
                self.lw_offset = int((last_start - start_dict["LW_start"]).total_seconds() * start_dict["LW_fs"])
                print("    -Left wrist offset = ", str(self.lw_offset))

            if self.rw_exists and start_dict["RW_start"] > last_start:
                self.rw_offset = int((last_start - start_dict["RW_start"]).total_seconds() * start_dict["RW_fs"])
                print("    -Right wrist offset = ", str(self.rw_offset))

            if self.la_offset == 0 and self.ra_offset == 0 and self.lw_offset == 0 and self.rw_offset == 0:
                print("    -All files begin at same time. No cropping will be performed.")

        if self.la_exists + self.ra_exists + self.lw_exists + self.rw_exists <= 1:
            print("-Only one file input/found. No cropping will be performed.")

    def create_wrist_obj(self, filepath, temp_filepath, offset_index):
        """Creates wrist accelerometer data object.
           Scales accelerometer cutpoints from Powell et al. (2017) to selected epoch length.

        :returns
        -wrist object
        -cutpoints: dictionary
        """

        print("\n--------------------------------------------- Wrist file --------------------------------------------")
        if filepath is not None and os.path.exists(self.lw_filepath):
            wrist = AccelerometerCondensed(raw_filepath=filepath,
                                           temp_filepath=temp_filepath,
                                           load_raw=self.load_raw,
                                           start_offset=offset_index)
            fs = wrist.sample_rate

        if filepath is None or not os.path.exists(filepath):
            wrist = None
            fs = 1

        cutpoint_dict = {"Light": 47 * fs / 30 * self.epoch_len / 15,
                         "Moderate": 64 * fs / 30 * self.epoch_len / 15,
                         "Vigorous": 157 * fs / 30 * self.epoch_len / 15}

        return wrist, cutpoint_dict

    def create_ankle_obj(self):
        """Creates ankle accelerometer data object.

        :returns
        -ankle object
        """

        print("\n--------------------------------------------- Ankle file --------------------------------------------")

        ankle = AccelerometerCondensed(raw_filepath=self.ankle_filepath,
                                       temp_filepath=self.ankle_temp_filepath,
                                       load_raw=self.load_raw,
                                       start_offset=self.ankle_offset)

        return ankle

    def epoch_accel(self, acc_type, fs, vm_data):
        """Epochs accelerometer data. Calculates sum of vector magnitudes (SVM) and average vector magnitude (AVM)
           values for specified epoch length.

           :returns
           -svm: list
           -avm: list
        """

        # Epochs data if read in raw and didn't read in processed data -----------------------------------------------
        if not self.load_raw or self.from_processed or vm_data is None:
            return None, None

        if self.load_raw and not self.from_processed:

            print("\nEpoching {} data into {}-second epochs...".format(acc_type, self.epoch_len))
            t0 = datetime.datetime.now()

            vm = [i for i in vm_data]
            svm = []
            avm = []

            for i in range(0, len(vm), int(fs * self.epoch_len)):

                if i + self.epoch_len * fs > len(vm):
                    break

                vm_sum = sum(vm[i:i + self.epoch_len * fs])

                avg = vm_sum * 1000 / len(vm[i:i + self.epoch_len * fs])

                svm.append(round(vm_sum, 2))
                avm.append(round(avg, 2))

            t1 = datetime.datetime.now()
            print("Complete ({} seconds)".format(round((t1 - t0).total_seconds(), 1)))

            return svm, avm

    def fill_missing_data(self):

        if self.wrist_filepath is not None and self.ankle_filepath is None:
            data_len = len(self.wrist_svm)

            self.ankle_svm = [None for i in range(data_len)]
            self.ankle_avm = [None for i in range(data_len)]

        if self.wrist_filepath is None and self.ankle_filepath is not None:
            data_len = len(self.ankle_svm)

            self.wrist_svm = [None for i in range(data_len)]
            self.wrist_avm = [None for i in range(data_len)]

    def calculate_wrist_intensity(self):
        """Calculates activity intensity using wrist cutpoints from Powell et al. (2017). Requires 15-second epochs.
           Calculates total and daily activity volumes."""

        print("\nCalculating activity intensity data using wrist accelerometer...")

        if self.epoch_len != 15:
            print("-Requires 15-second epoch length. Reprocess data and try again.")
            return None

        data = self.df_epoch["WristSVM"]

        intensity = []
        for i in data:
            if i < self.cutpoint_dict["Light"]:
                intensity.append("Sedentary")
            if self.cutpoint_dict["Light"] <= i < self.cutpoint_dict["Moderate"]:
                intensity.append("Light")
            if self.cutpoint_dict["Moderate"] <= i < self.cutpoint_dict["Vigorous"]:
                intensity.append("Moderate")
            if self.cutpoint_dict["Vigorous"] <= i:
                intensity.append("Vigorous")

        self.df_epoch["WristIntensity"] = intensity

        epoch_to_mins = 60 / self.epoch_len
        values = self.df_epoch["WristIntensity"].value_counts()

        if "Light" not in values.keys():
            values["Light"] = 0
        if "Moderate" not in values.keys():
            values["Moderate"] = 0
        if "Vigorous" not in values.keys():
            values["Vigorous"] = 0

        # TOTAL ACTIVITY ---------------------------------------------------------------------------------------------
        self.activity_totals = {"Sedentary": values["Sedentary"] / epoch_to_mins,
                                "Light": values["Light"] / epoch_to_mins,
                                "Moderate": values["Moderate"] / epoch_to_mins,
                                "Vigorous": values["Vigorous"] / epoch_to_mins,
                                "MVPA": values["Moderate"] / epoch_to_mins + values["Vigorous"] / epoch_to_mins}

        # DAILY ACTIVITY ---------------------------------------------------------------------------------------------
        dates = set([i.date() for i in self.df_epoch["Timestamp"]])
        self.df_epoch["Date"] = [i.date() for i in self.df_epoch["Timestamp"]]

        daily_totals = []

        for date in sorted(dates):
            df = self.df_epoch.loc[self.df_epoch["Date"] == date]
            values = df["WristIntensity"].value_counts()

            if "Light" not in values.keys():
                values["Light"] = 0
            if "Moderate" not in values.keys():
                values["Moderate"] = 0
            if "Vigorous" not in values.keys():
                values["Vigorous"] = 0

            values = values/4
            daily_data = [date, values["Sedentary"], values["Light"], values["Moderate"],
                          values["Vigorous"], values["Moderate"] + values["Vigorous"]]
            daily_totals.append(daily_data)

        self.df_daily = pd.DataFrame(daily_totals,
                                     columns=["Date", "Sedentary", "Light", "Moderate", "Vigorous", "MVPA"])

        # Adds totals as final row
        final_row = pd.DataFrame(list(zip(["TOTAL", self.activity_totals["Sedentary"], self.activity_totals["Light"],
                                           self.activity_totals["Moderate"], self.activity_totals["Vigorous"],
                                           self.activity_totals["MVPA"]])),
                                 index=["Date", "Sedentary", "Light", "Moderate", "Vigorous", "MVPA"]).transpose()
        self.df_daily = self.df_daily.append(final_row)
        self.df_daily = self.df_daily.reset_index()
        self.df_daily = self.df_daily.drop("index", axis=1)

        # Removes date column
        self.df_epoch = self.df_epoch.drop("Date", axis=1)

        # Writing activity totals data --------------------------------------------------------------------------------
        if self.write_intensity_data:
            write_file = False

            file_list = os.listdir(self.output_dir)
            f_name = "{}_DailyActivityVolume.csv".format(self.subj_id)

            # What to do if file already exists
            if f_name in file_list:

                # If overwrite set to True
                if self.overwrite_output:
                    write_file = True
                    print("Automatically overwritting existing file.")

                # If overwrite set to False, prompts user
                if not self.overwrite_output:
                    user_input = input("Overwrite existing file? y/n: ")

                    if user_input.capitalize() == "Y" or user_input.capitalize() == "Yes":
                        write_file = True

                    if user_input.capitalize() == "N" or user_input.capitalize() == "No":
                        print("File will not be overwritten.")

            # What to do if file does not exist
            if f_name not in file_list:
                write_file = True

            # Writing file?
            if write_file:
                print("Writing total activity volume data to "
                      "{}{}_DailyActivityVolume.csv".format(self.output_dir, self.subj_id))

                df = self.df_daily.copy()

                df.insert(loc=0, column="ID", value=[self.subj_id for i in range(self.df_daily.shape[0])])

                df.to_csv("{}{}_DailyActivityVolume.csv".format(self.output_dir, self.subj_id),
                          index=False, float_format='%.2f')

    def create_epoch_df(self, write_df=False):
        """Creates dataframe for epoched wrist and ankle data.
           Deletes corresponding data objects for memory management.
           Option to write to .csv and to automatically overwrite existing file. If file is not to be overwritten,
           user is prompted to manually overwrite.

        :argument
        -write_df: boolean

        :returns
        -epoched dataframe: df
        """

        print("\nCombining data into single dataframe...")

        if self.wrist_filepath is not None:
            timestamps = self.wrist.timestamps[::self.epoch_len * self.wrist.sample_rate]

        if self.ankle_filepath is not None and self.wrist_filepath is None:
            timestamps = self.ankle.timestamps[::self.epoch_len * self.ankle.sample_rate]

        if self.ankle_filepath is None or self.wrist_filepath is None:
            self.fill_missing_data()

        df = pd.DataFrame(list(zip(timestamps, self.wrist_svm, self.wrist_avm, self.ankle_svm, self.ankle_avm)),
                          columns=["Timestamp", "WristSVM", "WristAVM", "AnkleSVM", "AnkleAVM"])

        del self.wrist_svm, self.wrist_avm, self.ankle_svm, self.ankle_avm

        if write_df:
            write_file = False

            file_list = os.listdir(self.output_dir)
            f_name = "{}_EpochedAccelerometer.csv".format(self.subj_id)

            # What to do if file already exists -----------------------------------------------------------------------
            if f_name in file_list:

                # If overwrite set to True
                if self.overwrite_output:
                    write_file = True
                    print("Automatically overwritting existing file.")

                # If overwrite set to False, prompts user
                if not self.overwrite_output:
                    user_input = input("Overwrite existing file? y/n: ")

                    if user_input.capitalize() == "Y" or user_input.capitalize() == "Yes":
                        write_file = True

                    if user_input.capitalize() == "N" or user_input.capitalize() == "No":
                        print("File will not be overwritten.")

            # What to do if file does not exist -----------------------------------------------------------------------
            if f_name not in file_list:
                write_file = True

            # Writing file? -------------------------------------------------------------------------------------------
            if write_file:
                print("Writing epoched data to {}{}_EpochedAccelerometer.csv".format(self.output_dir, self.subj_id))
                df.to_csv("{}{}_EpochedAccelerometer.csv".format(self.output_dir, self.subj_id),
                          index=False, float_format='%.2f')

        return df

    def import_processed_df(self):
        """Imports existing processed epoch data file (.csv).

        :returns
        -dataframe: df
        """

        print("\nImporting existing data ({})".format(self.processed_filepath.split("/")[-1]))

        if "csv" in self.processed_filepath:
            df = pd.read_csv(self.processed_filepath)
        if "xlsx" in self.processed_filepath:
            df = pd.read_excel(self.processed_filepath)

        return df

    def plot_epoched(self, outcome_measure="SVM", show_cutpoints=False):

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(12, 7))
        plt.suptitle("{}-second epoch data".format(self.epoch_len))

        ax1.plot(self.df_epoch["Timestamp"], self.df_epoch["Wrist{}".format(outcome_measure)], color='red')
        ax1.set_title("Wrist")

        if show_cutpoints:
            ax1.axhline(self.cutpoint_dict["Light"], color='green')
            ax1.axhline(self.cutpoint_dict["Moderate"], color='orange')
            ax1.axhline(self.cutpoint_dict["Vigorous"], color='red')

        ax2.plot(self.df_epoch["Timestamp"], self.df_epoch["Ankle{}".format(outcome_measure)], color='dodgerblue')
        ax2.set_title("Ankle")

        ax2.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)


subj = "9844"

data = Subject(subj_id="OND06_SHB_{}".format(str(subj)),
               la_filepath="F:/OND06_SBH_{}_GNAC_ACCELEROMETER_LAnkle.edf".format(subj),
               ra_filepath="C:/Users/ksweber/Desktop/OND06_LeftoverFiles/"
                           "OND06_SBH_{}_GNAC_ACCELEROMETER_RAnkle.edf".format(subj),
               lw_filepath="D:/Accelerometer/OND06_SBH_{}_GNAC_ACCELEROMETER_LWrist.edf".format(subj),
               rw_filepath="D:/Accelerometer/OND06_SBH_{}_GNAC_ACCELEROMETER_RWrist.edf".format(subj),

               la_temp_filepath="O:/Data/ReMiNDD/Processed Data/GENEActiv/OND06_ALL_01_SNSR_GNAC_2020MAY31_DATAPKG/"
                                "Temperature/DATAFILES/OND06_SBH_{}_GNAC_TEMPERATURE_RAnkle.edf".format(subj),
               rw_temp_filepath="O:/Data/ReMiNDD/Processed Data/GENEActiv/OND06_ALL_01_SNSR_GNAC_2020MAY31_DATAPKG/"
                                "Temperature/DATAFILES/OND06_SBH_{}_GNAC_TEMPERATURE_LWrist.edf".format(subj),

               crop_starts=True, load_raw=True, write_epoched_data=False,
               processed_filepath=None, from_processed=False,

               output_dir="C:/Users/ksweber/Desktop/OND06_PD_Wrists/")


# TODO
# Add RWrist/dominant cutpoints (create_wrist_obj)
# Add data to epoch df
