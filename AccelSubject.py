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
                 crop_starts=False, dom_wrist="Right",
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
        -dom_wrist: "Right" or "Left" for hand dominance

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
        self.dom_wrist = dom_wrist

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

        self.la_temperature, self.ra_temperature, self.lw_temperature, self.rw_temperature = None, None, None, None

        # ================================================== RUNS METHODS =============================================

        # Prints summary of what data will be imported
        self.print_summary()

        # if self.load_raw:
        if crop_starts:
            self.sync_starts()

        # Imports and epochs wrist data
        self.lw, self.rw, self.lw_fs, self.rw_fs, self.lw_cutpoints, self.rw_cutpoints = self.create_wrist_obj()

        self.lw_svm, self.lw_avm = self.epoch_accel(acc_type="left wrist",
                                                    fs=self.lw.sample_rate if self.lw is not None else 1,
                                                    vm_data=self.lw.accel_vm if self.lw is not None else [])

        self.rw_svm, self.rw_avm = self.epoch_accel(acc_type="right wrist",
                                                    fs=self.rw.sample_rate if self.rw is not None else 1,
                                                    vm_data=self.rw.accel_vm if self.rw is not None else [])

        # Imports and epochs ankle data
        self.la, self.ra, self.la_fs, self.ra_fs = self.create_ankle_obj()

        self.la_svm, self.la_avm = self.epoch_accel(acc_type="left ankle",
                                                    fs=self.la_fs if self.la is not None else 1,
                                                    vm_data=self.la.accel_vm if self.la is not None else [])

        self.ra_svm, self.ra_avm = self.epoch_accel(acc_type="right ankle",
                                                    fs=self.ra_fs if self.ra is not None else 1,
                                                    vm_data=self.ra.accel_vm if self.ra is not None else [])

        if self.load_raw:
            self.df_epoch = self.create_epoch_df(write_df=self.write_epoched)

        if self.from_processed:
            self.df_epoch = self.import_processed_df()

        self.calculate_wrist_intensity()

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

            start_dict["LA_start"], start_dict["LA_fs"] = self.check_file(filepath=self.la_filepath,
                                                                          print_summary=False)
            start_dict["RA_start"], start_dict["RA_fs"] = self.check_file(filepath=self.ra_filepath,
                                                                          print_summary=False)
            start_dict["LW_start"], start_dict["LW_fs"] = self.check_file(filepath=self.lw_filepath,
                                                                          print_summary=False)
            start_dict["RW_start"], start_dict["RW_fs"] = self.check_file(filepath=self.rw_filepath,
                                                                          print_summary=False)

            last_start = max([i for i in [start_dict["LA_start"], start_dict["RA_start"],
                             start_dict["LW_start"], start_dict["RW_start"]] if i is not None])

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

    def create_wrist_obj(self):
        """Creates wrist accelerometer data object.
           Scales accelerometer cutpoints from Powell et al. (2017) to selected epoch length.

        :returns
        -wrist object
        -Dictionaries of Powell et al. 2017 wrist cutpoints scaled to sampling rate and epoch length
            -nd_cutpoints, d_cutpoints
        """

        if self.lw_filepath is not None and os.path.exists(self.lw_filepath):
            print("\n--------------------------------------- Left wrist file ----------------------------------------")
            lw = AccelerometerCondensed(raw_filepath=self.lw_filepath,
                                        temp_filepath=self.lw_temp_filepath,
                                        load_raw=self.load_raw,
                                        start_offset=self.lw_offset)
            lw_fs = lw.sample_rate

            if self.dom_wrist == "Right":
                lw_cutpoints = {"Light": 47 * lw_fs / 30 * self.epoch_len / 15,
                                "Moderate": 64 * lw_fs / 30 * self.epoch_len / 15,
                                "Vigorous": 157 * lw_fs / 30 * self.epoch_len / 15}
            if self.dom_wrist == "Left":
                lw_cutpoints = {"Light": 51 * lw_fs / 30 * self.epoch_len / 15,
                                "Moderate": 68 * lw_fs / 30 * self.epoch_len / 15,
                                "Vigorous": 142 * lw_fs / 30 * self.epoch_len / 15}

        if self.lw_filepath is None or not os.path.exists(self.lw_filepath):
            lw = None
            lw_fs = 1
            lw_cutpoints = {"Light": 1, "Moderate": 1, "Vigorous": 0}

        if self.rw_filepath is not None and os.path.exists(self.rw_filepath):
            print("\n--------------------------------------- Right wrist file ----------------------------------------")
            rw = AccelerometerCondensed(raw_filepath=self.rw_filepath,
                                        temp_filepath=self.rw_temp_filepath,
                                        load_raw=self.load_raw,
                                        start_offset=self.rw_offset)
            rw_fs = rw.sample_rate

            if self.dom_wrist == "Right":
                rw_cutpoints = {"Light": 47 * rw_fs / 30 * self.epoch_len / 15,
                                "Moderate": 64 * rw_fs / 30 * self.epoch_len / 15,
                                "Vigorous": 157 * rw_fs / 30 * self.epoch_len / 15}
            if self.dom_wrist == "Left":
                rw_cutpoints = {"Light": 51 * rw_fs / 30 * self.epoch_len / 15,
                                "Moderate": 68 * rw_fs / 30 * self.epoch_len / 15,
                                "Vigorous": 142 * rw_fs / 30 * self.epoch_len / 15}

        if self.rw_filepath is None or not os.path.exists(self.rw_filepath):
            rw = None
            rw_fs = 1
            rw_cutpoints = {"Light": 1, "Moderate": 1, "Vigorous": 0}

        return lw, rw, lw_fs, rw_fs, lw_cutpoints, rw_cutpoints

    def create_ankle_obj(self):
        """Creates ankle accelerometer data object.

        :returns
        -ankle object
        """

        la, ra, la_fs, ra_fs = None, None, 1, 1

        if self.la_filepath is not None and os.path.exists(self.la_filepath):
            print("\n--------------------------------------- Left ankle file ----------------------------------------")
            la = AccelerometerCondensed(raw_filepath=self.la_filepath,
                                        temp_filepath=self.la_temp_filepath,
                                        load_raw=self.load_raw,
                                        start_offset=self.la_offset)

            la_fs = la.sample_rate

        if self.ra_filepath is not None and os.path.exists(self.ra_filepath):
            print("\n--------------------------------------- Right ankle file ----------------------------------------")
            ra = AccelerometerCondensed(raw_filepath=self.ra_filepath,
                                        temp_filepath=self.ra_temp_filepath,
                                        load_raw=self.load_raw,
                                        start_offset=self.ra_offset)
            ra_fs = ra.sample_rate

        return la, ra, la_fs, ra_fs

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

        # Creates list of epoched data lengths
        data_lens = []

        if self.lw_svm is not None:
            data_lens.append(len(self.lw_svm))

        if self.rw_svm is not None:
            data_lens.append(len(self.rw_svm))

        if self.la_svm is not None:
            data_lens.append(len(self.la_svm))

        if self.ra_svm is not None:
            data_lens.append(len(self.ra_svm))

        # Fills data with lists of Nones
        data_len = max(data_lens)

        if self.lw_svm is None:
            self.lw_svm = [None for i in range(data_len)]
            self.lw_avm = [None for i in range(data_len)]

        if self.lw.temperature is None:
            self.lw_temperature = [None for i in range(data_len)]

        if self.la_svm is None:
            self.la_svm = [None for i in range(data_len)]
            self.la_avm = [None for i in range(data_len)]

        if self.la.temperature is None:
            self.la_temperature = [None for i in range(data_len)]

        if self.rw_svm is None:
            self.rw_svm = [None for i in range(data_len)]
            self.rw_avm = [None for i in range(data_len)]

        if self.rw.temperature is None:
            self.rw_temperature = [None for i in range(data_len)]

        if self.ra_svm is None:
            self.ra_svm = [None for i in range(data_len)]
            self.ra_avm = [None for i in range(data_len)]

        if self.ra.temperature is None:
            self.ra_temperature = [None for i in range(data_len)]

    def calculate_wrist_intensity(self):
        """Calculates activity intensity using wrist cutpoints from Powell et al. (2017). Requires 15-second epochs.
           Calculates total and daily activity volumes."""

        print("\nCalculating activity intensity data using wrist accelerometer...")

        if self.epoch_len != 15:
            print("-Requires 15-second epoch length. Reprocess data and try again.")
            return None

        # LEFT WRIST -------------------------------------------------------------------------------------------------
        lw = self.df_epoch["LW_SVM"]

        lw_intensity = []
        for i in lw:
            if i < self.lw_cutpoints["Light"]:
                lw_intensity.append("Sedentary")
            if self.lw_cutpoints["Light"] <= i < self.lw_cutpoints["Moderate"]:
                lw_intensity.append("Light")
            if self.lw_cutpoints["Moderate"] <= i < self.lw_cutpoints["Vigorous"]:
                lw_intensity.append("Moderate")
            if self.lw_cutpoints["Vigorous"] <= i:
                lw_intensity.append("Vigorous")

        self.df_epoch["LW_Intensity"] = lw_intensity

        # RIGHT WRIST -------------------------------------------------------------------------------------------------
        rw = self.df_epoch["RW_SVM"]

        rw_intensity = []
        for i in rw:
            if i < self.rw_cutpoints["Light"]:
                rw_intensity.append("Sedentary")
            if self.rw_cutpoints["Light"] <= i < self.rw_cutpoints["Moderate"]:
                rw_intensity.append("Light")
            if self.rw_cutpoints["Moderate"] <= i < self.rw_cutpoints["Vigorous"]:
                rw_intensity.append("Moderate")
            if self.rw_cutpoints["Vigorous"] <= i:
                rw_intensity.append("Vigorous")

        self.df_epoch["RW_Intensity"] = rw_intensity

        # SUMMARY MEASURES --------------------------------------------------------------------------------------------
        epoch_to_mins = 60 / self.epoch_len

        # Left wrist
        lw_values = self.df_epoch["LW_Intensity"].value_counts()

        if "Light" not in lw_values.keys():
            lw_values["Light"] = 0
        if "Moderate" not in lw_values.keys():
            lw_values["Moderate"] = 0
        if "Vigorous" not in lw_values.keys():
            lw_values["Vigorous"] = 0

        # Right wrist
        rw_values = self.df_epoch["RW_Intensity"].value_counts()

        if "Light" not in rw_values.keys():
            rw_values["Light"] = 0
        if "Moderate" not in rw_values.keys():
            rw_values["Moderate"] = 0
        if "Vigorous" not in rw_values.keys():
            rw_values["Vigorous"] = 0

        # TOTAL ACTIVITY ---------------------------------------------------------------------------------------------
        self.activity_totals = {"LW_Sedentary": lw_values["Sedentary"] / epoch_to_mins,
                                "LW_Light": lw_values["Light"] / epoch_to_mins,
                                "LW_Moderate": lw_values["Moderate"] / epoch_to_mins,
                                "LW_Vigorous": lw_values["Vigorous"] / epoch_to_mins,
                                "LW_MVPA": lw_values["Moderate"] / epoch_to_mins +
                                           lw_values["Vigorous"] / epoch_to_mins,
                                "RW_Sedentary": rw_values["Sedentary"] / epoch_to_mins,
                                "RW_Light": rw_values["Light"] / epoch_to_mins,
                                "RW_Moderate": rw_values["Moderate"] / epoch_to_mins,
                                "RW_Vigorous": rw_values["Vigorous"] / epoch_to_mins,
                                "RW_MVPA": rw_values["Moderate"] / epoch_to_mins +
                                           rw_values["Vigorous"] / epoch_to_mins
                                }

        # DAILY ACTIVITY ---------------------------------------------------------------------------------------------
        dates = set([i.date() for i in self.df_epoch["Timestamp"]])
        self.df_epoch["Date"] = [i.date() for i in self.df_epoch["Timestamp"]]

        daily_totals = []

        for date in sorted(dates):
            df = self.df_epoch.loc[self.df_epoch["Date"] == date]

            # Left wrist
            lw_values = df["LW_Intensity"].value_counts()

            if "Light" not in lw_values.keys():
                lw_values["Light"] = 0
            if "Moderate" not in lw_values.keys():
                lw_values["Moderate"] = 0
            if "Vigorous" not in lw_values.keys():
                lw_values["Vigorous"] = 0

            lw_values = lw_values/4

            # Right wrist
            rw_values = df["RW_Intensity"].value_counts()

            if "Light" not in rw_values.keys():
                rw_values["Light"] = 0
            if "Moderate" not in rw_values.keys():
                rw_values["Moderate"] = 0
            if "Vigorous" not in rw_values.keys():
                rw_values["Vigorous"] = 0

            rw_values = rw_values/4

            daily_data = [date, lw_values["Sedentary"], lw_values["Light"], lw_values["Moderate"],
                          lw_values["Vigorous"], lw_values["Moderate"] + lw_values["Vigorous"],
                          rw_values["Sedentary"], rw_values["Light"], rw_values["Moderate"],
                          rw_values["Vigorous"], rw_values["Moderate"] + rw_values["Vigorous"]
                          ]
            daily_totals.append(daily_data)

        self.df_daily = pd.DataFrame(daily_totals,
                                     columns=["Date",
                                              "LW_Sedentary", "LW_Light", "LW_Moderate", "LW_Vigorous", "LW_MVPA",
                                              "RW_Sedentary", "RW_Light", "RW_Moderate", "RW_Vigorous", "RW_MVPA"])

        # Adds totals as final row
        final_row = pd.DataFrame(list(zip(["TOTAL",
                                           self.activity_totals["LW_Sedentary"], self.activity_totals["LW_Light"],
                                           self.activity_totals["LW_Moderate"], self.activity_totals["LW_Vigorous"],
                                           self.activity_totals["LW_MVPA"],
                                           self.activity_totals["RW_Sedentary"], self.activity_totals["RW_Light"],
                                           self.activity_totals["RW_Moderate"], self.activity_totals["RW_Vigorous"],
                                           self.activity_totals["RW_MVPA"]
                                           ])),
                                 index=["Date",
                                        "LW_Sedentary", "LW_Light", "LW_Moderate",
                                        "LW_Vigorous", "LW_MVPA",
                                        "RW_Sedentary", "RW_Light", "RW_Moderate",
                                        "RW_Vigorous", "RW_MVPA"]).transpose()

        self.df_daily = self.df_daily.append(final_row)
        self.df_daily = self.df_daily.reset_index()
        self.df_daily = self.df_daily.drop("index", axis=1)

        # Removes date column
        self.df_epoch = self.df_epoch.drop("Date", axis=1)

        print("Complete.")

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

        # Finds data that contains timestamps
        timestamps = None

        if self.lw is not None:
            timestamps = self.lw.timestamps[::self.epoch_len * self.lw_fs]

        if self.rw is not None and timestamps is None:
            timestamps = self.rw.timestamps[::self.epoch_len * self.rw_fs]

        if self.la is not None and timestamps is None:
            timestamps = self.la.timestamps[::self.epoch_len * self.la_fs]

        if self.ra is not None and timestamps is None:
            timestamps = self.ra.timestamps[::self.epoch_len * self.ra_fs]

        # Creates empty lists for df creation
        if None in [self.lw, self.rw, self.la, self.ra]:
            self.fill_missing_data()

        df = pd.DataFrame(list(zip(timestamps,
                                   self.lw_svm, self.lw_avm, self.lw_temperature,
                                   self.rw_svm, self.rw_avm, self.rw_temperature,
                                   self.la_svm, self.la_avm, self.la_temperature,
                                   self.ra_svm, self.ra_avm, self.ra_temperature)),
                          columns=["Timestamp", "LW_SVM", "LW_AVM", "LW_Temp",
                                   "RW_SVM", "RW_AVM", "RW_Temp",
                                   "LA_SVM", "LA_AVM", "LA_Temp",
                                   "RA_SVM", "RA_AVM", "RA_Temp"])

        del self.lw_svm, self.lw_avm, self.rw_svm, self.rw_avm, self.la_svm, self.la_avm, self.ra_svm, self.ra_avm

        print("Complete.")

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

        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        return df

    def plot_epoched(self, outcome_measure="SVM", show_cutpoints=False):

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex='col', figsize=(12, 7))
        plt.suptitle("{}-second epoch data".format(self.epoch_len))
        plt.subplots_adjust(hspace=.25)

        ax1.plot(self.df_epoch["Timestamp"], self.df_epoch["LW_{}".format(outcome_measure)], color='black')
        ax1.set_title("Left Wrist")
        ax1.set_ylabel("G*s/{} sec".format(self.epoch_len))

        if show_cutpoints and outcome_measure == "SVM":
            ax1.axhline(self.lw_cutpoints["Light"], color='green')
            ax1.axhline(self.lw_cutpoints["Moderate"], color='orange')
            ax1.axhline(self.lw_cutpoints["Vigorous"], color='red')

        ax2.plot(self.df_epoch["Timestamp"], self.df_epoch["RW_{}".format(outcome_measure)], color='red')
        ax2.set_title("Right Wrist")
        ax2.set_ylabel("G*s/{} sec".format(self.epoch_len))

        if show_cutpoints and outcome_measure == "SVM":
            ax2.axhline(self.rw_cutpoints["Light"], color='green')
            ax2.axhline(self.rw_cutpoints["Moderate"], color='orange')
            ax2.axhline(self.rw_cutpoints["Vigorous"], color='red')

        ax3.plot(self.df_epoch["Timestamp"], self.df_epoch["LA_{}".format(outcome_measure)], color='black')
        ax3.set_title("Left Ankle")
        ax3.set_ylabel("G*s/{} sec".format(self.epoch_len))

        ax4.plot(self.df_epoch["Timestamp"], self.df_epoch["RA_{}".format(outcome_measure)], color='red')
        ax4.set_title("Right Ankle")
        ax4.set_ylabel("G*s/{} sec".format(self.epoch_len))

        ax4.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)

    def plot_daily_activity(self):

        df = self.df_daily.iloc[:-1]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12, 8), sharex='col')
        plt.subplots_adjust(hspace=.35)

        labels = df["Date"]
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        ax1.bar(x - width / 2, df["LW_Sedentary"], width, color='dodgerblue', edgecolor='black', label="LW")
        ax1.bar(x + width / 2, df["RW_Sedentary"], width, color='red', edgecolor='black', label="RW")
        ax1.set_ylabel("Minutes")
        ax1.set_title("Sedentary")
        ax1.legend()

        ax2.bar(x - width / 2, df["LW_Light"], width, color='dodgerblue', edgecolor='black')
        ax2.bar(x + width / 2, df["RW_Light"], width, color='red', edgecolor='black')

        ax2.set_title("Light")

        ax2.set_ylabel("Minutes")

        ax3.bar(x - width / 2, df["LW_Moderate"], width, color='dodgerblue', edgecolor='black')
        ax3.bar(x + width / 2, df["RW_Moderate"], width, color='red', edgecolor='black')
        ax3.set_title("Moderate")
        ax3.set_ylabel("Minutes")

        ax4.bar(x - width / 2, df["LW_Vigorous"], width, color='dodgerblue', edgecolor='black', label="LW")
        ax4.bar(x + width / 2, df["RW_Vigorous"], width, color='red', edgecolor='black', label="RW")
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.set_title("Vigorous")
        ax4.set_ylabel("Minutes")


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
               # processed_filepath="C:/Users/ksweber/Desktop/OND06_SHB_9844_EpochedAccelerometer.csv",
               processed_filepath=None,
               from_processed=False,

               output_dir="C:/Users/ksweber/Desktop/")


# TODO

# add temp to epoched data --> jumping window average
    # df_epoch needs updating
# format epoch_df timestamps (round to second)
