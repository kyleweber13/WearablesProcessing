"""
import SleepData
import ModelStats
import ValidData
import ImportCropIndexes
import HRAcc
import DailyReports
"""

import ImportEDF
import DeviceSync
import ECG
import Accelerometer
import Nonwear

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
import warnings
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
warnings.filterwarnings("ignore")


# Gets Desktop pathway; used as default write directory
try:
    desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop/')
except KeyError:
    desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop/')

# ====================================================== SUBJECT CLASS ================================================


class Subject:

    def __init__(self, from_processed=True, raw_edf_folder=None, subject_id=None,
                 load_wrist=False, load_ankle=False, load_ecg=False,
                 load_raw_ecg=False, load_bittium_accel=False, load_raw_ankle=False, load_raw_wrist=False,
                 epoch_len=15, remove_epoch_baseline=False,
                 rest_hr_window=60, n_epochs_rest_hr=30, hracc_threshold=30,
                 output_dir=desktop_path, processed_folder=None,
                 write_results=False, treadmill_log_file=None,
                 nonwear_log_file=None,
                 demographics_file=None, sleeplog_file=None):

        print()
        print("========================================= SUBJECT #{} "
              "=============================================".format(subject_id))
        print()

        # ============================================== DEFAULT VALUES ===============================================

        # Model objects
        self.wrist, self.ankle, self.ecg, self.hr_acc = None, None, None, None

        self.subject_id = subject_id  # 4-digit ID code
        self.raw_edf_folder = raw_edf_folder  # Folder where raw EDF files are stored

        # Processing variables
        self.epoch_len = epoch_len  # Epoch length; seconds
        self.remove_epoch_baseline = remove_epoch_baseline  # Removes lowest value from epoched data to remove bias
        self.rest_hr_window = rest_hr_window  # Number of epochs over which rolling average HR is calculated
        self.n_epochs_rest_hr = n_epochs_rest_hr  # Number of epochs over which resting HR is calculate
        self.hracc_threshold = hracc_threshold  # Threshold for HR-Acc model

        self.from_processed = from_processed  # Whether to import already-processed data
        self.write_results = write_results  # Whether to write results to CSV

        if self.from_processed:  # overrides write_results if reading from processed
            self.write_results = False

        self.output_dir = output_dir  # Where files are written
        self.processed_folder = processed_folder  # Folder that contains already-processed data files

        self.demographics_file = demographics_file  # pathway to demographics file
        self.treadmill_log_file = treadmill_log_file   # pathway to treadmill data file
        self.sleeplog_file = sleeplog_file   # pathway to sleep data file (visual inspection)
        self.nonwear_file = nonwear_log_file   # pathway to non-wear data file (visual inspection)

        # Data cropping data
        self.starttime_dict = {"Ankle": None, "Wrist": None, "ECG": None}  # timestamps for when each device started
        self.offset_dict = {"AnkleStart": 0, "AnkleEnd": 0,
                            "WristStart": 0, "WristEnd": 0,
                            "ECGStart": 0, "ECGEnd": 0}

        # Which device data to load
        self.load_wrist = load_wrist  # boolean
        self.load_raw_wrist = load_raw_wrist  # boolean
        self.wrist_filepath = None

        self.load_ankle = load_ankle
        self.load_raw_ankle = load_raw_ankle
        self.ankle_filepath = None

        self.load_ecg = load_ecg
        self.ecg_filepath = None
        self.load_raw_ecg = load_raw_ecg
        self.load_bittium_accel = load_bittium_accel

        # Boolean of whether ECG is included or not
        # Used for output file naming; important for data cropping
        if not self.load_ecg and (self.load_ankle or self.load_wrist):
            self.accel_only = True
        if self.load_ecg:
            self.accel_only = False

        # Objects that contain data from valid epochs only
        self.valid_all = None
        self.valid_accelonly = None

        # Default demographics
        self.demographics = {"Age": 40, "Sex": None, "Weight": 1, "Height": 1, "Hand": "Right", "RestVO2": 3.5}

        # Stats + summary objects
        self.stats = None
        self.daily_summary = None

        # =============================================== RUNS METHODS ================================================

        # Retrieves filenames based on specified data to import
        self.wrist_filepath, self.wrist_temperature_filepath, \
            self.ankle_filepath, self.ecg_filepath = self.get_raw_filepaths()

        # Imports demographic information
        self.import_demographics()

        # Calculates data cropping indexes based on files to import
        self.crop_files()

        # Creates device objects
        self.create_device_objects()

        # Sleep data
        # self.sleep = SleepData.Sleep(subject_object=self)

        # Non-wear data
        self.nonwear = Nonwear.NonwearLog(subject_object=self)  # Manually-logged

    def get_raw_filepaths(self):
        """Retrieves filenames associated with current subject."""

        print("Checking {} for files...".format(self.raw_edf_folder))

        subject_file_list = [i for i in os.listdir(self.raw_edf_folder) if
                             (".EDF" in i or ".edf" in i)
                             and i.count("_") >= 2
                             and str(self.subject_id) == str(i.split("_")[2])]

        # Returns Nones if no files found
        if len(subject_file_list) == 0:
            print("-No files found for this subject ID.")
            return None, None, None, None

        dom_hand = self.demographics["Hand"][0]

        wrist_filename, wrist_temperature_filename, ankle_filename, ecg_filename = None, None, None, None

        # Loads wrist data --------------------------------------------------------------------------------------------
        if self.load_wrist:

            # Subset of wrist file(s) from all subject files
            wrist_filenames = [self.raw_edf_folder + i for i in subject_file_list
                               if "Wrist" in i and "Accelerometer" in i]
            wrist_temperature_filenames = [self.raw_edf_folder + i for i in subject_file_list
                                           if "Wrist" in i and "Temperature" in i]

            # Selects correct wrist file
            if len(wrist_filenames) == 2:
                wrist_filename = [i for i in wrist_filenames if dom_hand + "Wrist" not in i][0]
            if len(wrist_filenames) == 1:
                wrist_filename = wrist_filenames[0]
            if len(wrist_filenames) == 0:
                print("-Could not find the correct wrist accelerometer file.")
                wrist_filename = None
                self.load_wrist = False

            # Selects correct wrist temperature file
            if len(wrist_temperature_filenames) == 2:
                wrist_temperature_filename = [i for i in wrist_temperature_filenames if dom_hand + "Wrist" not in i][0]
            if len(wrist_temperature_filenames) == 1:
                wrist_temperature_filename = wrist_temperature_filenames[0]
            if len(wrist_temperature_filenames) == 0:
                print("-Could not find the correct wrist temperature file.")
                wrist_temperature_filename = None

        # Loads ankle data --------------------------------------------------------------------------------------------
        if self.load_ankle:
            ankle_filenames = [self.raw_edf_folder + i for i in subject_file_list if "Ankle" in i]

            # Selects correct ankle file
            if len(ankle_filenames) == 2:
                ankle_filename = [i for i in ankle_filenames if dom_hand + "Ankle" not in i][0]
            if len(ankle_filenames) == 1:
                ankle_filename = ankle_filenames[0]
            if len(ankle_filenames) == 0:
                print("-Could not find the correct ankle accelerometer file.")
                ankle_filename = None
                self.load_ankle = None

        # Loads ECG data --------------------------------------------------------------------------------------------
        if self.load_ecg:
            ecg_filename = [self.raw_edf_folder + i for i in subject_file_list if "BF" in i][0]

            if len([self.raw_edf_folder + i for i in subject_file_list if "BF" in i]) == 0:
                print("Could not find the correct ECG file.")
                ecg_filename = None
                self.load_ecg = None

        return wrist_filename, wrist_temperature_filename, ankle_filename, ecg_filename

    def import_demographics(self):
        """Function that imports demographics data from spreadsheet for desired participants. Works with either
           .csv or .xlsx file types. If any data is missing, will use default values declared in __init__ method.

        :returns
        -demos_dict: dictionary containing demographics information
        """

        print("\nChecking for demographics information...")

        # Check to ensure file exists ---------------------------------------------------------------------------------
        if self.demographics_file is None:
            print("-No demographics file input.")
            return None
        if not os.path.exists(self.demographics_file):
            print("-Demographics file does not exist.")
            return None

        # Loads correct demographics format: xlsx or csv --------------------------------------------------------------
        demos_file_ext = self.demographics_file.split("/")[-1].split(".")[-1]

        if demos_file_ext == "csv" or demos_file_ext == "CSV":
            data = pd.read_csv(self.demographics_file)
        if demos_file_ext == "xlsx" or demos_file_ext == "XLSX":
            data = pd.read_excel(self.demographics_file)

        for i in range(data.shape[0]):
            data_row = data.iloc[i]
            if str(self.subject_id) == data_row["SUBJECT"].split("_")[2]:

                print("-Demographics information found for subject {}.".format(self.subject_id))

                # Sets resting VO2 according to Kwan et al. (2004) values based on age/sex
                missing_value = False

                try:
                    self.demographics["Age"] = int(data_row["AGE"])
                except ValueError:
                    print("-No age was found. Default is 40 years.")
                    missing_value = True

                try:
                    self.demographics["Sex"] = data_row["SEX"]
                except ValueError:
                    print("-No sex was specified.")
                    missing_value = True

                try:
                    self.demographics["Weight"] = int(data_row["WEIGHT"])
                except ValueError:
                    print("-No weight was specified. Default is 1kg.")
                    missing_value = True

                try:
                    self.demographics["Height"] = int(data_row["HEIGHT"])
                except ValueError:
                    print("-No height was specified. Default is 1.00 m.")
                    missing_value = True

                try:
                    self.demographics["Hand"] = data_row["HANDEDNESS"]
                except ValueError:
                    print("-No handedness was specified. Default is right-handed.")
                    missing_value = True

                if self.demographics["Age"] < 65 and self.demographics["Sex"] == "Male":
                    rvo2 = 3.03
                if self.demographics["Age"] < 65 and self.demographics["Sex"] == "Female":
                    rvo2 = 3.32
                if self.demographics["Age"] >= 65 and self.demographics["Sex"] == "Male":
                    rvo2 = 2.84
                if self.demographics["Age"] >= 65 and self.demographics["Sex"] == "Female":
                    rvo2 = 2.82

                self.demographics["BMI"] = round(float((self.demographics["Weight"] /
                                                        (self.demographics["Height"] / 100) ** 2)), 2)

                if self.demographics["Sex"] is not None:
                    self.demographics["RestVO2"] = rvo2

                if not missing_value:
                    print("-No demographics data are missing.")

    def crop_files(self):
        """Method that checks timestamps from all EDF files and determines how many data points to crop off start/end
           of files so all device data starts and stops at the same time.
           Updates crop indexes into dictionaries.
        """
        # Skips procedure if reading data from processed (already cropped) -------------------------------------------
        if self.from_processed:
            print("\nData is being imported from processed. Skipping file crop.")
            return None

        # Performs procedure if reading from raw ---------------------------------------------------------------------
        if not self.from_processed:
            print("\nChecking file start/end times to perform file crop...")

            # File summaries
            print("\nRaw EDF file summaries:")
            ImportEDF.check_file(self.ankle_filepath, print_summary=True)
            ImportEDF.check_file(self.wrist_filepath, print_summary=True)
            ImportEDF.check_file(self.ecg_filepath, print_summary=True)

            # Default values for dictionaries
            start_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}
            end_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}

            # If only one device available
            if self.load_ecg + self.load_wrist + self.load_ankle == 1:
                # Leaves values as all 0 (defaults)
                pass

            # If ECG and at least one accelerometer are available
            if self.ecg_filepath is not None and (self.wrist_filepath is not None or self.ankle_filepath is not None):

                # Reads data from raw if crop file not available or no data found for participant
                start_dict = DeviceSync.crop_start(subject_object=self)
                end_dict = DeviceSync.crop_end(subject_object=self, start_offset_dictionary=start_dict)

            # If ECG not available but wrist and ankle accelerometers are
            if self.ecg_filepath is None and self.wrist_filepath is not None and self.ankle_filepath is not None:

                # Reads from raw if participant not found in csv or csv does not exist
                start_dict = DeviceSync.crop_start(subject_object=self)

                # Overwrites end indexes with values from raw accel files (excludes ECG)
                end_dict = DeviceSync.crop_end(subject_object=self, start_offset_dictionary=start_dict)

            # Updates dictionary
            self.offset_dict["AnkleStart"] = start_dict["Ankle"]
            self.offset_dict["AnkleEnd"] = end_dict["Ankle"]
            self.offset_dict["WristStart"] = start_dict["Wrist"]
            self.offset_dict["WristEnd"] = end_dict["Wrist"]
            self.offset_dict["ECGStart"] = start_dict["ECG"]
            self.offset_dict["ECGEnd"] = end_dict["ECG"]

            print("-Start indexes: ankle = {}, wrist = {}, ECG = {}".format(self.offset_dict["AnkleStart"],
                                                                            self.offset_dict["WristStart"],
                                                                            self.offset_dict["ECGStart"]))
            print("-Data points to be read: ankle = {}, wrist = {}, ECG = {}".format(self.offset_dict["AnkleEnd"],
                                                                                     self.offset_dict["WristEnd"],
                                                                                     self.offset_dict["ECGEnd"]))

    def create_device_objects(self):

        # Reads in ECG data
        if self.ecg_filepath is not None:
            self.ecg = ECG.ECG(filepath=self.ecg_filepath,
                               from_processed=self.from_processed, load_raw=self.load_raw_ecg,
                               load_accel=self.load_bittium_accel,
                               output_dir=self.output_dir, write_results=self.write_results, epoch_len=self.epoch_len,
                               start_offset=self.offset_dict["ECGStart"], end_offset=self.offset_dict["ECGEnd"],
                               age=self.demographics["Age"],
                               rest_hr_window=self.rest_hr_window, n_epochs_rest=self.n_epochs_rest_hr)

        # Objects from Accelerometer script ---------------------------------------------------------------------------

        # Wrist accelerometer
        if self.wrist_filepath is not None:
            self.wrist = Accelerometer.Wrist(subject_id=self.subject_id,
                                             filepath=self.wrist_filepath,
                                             temperature_filepath=self.wrist_temperature_filepath,
                                             load_raw=self.load_raw_wrist,
                                             epoch_len=self.epoch_len,
                                             accel_only=self.accel_only,
                                             output_dir=self.output_dir,
                                             processed_folder=self.processed_folder, 
                                             from_processed=self.from_processed,
                                             write_results=self.write_results,
                                             start_offset=self.offset_dict["WristStart"],
                                             end_offset=self.offset_dict["WristEnd"],
                                             ecg_object=self.ecg)

        # Ankle acclerometer
        if self.ankle_filepath is not None:
            self.ankle = Accelerometer.Ankle(subject_id=self.subject_id, epoch_len=self.epoch_len,
                                             filepath=self.ankle_filepath, load_raw=self.load_raw_ankle,
                                             output_dir=self.output_dir, accel_only=self.accel_only,
                                             remove_baseline=self.remove_epoch_baseline,
                                             processed_folder=self.processed_folder,
                                             from_processed=self.from_processed,
                                             treadmill_log_file=self.treadmill_log_file,
                                             write_results=self.write_results,
                                             start_offset=self.offset_dict["AnkleStart"],
                                             end_offset=self.offset_dict["AnkleEnd"],
                                             age=self.demographics["Age"], rvo2=self.demographics["RestVO2"],
                                             bmi=self.demographics["BMI"], ecg_object=self.ecg)

        # No files
        if self.ankle_filepath is None and self.wrist_filepath is None and self.ecg_filepath is None:
            print("No files were imported.")
            return None
