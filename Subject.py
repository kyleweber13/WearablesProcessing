"""
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
                 epoch_len=15, remove_epoch_baseline=False, crop_file_start=True, crop_file_end=False,
                 rest_hr_window=60, n_epochs_rest_hr=30, hracc_threshold=30,
                 output_dir=desktop_path, processed_folder=None,
                 write_results=False, treadmill_log_file=None,
                 nonwear_log_file=None, sleeplog_file=None,
                 demographics_file=None):

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
        self.crop_file_start = crop_file_start  # Whether to crop all devices to start at same time
        self.crop_file_end = crop_file_end  # Whether to crop all devices to end at same time

        self.from_processed = from_processed  # Whether to import already-processed data
        self.write_results = write_results  # Whether to write results to CSV

        if self.from_processed:  # overrides write_results if reading from processed
            self.write_results = False

        self.output_dir = output_dir  # Where files are written
        self.processed_folder = processed_folder  # Folder that contains already-processed data files

        self.demographics_file = demographics_file  # pathway to demographics file
        self.treadmill_log_file = treadmill_log_file  # pathway to treadmill data file
        self.sleeplog_file = sleeplog_file  # pathway to sleep data file
        self.nonwear_file = nonwear_log_file  # pathway to non-wear data file

        # Data cropping data
        self.starttime_dict = {"Ankle": None, "Wrist": None, "ECG": None, "Overall": None}  # device start timestamps
        self.offset_dict = {"AnkleStart": 0, "AnkleEnd": 0,
                            "WristStart": 0, "WristEnd": 0,
                            "ECGStart": 0, "ECGEnd": 0}
        self.epoch_offset_dict = {"AnkleStart": 0, "AnkleEnd": 0,
                                  "WristStart": 0, "WristEnd": 0,
                                  "ECGStart": 0, "ECGEnd": 0}
        self.start_timestamp = None
        self.data_len = 0

        # Which device data to load
        self.load_wrist = load_wrist  # boolean
        self.load_raw_wrist = load_raw_wrist  # boolean
        self.wrist_filepath = None  # pathway to wrist EDF file
        self.wrist_temperature_filepath = None  # pathway to wrist temperature EDF file
        self.wrist_filename = None  # file name, no extension, EDF file
        self.wrist_proc_filepath = None  # pathway to processed wrist file (csv/xlsx)
        self.wrist_temp_proc_filepath = None  # pathway to processed wrist temperature file (csv/xlsx)
        self.wrist_temp_filename = None  # file name, no extension, wrist temperature file

        self.load_ankle = load_ankle
        self.load_raw_ankle = load_raw_ankle
        self.ankle_filepath = None
        self.ankle_proc_filepath = None
        self.ankle_filename = None

        self.load_ecg = load_ecg
        self.ecg_filepath = None
        self.load_raw_ecg = load_raw_ecg
        self.load_bittium_accel = load_bittium_accel
        self.ecg_proc_filepath = None
        self.ecg_filename = None

        # Boolean of whether ECG is included or not
        # Used for output file naming; important for data cropping
        if not self.load_ecg and (self.load_ankle or self.load_wrist):
            self.accel_only = True
        if self.load_ecg:
            self.accel_only = False

        self.epoch_df = None
        self.sleep = None
        self.nonwear = None

        # Objects that contain data from valid epochs only
        self.valid_all = None
        self.valid_accelonly = None

        # Default demographics
        self.demographics = {"Age": 40, "Sex": None, "Weight": 1, "Height": 1, "Hand": "Right", "RestVO2": 3.5}

        # Stats + summary objects
        self.stats = None
        self.daily_summary = None

    def get_edf_filepaths(self):
        """Retrieves EDF filenames associated with current subject."""

        if self.load_raw_wrist + self.load_raw_ankle + self.load_raw_ecg >= 1:
            print("Checking {} for EDF files...".format(self.raw_edf_folder))

        # Default values to return if no file(s) found
        # wrist_filename, wrist_temperature_filename, ankle_filename, ecg_filename = None, None, None, None

        # List of all files with subject_id in filename
        subject_file_list = [i for i in os.listdir(self.raw_edf_folder) if
                             (".EDF" in i or ".edf" in i)
                             and i.count("_") >= 2
                             and str(self.subject_id) == str(i.split("_")[2])]

        # Returns Nones if no files found
        if len(subject_file_list) == 0:
            print("-No files found for this subject ID.")

            self.load_raw_wrist, self.load_raw_ankle, self.load_raw_ecg = False, False, False

        dom_hand = self.demographics["Hand"][0]

        # Loads wrist data --------------------------------------------------------------------------------------------
        if self.load_wrist and self.load_raw_wrist:

            # Subset of wrist file(s) from all subject files
            wrist_filenames = [self.raw_edf_folder + i for i in subject_file_list
                               if "Wrist" in i and "Accelerometer" in i]
            wrist_temperature_filenames = [self.raw_edf_folder + i for i in subject_file_list
                                           if "Wrist" in i and "Temperature" in i]

            # Selects non-dominant wrist file if right and left available
            if len(wrist_filenames) == 2:
                self.wrist_filepath = [i for i in wrist_filenames if dom_hand + "Wrist" not in i][0]
            if len(wrist_filenames) == 1:
                self.wrist_filepath = wrist_filenames[0]
            if len(wrist_filenames) == 0:
                print("-Could not find the correct wrist accelerometer file.")
                self.wrist_filepath = None
                self.load_wrist = False

            if self.wrist_filepath is not None:
                print("-Found {}".format(self.wrist_filepath.split("/")[-1]))

            if len(wrist_temperature_filenames) == 2:
                self.wrist_temperature_filepath = [i for i in wrist_temperature_filenames if
                                                   dom_hand + "Wrist" not in i][0]
            if len(wrist_temperature_filenames) == 1:
                self.wrist_temperature_filepath = wrist_temperature_filenames[0]
            if len(wrist_temperature_filenames) == 0:
                print("-Could not find the correct wrist temperature file.")
                self.wrist_temperature_filepath = None

            if self.wrist_temperature_filepath is not None:
                print("-Found {}".format(self.wrist_temperature_filepath.split("/")[-1]))

        # Loads ankle data --------------------------------------------------------------------------------------------
        if self.load_ankle and self.load_raw_ankle:
            ankle_filenames = [self.raw_edf_folder + i for i in subject_file_list if "Ankle" in i]

            # Selects non-dominant ankle file if right and left available
            if len(ankle_filenames) == 2:
                self.ankle_filepath = [i for i in ankle_filenames if dom_hand + "Ankle" not in i][0]
            if len(ankle_filenames) == 1:
                self.ankle_filepath = ankle_filenames[0]
            if len(ankle_filenames) == 0:
                print("-Could not find the correct ankle accelerometer file.")
                self.ankle_filepath = None
                self.load_ankle = None

            if self.ankle_filepath is not None:
                print("-Found {}".format(self.ankle_filepath.split("/")[-1]))

        # Loads ECG data --------------------------------------------------------------------------------------------
        if self.load_ecg and self.load_raw_ecg:
            ecg_filename = [self.raw_edf_folder + i for i in subject_file_list if "BF" in i]

            if len([self.raw_edf_folder + i for i in subject_file_list if "BF" in i]) == 0:
                print("-Could not find the correct ECG file.")
                self.ecg_filepath = None
                self.load_ecg = None

            if len(ecg_filename) == 1:
                self.ecg_filepath = ecg_filename[0]
                print("-Found {}".format(self.ecg_filepath.split("/")[-1]))

        # Sets filenames from file pathways --------------------------------------------------------------------------
        if self.wrist_filepath is not None:
            self.wrist_filename = self.wrist_filepath.split("/")[-1]

        if self.wrist_temperature_filepath is not None:
            self.wrist_temp_filename = self.wrist_temperature_filepath.split("/")[-1]

        if self.ankle_filepath is not None:
            self.ankle_filename = self.ankle_filepath.split("/")[-1]

        if self.ecg_filepath is not None:
            self.ecg_filename = self.ecg_filepath.split("/")[-1]

    def get_processed_filepaths(self):

        if not self.from_processed:
            return None

        print("\nChecking {} for processed files...".format(self.processed_folder))

        subject_file_list = [i for i in os.listdir(self.processed_folder) if
                             ("csv" in i or "CSV" in i)
                             and i.count("_") >= 2
                             and str(self.subject_id) == str(i.split("_")[2])]

        # Returns Nones if no files found
        if len(subject_file_list) == 0:
            print("-No processed files found for this subject ID.")

            # Sets from_processed to False if no files found
            self.from_processed = False
            self.wrist_proc_filepath, self.ankle_proc_filepath, self.ecg_proc_filepath = None, None, None

        dom_hand = self.demographics["Hand"][0]

        # Loads wrist data --------------------------------------------------------------------------------------------
        if self.load_wrist:

            # Subset of wrist file(s) from all subject files
            wrist_filenames = [self.processed_folder + i for i in subject_file_list
                               if "Wrist" in i and "Accelerometer" in i]
            wrist_temperature_filenames = [self.processed_folder + i for i in subject_file_list
                                           if "Wrist" in i and "Temperature" in i]

            # Selects correct wrist file
            if len(wrist_filenames) == 2:
                self.wrist_proc_filepath = [i for i in wrist_filenames if dom_hand + "Wrist" not in i][0]
            if len(wrist_filenames) == 1:
                self.wrist_proc_filepath = wrist_filenames[0]
            if len(wrist_filenames) == 0:
                print("-Could not find the correct wrist accelerometer file.")
                self.wrist_proc_filepath = None
                self.load_raw_wrist = False

            if self.wrist_proc_filepath is not None:
                print("-Found {}".format(self.wrist_proc_filepath.split("/")[-1]))

            # Selects correct wrist temperature file
            if len(wrist_temperature_filenames) == 2:
                if self.accel_only:
                    self.wrist_temp_proc_filepath = \
                    [i for i in wrist_temperature_filenames if dom_hand + "Wrist" not in i and "AccelOnly" in i][0]
                if not self.accel_only:
                    self.wrist_temp_proc_filepath = \
                    [i for i in wrist_temperature_filenames if dom_hand + "Wrist" not in i and "AccelOnly" not in i][0]

            if len(wrist_temperature_filenames) == 1:
                self.wrist_temp_proc_filepath = wrist_temperature_filenames[0]
            if len(wrist_temperature_filenames) == 0:
                print("-Could not find the correct wrist temperature file.")
                self.wrist_temp_proc_filepath = None

            if self.wrist_temp_proc_filepath is not None:
                print("-Found {}".format(self.wrist_temp_proc_filepath.split("/")[-1]))

        # Loads ankle data --------------------------------------------------------------------------------------------
        if self.load_ankle:
            ankle_filenames = [self.processed_folder + i for i in subject_file_list if "Ankle" in i]

            # Selects correct ankle file
            if len(ankle_filenames) == 2:
                if self.accel_only:
                    self.ankle_proc_filepath = [i for i in ankle_filenames if dom_hand + "Ankle" not in i and
                                                "AccelOnly" in i][0]
                if not self.accel_only:
                    self.ankle_proc_filepath = [i for i in ankle_filenames if dom_hand + "Ankle" not in i and
                                                "AccelOnly" not in i][0]
            if len(ankle_filenames) == 1:
                self.ankle_proc_filepath = ankle_filenames[0]
            if len(ankle_filenames) == 0:
                print("-Could not find the correct ankle accelerometer file.")
                self.ankle_proc_filepath = None
                self.load_raw_ankle = False

            if self.ankle_proc_filepath is not None:
                print("-Found {}".format(self.ankle_proc_filepath.split("/")[-1]))

        # Loads ECG data --------------------------------------------------------------------------------------------
        if self.load_ecg:
            ecg_filename = [self.processed_folder + i for i in subject_file_list if "BF" in i]

            if len([self.processed_folder + i for i in subject_file_list if "BF" in i]) == 0:
                print("-Could not find the correct ECG file.")
                self.ecg_proc_filepath = None
                self.load_ecg = False

            if len(ecg_filename) == 1:
                self.ecg_proc_filepath = ecg_filename[0]
                print("-Found {}".format(self.ecg_proc_filepath.split("/")[-1]))

            # If mulitple data collection files found
            if len(ecg_filename) > 1:
                trial_num = subject_file_list[0].split("_")[3]
                self.ecg_proc_filepath = [i for i in ecg_filename if i.split("/")[-1].split("_")[3] == trial_num][0]

        # Sets filenames that weren't set using EDF pathways ---------------------------------------------------------
        if self.wrist_filepath is None and self.load_wrist and self.wrist_proc_filepath is not None:
            self.wrist_filename = self.wrist_proc_filepath.split("Accelerometer")[0].split("/")[-1] + "Accelerometer"

        if self.ankle_filepath is None and self.load_ankle and self.ankle_proc_filepath is not None:
            self.ankle_filename = self.ankle_proc_filepath.split("Accelerometer")[0].split("/")[-1] + "Accelerometer"

        if self.ecg_filepath is None and self.load_ecg and self.ecg_proc_filepath is not None:
            self.ecg_filename = self.ecg_proc_filepath.split("BF")[0].split("/")[-1] + "BF"

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

        # Gets row of data for correct participant
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

        if "BMI" not in self.demographics.keys():
            self.demographics["BMI"] = 1

    def crop_files(self):
        """Method that checks timestamps from all EDF files and determines how many data points to crop off start/end
           of files so all device data starts and stops at the same time.
           Updates crop indexes into dictionaries.
        """

        # Default values for dictionaries
        start_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}
        end_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}

        if not self.crop_file_start and not self.crop_file_end:
            print("\nData files are not being cropped.")

        # Data cropping ===============================================================================================
        if self.crop_file_start or self.crop_file_end:
            print("\n------------------------------------------------------------------------------------------------")
            print("Checking file start/end times to perform file crop...")

            # Skips procedure if reading data from processed (already cropped) ---------------------------------------
            if self.load_raw_wrist + self.load_raw_ankle + self.load_raw_ecg == 0:
                print("\nNo raw data are being imported. Skipping file crop.")
                return None

            # Performs procedure if reading from raw -----------------------------------------------------------------
            if self.load_wrist + self.load_raw_ankle + self.load_raw_ecg >= 1:

                # File summaries
                print("\nRaw EDF file summaries:")
                ankle_start, ankle_end, ankle_fs = ImportEDF.check_file(self.ankle_filepath, print_summary=True)
                wrist_start, wrist_end, wrist_fs = ImportEDF.check_file(self.wrist_filepath, print_summary=True)
                ecg_start, ecg_end, ecg_fs = ImportEDF.check_file(self.ecg_filepath, print_summary=True)

                self.starttime_dict = {"Ankle": ankle_start, "Wrist": wrist_start, "ECG": ecg_start}
                self.starttime_dict ["Overall"] = max([i for i in self.starttime_dict.values() if i is not None])

                # If only one device available
                if self.load_ecg + self.load_wrist + self.load_ankle == 1:
                    # Leaves values as all 0 (defaults)
                    pass

                # If ECG and at least one accelerometer are available
                if self.ecg_filepath is not None and \
                        (self.wrist_filepath is not None or self.ankle_filepath is not None):
                    # Reads data from raw if crop file not available or no data found for participant
                    start_dict = DeviceSync.crop_start(subject_object=self)
                    end_dict = DeviceSync.crop_end(subject_object=self, start_offset_dictionary=start_dict)

                # If ECG not available but wrist and ankle accelerometers are
                if self.ecg_filepath is None and self.wrist_filepath is not None and self.ankle_filepath is not None:
                    # Reads from raw if participant not found in csv or csv does not exist
                    start_dict = DeviceSync.crop_start(subject_object=self)

                    # Overwrites end indexes with values from raw accel files (excludes ECG)
                    end_dict = DeviceSync.crop_end(subject_object=self, start_offset_dictionary=start_dict)

                # Updates dictionaries -------------------------------------------------------------------------------
                if self.crop_file_start:
                    print("Cropping start of files.")
                    self.offset_dict["AnkleStart"] = start_dict["Ankle"]
                    self.offset_dict["WristStart"] = start_dict["Wrist"]
                    self.offset_dict["ECGStart"] = start_dict["ECG"]

                    print("-Start indexes: ankle = {}, wrist = {}, ECG = {}".format(self.offset_dict["AnkleStart"],
                                                                                    self.offset_dict["WristStart"],
                                                                                    self.offset_dict["ECGStart"]))
                if self.crop_file_end:
                    print("Cropping end of files.")
                    self.offset_dict["AnkleEnd"] = end_dict["Ankle"]
                    self.offset_dict["WristEnd"] = end_dict["Wrist"]
                    self.offset_dict["ECGEnd"] = end_dict["ECG"]

                    print("-Data points to be read: ankle = {}, "
                          "wrist = {}, ECG = {}".format(self.offset_dict["AnkleEnd"],
                                                        self.offset_dict["WristEnd"],
                                                        self.offset_dict["ECGEnd"]))

    def get_data_len(self):

        # MODEL DATA =================================================================================================
        # Sets length of data (number of epochs) and timestamps based on any data that is available
        try:
            self.data_len = len(self.ankle.epoch.timestamps)
            self.start_timestamp = self.ankle.epoch.timestamps[0]
        except AttributeError:
            try:
                self.data_len = len(self.wrist.epoch.timestamps)
                self.start_timestamp = self.wrist.epoch.timestamps[0]
            except AttributeError:
                self.data_len = len(self.ecg.epoch_timestamps)
                self.start_timestamp = self.ecg.epoch_timestamps[0]

        try:
            self.start_timestamp = datetime.strptime(str(self.start_timestamp)[:-3], "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            pass

    def create_device_objects(self):

        # Reads in ECG data
        if self.load_ecg and (self.ecg_filepath is not None or self.ecg_proc_filepath is not None):
            self.ecg = ECG.ECG(filepath=self.ecg_filepath,
                               filename=self.ecg_filename.split(".")[0],
                               load_raw=self.load_raw_ecg,
                               from_processed=self.from_processed,
                               processed_folder=self.processed_folder,
                               load_accel=self.load_bittium_accel,
                               epoch_len=self.epoch_len,
                               start_offset=self.offset_dict["ECGStart"], end_offset=self.offset_dict["ECGEnd"],
                               age=self.demographics["Age"],
                               rest_hr_window=self.rest_hr_window, n_epochs_rest=self.n_epochs_rest_hr,
                               output_dir=self.output_dir, write_results=self.write_results)

        # Objects from Accelerometer script ---------------------------------------------------------------------------

        # Wrist accelerometer
        if self.load_wrist and (self.wrist_filepath is not None or self.wrist_proc_filepath is not None):
            self.wrist = Accelerometer.Wrist(subject_id=self.subject_id,
                                             raw_filepath=self.wrist_filepath,
                                             filename=self.wrist_filename,
                                             proc_filepath=self.wrist_proc_filepath,
                                             temperature_filepath=self.wrist_temperature_filepath,
                                             load_raw=self.load_raw_wrist,
                                             from_processed=self.from_processed,
                                             epoch_len=self.epoch_len,
                                             accel_only=self.accel_only,
                                             start_offset=self.offset_dict["WristStart"],
                                             end_offset=self.offset_dict["WristEnd"],
                                             ecg_object=self.ecg,
                                             output_dir=self.output_dir,
                                             processed_folder=self.processed_folder,
                                             write_results=self.write_results)

        # Ankle accelerometer
        if self.load_ankle and (self.ankle_filepath is not None or self.ankle_proc_filepath is not None):
            self.ankle = Accelerometer.Ankle(subject_id=self.subject_id,
                                             raw_filepath=self.ankle_filepath,
                                             proc_filepath=self.ankle_proc_filepath,
                                             filename=self.ankle_filename,
                                             load_raw=self.load_raw_ankle,
                                             from_processed=self.from_processed,
                                             epoch_len=self.epoch_len,
                                             accel_only=self.accel_only,
                                             start_offset=self.offset_dict["AnkleStart"],
                                             end_offset=self.offset_dict["AnkleEnd"],
                                             age=self.demographics["Age"],
                                             rvo2=self.demographics["RestVO2"],
                                             bmi=self.demographics["BMI"],
                                             ecg_object=self.ecg,
                                             output_dir=self.output_dir,
                                             remove_baseline=self.remove_epoch_baseline,
                                             processed_folder=self.processed_folder,
                                             treadmill_log_file=self.treadmill_log_file,
                                             write_results=self.write_results)

        # No files
        if self.ankle_filepath is None and self.wrist_filepath is None and self.ecg_filepath is None and \
            self.ankle_proc_filepath is None and self.wrist_proc_filepath is None and self.ecg_proc_filepath is None:
            print("No files were imported.")
            return None

    def create_epoch_df(self):

        print("\nCreating dataframe of all epoched data...")

        # Determines length of longest device data
        max_list = []

        check_list = []

        if self.wrist is not None:
            check_list.append(self.wrist.epoch.timestamps)
        if self.ankle is not None:
            check_list.append(self.ankle.epoch.timestamps)
        if self.ecg is not None:
            check_list.append(self.ecg.epoch_timestamps)

        # for obj in [self.wrist.epoch.timestamps, self.ankle.epoch.timestamps, self.ecg.epoch_timestamps]:
        for obj in check_list:
            try:
                max_list.append(len(obj))
            except (TypeError, AttributeError):
                pass

        max_len = max(max_list)

        # Default lists of None
        timestamps = [None for i in range(max_len)]
        wrist_svm = [None for i in range(max_len)]
        wrist_intensity_cat = [None for i in range(max_len)]
        ankle_svm = [None for i in range(max_len)]
        ankle_intensity_cat = [None for i in range(max_len)]
        ankle_pred_mets = [None for i in range(max_len)]
        ankle_pred_speed = [None for i in range(max_len)]
        epoch_hr = [None for i in range(max_len)]
        hrr = [None for i in range(max_len)]
        hr_intensity = [None for i in range(max_len)]
        ecg_validity = [None for i in range(max_len)]
        sleep_status = [None for i in range(max_len)]
        nonwear_status = [None for i in range(max_len)]

        if not self.load_ankle and self.load_wrist:
            timestamps = self.wrist.epoch.timestamps

            wrist_svm = self.wrist.epoch.svm
            wrist_intensity_cat = self.wrist.model.epoch_intensity

            ankle_svm = [None for i in range(len(self.wrist.epoch.svm))]
            ankle_intensity_cat = [None for i in range(len(self.wrist.epoch.svm))]
            ankle_pred_mets = [None for i in range(len(self.wrist.epoch.svm))]
            ankle_pred_speed = [None for i in range(len(self.wrist.epoch.svm))]

        if not self.load_wrist and self.load_ankle:
            timestamps = self.ankle.epoch.timestamps

            ankle_svm = self.ankle.epoch.svm
            ankle_intensity_cat = self.ankle.model.epoch_intensity
            ankle_pred_mets = self.ankle.model.predicted_mets
            ankle_pred_speed = self.ankle.model.linear_speed

        if self.load_wrist and self.load_ankle:
            timestamps = self.wrist.epoch.timestamps

            wrist_svm = self.wrist.epoch.svm
            wrist_intensity_cat = self.wrist.model.epoch_intensity

            ankle_svm = self.ankle.epoch.svm
            ankle_intensity_cat = self.ankle.model.epoch_intensity
            ankle_pred_mets = self.ankle.model.predicted_mets
            ankle_pred_speed = self.ankle.model.linear_speed

        """REQUIRES UPDATE"""
        if self.load_ecg:

            if not self.load_wrist and not self.load_ankle:
                timestamps = self.ecg.epoch_timestamps

            epoch_hr = self.ecg.valid_hr
            # hrr = self.ecg.perc_hrr
            # hr_intensity = self.ecg.epoch_intensity
            ecg_validity = ["Valid" if i == 0 else "Invalid" for i in self.ecg.epoch_validity]
            hr_intensity = [None for i in range(self.data_len)]

        if self.sleep.status is not None:
            sleep_status = ["Awake" if i == 0 else ("Napping" if i == 1 else "Asleep") for i in self.sleep.status]

        if self.nonwear.status is not None:
            nonwear_status = ["Wear" if i == 0 else "Nonwear" for i in self.nonwear.status]

        # Doesn't lose data due to zip shortest list
        df = pd.DataFrame({"Timestamps": pd.Series(timestamps), "Wrist_SVM": pd.Series(wrist_svm),
                          "Wrist_Intensity": pd.Series(wrist_intensity_cat), "Ankle_SVM": pd.Series(ankle_svm),
                           "Ankle_Intensity": pd.Series(ankle_intensity_cat),
                           "Ankle_Speed": pd.Series(ankle_pred_speed), "Ankle_METs": pd.Series(ankle_pred_mets),
                           "HR": pd.Series(epoch_hr), "%HRR": pd.Series(hrr),
                           "HR_Intensity": pd.Series(hr_intensity),
                           "ECG_Validity": pd.Series(ecg_validity), "Sleep_Status": pd.Series(sleep_status),
                           "Nonwear_Status": pd.Series(nonwear_status)})

        print("Complete.")

        return df

    def create_ecg_contingency_table(self, data_type="intensity", bin_size=100):
        """Creates dataframe which represents a contingency table for ECG signal validity by intensity category
           for both wrist and ankle acclerometers. Values are percentage of the time spent in each intensity.
           Used to try and find relationship between movement and invalid ECG signals.

           Able to use "data_type" argument to specify whether to calculate values based on "intensity" category or
           "counts" (bins of width "bin_size" [default = 100]).
        """

        if self.load_ecg and (self.load_wrist or self.load_ankle):
            print("\nCreating ECG signal validity contingency table based on wrist and ankle intensity data...")
        if not self.load_ecg:
            print("\nCannot create ECG signal validity contingency table based on accelerometer data.")
            print("-Please load some ECG data and try again.")
        if self.load_ecg and not self.load_wrist and not self.load_ankle:
            print("\nCannot create ECG signal validity contingency table based on accelerometer data.")
            print("-Please load some accelerometer data and try again.")

        def using_intensity():

            wrist_invalid = []
            wrist_valid = []
            ankle_invalid = []
            ankle_valid = []
            n_wrist_epochs = []
            n_ankle_epochs = []

            # Wrist data ---------------------------------------------------------------------------------------------
            if self.load_wrist and self.load_ecg:

                for intensity_cat in range(4):
                    wrist_invalid.append(self.epoch_df.loc[(self.epoch_df["Wrist_Intensity"] == intensity_cat) &
                                                           (self.epoch_df["ECG_Validity"] == "Invalid")].shape[0] /
                                         self.epoch_df.loc[self.epoch_df["Wrist_Intensity"] == intensity_cat].shape[0]
                                         * 100)
                    wrist_valid.append(self.epoch_df.loc[(self.epoch_df["Wrist_Intensity"] == intensity_cat) &
                                                         (self.epoch_df["ECG_Validity"] == "Valid")].shape[0] /
                                       self.epoch_df.loc[self.epoch_df["Wrist_Intensity"] == intensity_cat].shape[0]
                                       * 100)

                    n_wrist_epochs.append(self.epoch_df.loc[self.epoch_df["Wrist_Intensity"] == intensity_cat].shape[0])

                # MVPA epochs
                n_wrist_epochs.append(self.epoch_df.loc[self.epoch_df["Wrist_Intensity"] >= 2].shape[0])

                wrist_invalid.append(self.epoch_df.loc[(self.epoch_df["Wrist_Intensity"] >= 2) &
                                                       (self.epoch_df["ECG_Validity"] == "Invalid")].shape[0] /
                                     self.epoch_df.loc[self.epoch_df["Wrist_Intensity"] >= 2].shape[0]
                                     * 100)
                wrist_valid.append(self.epoch_df.loc[(self.epoch_df["Wrist_Intensity"] >= 2) &
                                                     (self.epoch_df["ECG_Validity"] == "Valid")].shape[0] /
                                   self.epoch_df.loc[self.epoch_df["Wrist_Intensity"] >= 2].shape[0]
                                   * 100)

                # Non-sedentary epochs
                n_wrist_epochs.append(self.epoch_df.loc[self.epoch_df["Wrist_Intensity"] >= 1].shape[0])

                wrist_invalid.append(self.epoch_df.loc[(self.epoch_df["Wrist_Intensity"] >= 1) &
                                                       (self.epoch_df["ECG_Validity"] == "Invalid")].shape[0] /
                                     self.epoch_df.loc[self.epoch_df["Wrist_Intensity"] >= 1].shape[0]
                                     * 100)
                wrist_valid.append(self.epoch_df.loc[(self.epoch_df["Wrist_Intensity"] >= 1) &
                                                     (self.epoch_df["ECG_Validity"] == "Valid")].shape[0] /
                                   self.epoch_df.loc[self.epoch_df["Wrist_Intensity"] >= 1].shape[0]
                                   * 100)

                # Activity volumes (% of epochs)
                wrist_perc_epochs = [self.wrist.model.intensity_totals["Sedentary%"]*100,
                                     self.wrist.model.intensity_totals["Light%"]*100,
                                     self.wrist.model.intensity_totals["Moderate%"]*100,
                                     self.wrist.model.intensity_totals["Vigorous%"]*100,
                                     (self.wrist.model.intensity_totals["Moderate%"] +
                                     self.wrist.model.intensity_totals["Vigorous%"]) * 100,
                                     (1-self.wrist.model.intensity_totals["Sedentary%"])*100]

            if not self.load_wrist or not self.load_ecg:
                wrist_invalid = [None, None, None, None, None, None]
                wrist_valid = [None, None, None, None, None, None]
                n_wrist_epochs = [None, None, None, None, None, None]
                wrist_perc_epochs = [None, None, None, None, None, None]

            # Ankle data ---------------------------------------------------------------------------------------------
            if self.load_ankle and self.load_ecg:

                for intensity_cat in range(4):
                    ankle_invalid.append(self.epoch_df.loc[(self.epoch_df["Ankle_Intensity"] == intensity_cat) &
                                                           (self.epoch_df["ECG_Validity"] == "Invalid")].shape[0] /
                                         self.epoch_df.loc[self.epoch_df["Ankle_Intensity"] == intensity_cat].shape[0]
                                         * 100)
                    ankle_valid.append(self.epoch_df.loc[(self.epoch_df["Ankle_Intensity"] == intensity_cat) &
                                                         (self.epoch_df["ECG_Validity"] == "Valid")].shape[0] /
                                       self.epoch_df.loc[self.epoch_df["Ankle_Intensity"] == intensity_cat].shape[0]
                                       * 100)
                    n_ankle_epochs.append(self.epoch_df.loc[self.epoch_df["Ankle_Intensity"] == intensity_cat].shape[0])

                # MVPA epochs
                n_ankle_epochs.append(self.epoch_df.loc[self.epoch_df["Ankle_Intensity"] >= 2].shape[0])

                ankle_invalid.append(self.epoch_df.loc[(self.epoch_df["Ankle_Intensity"] >= 2) &
                                                       (self.epoch_df["ECG_Validity"] == "Invalid")].shape[0] /
                                     self.epoch_df.loc[self.epoch_df["Ankle_Intensity"] >= 2].shape[0]
                                     * 100)
                ankle_valid.append(self.epoch_df.loc[(self.epoch_df["Ankle_Intensity"] >= 2) &
                                                     (self.epoch_df["ECG_Validity"] == "Valid")].shape[0] /
                                   self.epoch_df.loc[self.epoch_df["Ankle_Intensity"] >= 2].shape[0]
                                   * 100)

                # Non-sedentary epochs
                n_ankle_epochs.append(self.epoch_df.loc[self.epoch_df["Ankle_Intensity"] >= 1].shape[0])

                ankle_invalid.append(self.epoch_df.loc[(self.epoch_df["Ankle_Intensity"] >= 1) &
                                                       (self.epoch_df["ECG_Validity"] == "Invalid")].shape[0] /
                                     self.epoch_df.loc[self.epoch_df["Ankle_Intensity"] >= 1].shape[0]
                                     * 100)
                ankle_valid.append(self.epoch_df.loc[(self.epoch_df["Ankle_Intensity"] >= 1) &
                                                     (self.epoch_df["ECG_Validity"] == "Valid")].shape[0] /
                                   self.epoch_df.loc[self.epoch_df["Ankle_Intensity"] >= 1].shape[0]
                                   * 100)

                ankle_perc_epochs = [self.ankle.model.intensity_totals["Sedentary%"] * 100,
                                     self.ankle.model.intensity_totals["Light%"] * 100,
                                     self.ankle.model.intensity_totals["Moderate%"] * 100,
                                     self.ankle.model.intensity_totals["Vigorous%"] * 100,
                                     (self.ankle.model.intensity_totals["Moderate%"] +
                                     self.ankle.model.intensity_totals["Vigorous%"]) * 100,
                                     (1-self.ankle.model.intensity_totals["Sedentary%"]) * 100]

            if not self.load_ankle or not self.load_ecg:
                ankle_invalid = [None, None, None, None, None, None]
                ankle_valid = [None, None, None, None, None, None]
                n_ankle_epochs = [None, None, None, None, None, None]
                ankle_perc_epochs = [None, None, None, None, None, None]

            # Creates df ---------------------------------------------------------------------------------------------
            validity_df = pd.DataFrame(list(zip(["Sedentary", "Light", "Moderate", "Vigorous", "MVPA", "All Activity"],
                                                n_wrist_epochs, wrist_perc_epochs, wrist_valid, wrist_invalid,
                                                n_ankle_epochs, ankle_perc_epochs, ankle_valid, ankle_invalid)),
                                       columns=["Intensity",
                                                "N_Wrist_Epochs", "Wrist_Activity_%", "Wrist_Valid", "Wrist_Invalid",
                                                "N_Ankle_Epochs", "Ankle_Activity_%", "Ankle_Valid", "Ankle_Invalid"])

            validity_df = validity_df.set_index("Intensity")

            return validity_df

        def using_counts(bin_size=100):

            wrist_invalid = []
            wrist_valid = []
            ankle_invalid = []
            ankle_valid = []
            n_wrist_epochs = []
            n_ankle_epochs = []
            upper_lim = 1000

            """if self.load_ankle and self.load_wrist:
                upper_lim = max([max(self.epoch_df["Wrist_SVM"]), max(self.epoch_df["Ankle_SVM"])])
            if self.load_ankle and not self.load_wrist:
                upper_lim = max(self.epoch_df["Ankle_SVM"])
            if not self.load_ankle and self.load_wrist:
                upper_lim = max(self.epoch_df["Wrist_SVM"])"""

            # Wrist data ---------------------------------------------------------------------------------------------
            if self.load_wrist and self.load_ecg:

                for count_val in np.arange(0, upper_lim, bin_size):
                    # df where counts are in bin
                    df = self.epoch_df.loc[(self.epoch_df["Wrist_SVM"] >= count_val) &
                                           (self.epoch_df["Wrist_SVM"] < count_val + bin_size)]

                    if df.shape[0] > 0:
                        # Calculates what % of data falls within parameters
                        wrist_valid.append(df.loc[df["ECG_Validity"] == 0].shape[0] / df.shape[0] * 100)

                        # Locates rows where SVM is within bin and ECG is valid
                        wrist_invalid.append(df.loc[df["ECG_Validity"] == 1].shape[0] / df.shape[0] * 100)

                        n_wrist_epochs.append(df.shape[0])

                    if df.shape[0] == 0:
                        wrist_invalid.append(0)
                        wrist_valid.append(0)
                        n_wrist_epochs.append(0)

            if not self.load_wrist or not self.load_ecg:
                wrist_invalid = [None for i in range(len(np.arange(0, upper_lim, bin_size)))]
                wrist_valid = [None for i in range(len(np.arange(0, upper_lim, bin_size)))]
                n_wrist_epochs = [None for i in range(len(np.arange(0, upper_lim, bin_size)))]

            # Ankle data ---------------------------------------------------------------------------------------------
            if self.load_ankle and self.load_ecg:

                for count_val in np.arange(0, upper_lim, bin_size):
                    # df where counts are in bin
                    df = self.epoch_df.loc[(self.epoch_df["Ankle_SVM"] >= count_val) &
                                           (self.epoch_df["Ankle_SVM"] < count_val + bin_size)]

                    if df.shape[0] > 0:
                        # Calculates what % of data falls within parameters
                        ankle_valid.append(df.loc[df["ECG_Validity"] == 0].shape[0] / df.shape[0] * 100)

                        # Locates rows where SVM is within bin and ECG is valid
                        ankle_invalid.append(df.loc[df["ECG_Validity"] == 1].shape[0] / df.shape[0] * 100)

                        n_ankle_epochs.append(df.shape[0])

                    if df.shape[0] == 0:
                        ankle_invalid.append(0)
                        ankle_valid.append(0)
                        n_ankle_epochs.append(0)

            if not self.load_ankle or not self.load_ecg:
                ankle_invalid = [None for i in range(len(np.arange(0, upper_lim, bin_size)))]
                ankle_valid = [None for i in range(len(np.arange(0, upper_lim, bin_size)))]
                n_ankle_epochs = [None for i in range(len(np.arange(0, upper_lim, bin_size)))]

            # Creates df ---------------------------------------------------------------------------------------------
            validity_df = pd.DataFrame(list(zip(np.arange(0, upper_lim, bin_size),
                                                n_wrist_epochs, wrist_valid, wrist_invalid,
                                                n_ankle_epochs, ankle_valid, ankle_invalid)),
                                       columns=["Counts",
                                                "N_Wrist_Epochs", "Wrist_Valid", "Wrist_Invalid",
                                                "N_Ankle_Epochs", "Ankle_Valid", "Ankle_Invalid"])
            validity_df = validity_df.set_index("Counts")

            return validity_df

        if data_type == "intensity":
            validity_df = using_intensity()
        if data_type == "counts":
            validity_df = using_counts(bin_size=bin_size)

        validity_df = validity_df.round(2)

        print("Complete.")

        return validity_df

    def plot_epoched(self, show_nonwear=True, show_sleep=True):

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 6))
        plt.subplots_adjust(bottom=.15)

        plt.suptitle("Participant {}: all available epoched data".format(self.subject_id))

        # Non-wear data -----------------------------------------------------------------------------------------------
        if show_nonwear:
            for removal in self.nonwear.nonwear_log.itertuples():
                # Wrist overlay
                try:
                    # No label if not final period
                    if removal.Index != self.nonwear.nonwear_log.index[-1]:
                        ax1.fill_between(x=(removal.DEVICEOFF, removal.DEVICEON), y1=0, y2=max(self.wrist.epoch.svm),
                                         color='grey', alpha=.75)

                    # Label if final period
                    if removal.Index == self.nonwear.nonwear_log.index[-1]:
                        ax1.fill_between(x=(removal.DEVICEOFF, removal.DEVICEON), y1=0, y2=max(self.wrist.epoch.svm),
                                         color='grey', alpha=.75, label="Non-wear")

                except TypeError:
                    pass

                # Ankle overlay
                try:
                    if removal.Index != self.nonwear.nonwear_log.index[-1]:
                        ax2.fill_between(x=(removal.DEVICEOFF, removal.DEVICEON), y1=0, y2=max(self.ankle.epoch.svm),
                                         color='grey', alpha=.75)
                    if removal.Index == self.nonwear.nonwear_log.index[-1]:
                        ax2.fill_between(x=(removal.DEVICEOFF, removal.DEVICEON), y1=0, y2=max(self.ankle.epoch.svm),
                                         color='grey', alpha=.75, label='Non-wear')
                except TypeError:
                    pass

        # Sleep data -------------------------------------------------------------------------------------------------
        if show_sleep:
            for i in range(self.sleep.data.shape[0] - 1):

                # Wrist overlay -----------------
                try:
                    # No label if not final sleep
                    if i != self.sleep.data.shape[0] - 2:
                        ax1.fill_between(x=(self.sleep.data.iloc[i]["TIME_SLEEP"],
                                            self.sleep.data.iloc[i + 1]["TIME_WAKE"]),
                                         y1=0, y2=max(self.wrist.epoch.svm),
                                         color='lightskyblue', alpha=.75)
                    # Label if final sleep
                    if i == self.sleep.data.shape[0] - 2:
                        # No label if not final period
                        ax1.fill_between(x=(self.sleep.data.iloc[i]["TIME_SLEEP"],
                                            self.sleep.data.iloc[i + 1]["TIME_WAKE"]),
                                         y1=0, y2=max(self.wrist.epoch.svm),
                                         color='lightskyblue', alpha=.75, label="Overnight sleep")
                except TypeError:
                    pass

                # Ankle overlay -----------------
                try:
                    # No label if not final sleep
                    if i != self.sleep.data.shape[0] - 2:
                        ax2.fill_between(x=(self.sleep.data.iloc[i]["TIME_SLEEP"],
                                            self.sleep.data.iloc[i + 1]["TIME_WAKE"]),
                                         y1=0, y2=max(self.ankle.epoch.svm),
                                         color='lightskyblue', alpha=.75)
                    # Label if final sleep
                    if i == self.sleep.data.shape[0] - 2:
                        # No label if not final period
                        ax2.fill_between(x=(self.sleep.data.iloc[i]["TIME_SLEEP"],
                                            self.sleep.data.iloc[i + 1]["TIME_WAKE"]),
                                         y1=0, y2=max(self.ankle.epoch.svm),
                                         color='lightskyblue', alpha=.75, label="Overnight sleep")
                except TypeError:
                    pass

                # HR overlay --------------------
                try:
                    # No label if not final sleep
                    if i != self.sleep.data.shape[0] - 2:
                        ax3.fill_between(x=(self.sleep.data.iloc[i]["TIME_SLEEP"],
                                            self.sleep.data.iloc[i + 1]["TIME_WAKE"]),
                                         y1=min([i for i in self.ecg.valid_hr if i is not None]) * .9,
                                         y2=max([i for i in self.ecg.valid_hr if i is not None]) * 1.1,
                                         color='lightskyblue', alpha=.75)
                    # Label if final sleep
                    if i == self.sleep.data.shape[0] - 2:
                        # No label if not final period
                        ax3.fill_between(x=(self.sleep.data.iloc[i]["TIME_SLEEP"],
                                            self.sleep.data.iloc[i + 1]["TIME_WAKE"]),
                                         y1=min([i for i in self.ecg.valid_hr if i is not None]) * .9,
                                         y2=max([i for i in self.ecg.valid_hr if i is not None]) * 1.1,
                                         color='lightskyblue', alpha=.75)
                except TypeError:
                    pass

        # Wrist data -------------------------------------------------------------------------------------------------
        try:
            ax1.plot(self.wrist.epoch.timestamps, self.wrist.epoch.svm, color='black', label="Wrist")
        except AttributeError:
            ax1.axhline(y=0, linestyle='dashed', color='black', label="No wrist data")
        ax1.set_ylabel("Counts")
        ax1.legend()

        # Ankle data -------------------------------------------------------------------------------------------------
        try:
            ax2.plot(self.ankle.epoch.timestamps, self.ankle.epoch.svm, color='black', label="Ankle")
        except AttributeError:
            ax2.axhline(y=0, linestyle='dashed', color='black', label="No ankle data")
        ax2.set_ylabel("Counts")
        ax2.legend()

        # HR data ----------------------------------------------------------------------------------------------------
        try:
            ax3.plot(self.ecg.epoch_timestamps, self.ecg.valid_hr, color='red', label="HR")
        except AttributeError:
            ax3.axhline(y=0, linestyle='dashed', color='red', label='Missing HR data')
        ax3.set_ylabel("bpm")
        ax3.legend()

        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%a., %b. %d \n%H:%M:%S"))
        plt.xticks(rotation=45, fontsize=9)
