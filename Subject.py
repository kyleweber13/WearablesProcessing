"""import ImportDemographics
import Accelerometer
import ECG
import DeviceSync
import SleepData
import ModelStats
import ValidData
import ImportCropIndexes
import ImportEDF
import HRAcc
import DailyReports
import Nonwear"""

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

    def __init__(self, from_processed, raw_edf_folder=None, subjectID=None,
                 load_wrist=False, load_ankle=False, load_ecg=False,
                 load_raw_ecg=False, load_bittium_accel=False, load_raw_ankle=False, load_raw_wrist=False,
                 epoch_len=15, remove_epoch_baseline=False,
                 rest_hr_window=60, n_epochs_rest_hr=30, hracc_threshold=30,
                 crop_index_file=None, filter_ecg=False,
                 output_dir=desktop_path, processed_folder=None,
                 write_results=False, treadmill_log_file=None, nonwear_log_file=None,
                 demographics_file=None, sleeplog_file=None):

        print()
        print("========================================= SUBJECT #{} "
              "=============================================".format(subjectID))
        print()

        # Model objects
        self.wrist, self.ankle, self.ecg, self.hr_acc = None, None, None, None

        self.subjectID = subjectID  # 4-digit ID code
        self.raw_edf_folder = raw_edf_folder  # Folder where raw EDF files are stored

        # Processing variables
        self.epoch_len = epoch_len  # Epoch length; seconds
        self.remove_epoch_baseline = remove_epoch_baseline  # Removes lowest value from epoched data to remove bias
        self.rest_hr_window = rest_hr_window  # Number of epochs over which rolling average HR is calculated
        self.n_epochs_rest_hr = n_epochs_rest_hr  # Number of epochs over which resting HR is calculate
        self.hracc_threshold = hracc_threshold  # Threshold for HR-Acc model

        self.from_processed = from_processed  # Whether to import already-processed data
        self.write_results = write_results  # Whether to write results to CSV

        if self.from_processed:
            self.write_results = False

        self.output_dir = output_dir  # Where files are written
        self.processed_folder = processed_folder  # Folder that contains already-processed data files

        self.demographics_file = demographics_file
        self.treadmill_log_file = treadmill_log_file
        self.sleeplog_file = sleeplog_file
        self.nonwear_file = nonwear_log_file

        # Data cropping data
        self.starttime_dict = {"Ankle": None, "Wrist": None, "ECG": None}
        self.start_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}
        self.end_offset_dict = {"Ankle": 0, "Wrist": 0, "ECG": 0}
        self.crop_index_file = crop_index_file  # CSV file of crop indexes
        self.crop_indexes_found = False

        # Which device data to load
        self.load_wrist = load_wrist
        self.load_raw_wrist = load_raw_wrist
        self.wrist_filepath = None

        self.load_ankle = load_ankle
        self.load_raw_ankle = load_raw_ankle
        self.ankle_filepath = None

        self.load_ecg = load_ecg
        self.ecg_filepath = None
        self.load_raw_ecg = load_raw_ecg
        self.load_bittium_accel = load_bittium_accel
        self.filter_ecg = filter_ecg

        if not self.load_ecg and (self.load_ankle or self.load_wrist):
            self.accel_only = True
        if self.load_ecg:
            self.accel_only = False

        self.valid_all = None
        self.valid_accelonly = None

        self.demographics = {"Age": 18, "Sex": None, "Weight": 1, "Height": 1, "Hand": "Right", "RestVO2": 3.5}

        self.stats = None
        self.daily_summary = None

    def get_raw_filepaths(self):
        """Retrieves filenames associated with current subject."""

        subject_file_list = [i for i in os.listdir(self.raw_edf_folder) if
                             (".EDF" in i or ".edf" in i)
                             and str(self.subjectID) in i]

        # Returns Nones if no files found
        if len(subject_file_list) == 0:
            print("No files found for this subject ID.")
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
                print("Could not find the correct wrist accelerometer file.")
                wrist_filename = None
                self.load_wrist = False

            # Selects correct wrist temperature file
            if len(wrist_temperature_filenames) == 2:
                wrist_temperature_filename = [i for i in wrist_temperature_filenames if dom_hand + "Wrist" not in i][0]
            if len(wrist_temperature_filenames) == 1:
                wrist_temperature_filename = wrist_temperature_filenames[0]
            if len(wrist_temperature_filenames) == 0:
                print("Could not find the correct wrist temperature file.")
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
                print("Could not find the correct ankle accelerometer file.")
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