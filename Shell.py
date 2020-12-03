from Subject import Subject
import Nonwear
import SleepData
import os
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import scipy.stats
import pandas as pd
import numpy as np
import warnings
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
warnings.filterwarnings("ignore")

x = Subject(
    subject_id=3028,
    study_code="OND07",

    load_ecg=True, load_ankle=False, load_wrist=False,
    load_raw_ecg=False, load_bittium_accel=False, load_raw_ankle=False, load_raw_wrist=False,
    from_processed=True,

    crop_file_start=False,  # leave as True unless devices start at wildly different times
    crop_file_end=False,  # leave as False unless only want time period with all device data

    # Model parameters
    rest_hr_window=60,  # number of seconds over which HR is averaged
    n_epochs_rest_hr=30,  # number of epochs over which average HRs are averaged

    epoch_len=15,

    # Data files
    # raw_edf_folder="/Users/kyleweber/Desktop/Data/STEPS/",
    raw_edf_folder="/Users/kyleweber/Desktop/Data/OND07/EDF/",

    treadmill_log_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Treadmill_Log.csv",

    # demographics_file="/Users/kyleweber/Desktop/Data/STEPS/Demographics_Data.csv",
    demographics_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Demographics_Data.csv",

    # sleeplog_file="/Users/kyleweber/Desktop/Data/STEPS/Sleep_log_data.csv",
    # sleeplog_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/SleepLogs_All.csv",

    nonwear_log_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/NonwearLog.xlsx",

    # Where to write data
    # output_dir="/Users/kyleweber/Desktop/Data/STEPS/",
    output_dir="/Users/kyleweber/Desktop/",

    # Where to read already-processed data
    # processed_folder="/Users/kyleweber/Desktop/Data/STEPS/Model Output/",
    processed_folder="/Users/kyleweber/Desktop/Data/OND07/Processed Data/Model Output/",

    write_results=False)

# ================================================== RUNNING METHODS ==================================================
x.import_demographics()
x.create_filenames()
x.get_edf_filepaths()
x.import_epoch_df()
x.crop_files()
x.create_device_objects()
x.get_data_len()
x.sleep = SleepData.Sleep(subject_object=x)
x.nonwear = Nonwear.NonwearLog(subject_object=x)

print("\n======================================= ADDITIONAL ANALYSES =========================================")

# Activity counts and ECG validity contained in this df
# x.epoch_df = x.create_epoch_df()
# x.epoch_df.to_csv("{}/EpochDF_{}.csv".format(x.output_dir, x.subject_id), index=False)

# Data that describes ECG validity in the context of movement
# x.ecg_contingency_table = x.create_ecg_contingency_table(data_type="counts", bin_size=100)
# x.ecg_contingency_table.to_excel("{}/{}_ECG_Validity_Table.xlsx".format(x.output_dir, x.subject_id))

# TODO
# Fix zero divison error in validity_df creation
# Redo from_processed reading in: single epoched DF

# ECG.load_processed(): change read-in method to pandas
# Move stuff from Subject.create_epoch_df() into new function which runs after device objects created
    # Will need to change a bunch of "if __ is None" later downstream
# Get data from subject.ecg_contingency_table from all participants

# Investigate hrv through ecgdetectors package
# Need to do data cropping when from_processed
# Once cropping taken care of, remove all the accel_only stuff

# Re-make epoch_df file with all devices