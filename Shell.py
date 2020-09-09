from Subject import Subject
"""import LocateUsableParticipants
import Nonwear"""

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
    # What data to load in
    subject_id=3043,
    load_ecg=False, load_ankle=True, load_wrist=True,
    load_raw_ecg=False, load_bittium_accel=False, load_raw_ankle=False, load_raw_wrist=False,
    from_processed=True,

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
    sleeplog_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/SleepLogs_All.csv",

    nonwear_log_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/NonwearLog.xlsx",

    # output_dir="/Users/kyleweber/Desktop/Data/STEPS/",
    output_dir="/Users/kyleweber/Desktop/Data/OND07/Processed Data/",

    # processed_folder="/Users/kyleweber/Desktop/Data/STEPS/Model Output/",
    processed_folder="/Users/kyleweber/Desktop/Data/OND07/Processed Data/Model Output/",

    write_results=False)

# ================================================== RUNNING METHODS ==================================================
