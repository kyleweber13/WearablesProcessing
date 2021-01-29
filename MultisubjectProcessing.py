import LocateParticipants
from Subject import Subject
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
import statsmodels.stats.power as smp
from matplotlib import pyplot as plt
import numpy as np
import scipy
import os
import seaborn as sns
import pingouin as pg


usable_subjs = LocateParticipants.SubjectSubset(check_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/"
                                                                 "OND07_ProcessingStatus.xlsx",
                                                wrist_ankle=False, wrist_hr=True,
                                                wrist_hracc=False, hr_hracc=False,
                                                ankle_hr=False, ankle_hracc=False,
                                                wrist_only=False, ankle_only=False,
                                                hr_only=False, hracc_only=False,
                                                require_treadmill=False, require_all=True)

# 3002 = repeat, 3012 has no age, 3018 is missing Wrist file, 3019 missing weight
for subj in ["OND07_WTL_3002", "OND07_WTL_3012", "OND07_WTL_3018", "OND07_WTL_3019"]:
    try:
        usable_subjs.participant_list.remove(subj)
        print("Removed {} from usable subjects list.".format(subj))
    except ValueError:
        pass

for i in usable_subjs.participant_list:
    if not ("_" in i):
        usable_subjs.participant_list.remove(i)


def loop_subjects_standalone(subj_list):

    for subj in subj_list:
        try:
            x = Subject(
                # What data to load in
                subject_id=int(subj.split("_")[-1]),
                load_ecg=True, load_ankle=True, load_wrist=True,
                load_raw_ecg=False, load_raw_ankle=False, load_raw_wrist=False,
                from_processed=True,

                # Model parameters
                rest_hr_window=60,
                n_epochs_rest_hr=30,
                hracc_threshold=30,
                epoch_len=15,

                # Data files
                raw_edf_folder="/Users/kyleweber/Desktop/Data/OND07/EDF/",
                treadmill_log_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Treadmill_Log.csv",
                demographics_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/Demographics_Data.csv",
                sleeplog_file="/Users/kyleweber/Desktop/Data/OND07/Tabular Data/SleepLogs_All.csv",
                output_dir="/Users/kyleweber/Desktop/Data/OND07/Processed Data/",
                processed_folder="/Users/kyleweber/Desktop/Data/OND07/Processed Data/Model Output/",
                write_results=False)

            return x

        except:
            pass
