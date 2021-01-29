from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from random import randint
import random
import ImportEDF
from matplotlib.widgets import CheckButtons
import ECG
import os
from csv import DictWriter
import pandas as pd


# ======================================================= SET UP ======================================================
# Your initials
initials = "KW"

# WHETHER OR NOT TO SHOW WHAT ALGORITHM DECIDED
show_algorithm_verdict = False

# Full pathway to folder with EDF files. Make sure it ends with a slash.
# edf_folder = "/Users/kyleweber/Desktop/Data/OND07/EDF/"
edf_folder = "/Volumes/nimbal$/Data/ReMiNDD/Raw data/Bittium/"

# csv file to write to (full path)
data_file = "/Users/kyleweber/Desktop/ECG_Datafile.csv"

# Seconds
epoch_length = 15

# ===================================================== PROCESSING ====================================================
file_list = os.listdir(edf_folder)
file_list = [i for i in file_list if "BF" in i]

rand_sub = random.randint(0, len(file_list) - 1)

start_time, end_time, fs, duration = \
    ImportEDF.check_file(edf_folder + file_list[rand_sub],
                         print_summary=False)
rand_start = randint(0, duration * fs - 45 * fs)

print("\nImporting file {}".format(file_list[rand_sub]))

ecg_object = ECG.ECG(filepath=edf_folder+file_list[rand_sub], age=0,
                     start_offset=rand_start, end_offset=3 * epoch_length * fs,
                     epoch_len=epoch_length, load_raw=True, load_accel=True, from_processed=False)

qc_data = ECG.CheckQuality(ecg_object=ecg_object, start_index=epoch_length*fs,
                           template_data='wavelet', voltage_thresh=250, epoch_len=epoch_length)

parameters_dict = {"Initials": initials,
                   "ID": file_list[rand_sub].split(".")[0], "StartInd": rand_start,
                   "VisualInspection": None,
                   "OrphanidouAlgorithm": "Valid" if qc_data.rule_check_dict["Valid Period"] else "Invalid",

                   "HR": qc_data.rule_check_dict["HR"],
                   "HR_Valid": "Valid" if qc_data.rule_check_dict["HR Valid"] else "Invalid",

                   "MaxRRInt": qc_data.rule_check_dict["Max RR Interval"],
                   "MaxRRInt_Valid": "Valid" if qc_data.rule_check_dict["Max RR Interval Valid"] else "Invalid",

                   "RRRatio": qc_data.rule_check_dict["RR Ratio"],
                   "RRRatio_Valid": "Valid" if qc_data.rule_check_dict["RR Ratio Valid"] else "Invalid",

                   "Correlation": qc_data.rule_check_dict["Correlation"],
                   "CorrelationValid": "Valid" if qc_data.rule_check_dict["Correlation Valid"] else "Invalid",

                   "VoltRange": ecg_object.avg_voltage[1], "Avg_Accel": round(np.mean(ecg_object.accel_vm), 1),
                   }


def append_parameters_dict(data_file):

    with open(data_file, "a", newline="\n") as write_obj:
        dict_writer = DictWriter(write_obj, fieldnames=parameters_dict.keys())

        dict_writer.writerow(parameters_dict)
        print("\nNew data appended to {}".format(data_file))

    df = pd.read_csv(data_file, usecols=["ID", "VisualInspection"])

    n_total = df.shape[0]

    n_valid = df.loc[df["VisualInspection"] == "ValidWear"].shape[0]

    n_invalid = df.loc[df["VisualInspection"] == "InvalidWear"].shape[0] + \
                df.loc[df["VisualInspection"] == "Nonwear"].shape[0]

    n_unsure = df.loc[df["VisualInspection"] == 'Unsure'].shape[0]

    print("-File contains {} records ({} valid; {} invalid; {} unsure)".format(n_total, n_valid, n_invalid, n_unsure))


def get_value(label):
    if label == "Nonwear":
        parameters_dict["VisualInspection"] = "Nonwear"
        print("Period set as non-wear.")
    if label == "ValidWear":
        parameters_dict["VisualInspection"] = "ValidWear"
        print("Period set as valid wear.")
    if label == "InvalidWear":
        parameters_dict["VisualInspection"] = "InvalidWear"
        print("Period set as invalid wear.")
    if label == "Unsure":
        parameters_dict["VisualInspection"] = "Unsure"
        print("Period marked as unsure.")

    plt.draw()
    plt.close("all")

    append_parameters_dict(data_file=data_file)


# ===================================================== PLOTTING ======================================================
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 6), sharex='col')

if show_algorithm_verdict:
    plt.suptitle("{}: {}, {}".format(file_list[rand_sub].split(".")[0],
                                     qc_data.rule_check_dict["Valid Period"],
                                     datetime.strftime(datetime.strptime(str(ecg_object.timestamps[0])[:-3],
                                                                         "%Y-%m-%dT%H:%M:%S.%f"), "%I:%M:%S %p")))
if not show_algorithm_verdict:
    plt.suptitle("{}: {}".format(file_list[rand_sub].split("_")[2],
                                 datetime.strftime(datetime.strptime(str(ecg_object.timestamps[0])[:-3],
                                                                     "%Y-%m-%dT%H:%M:%S.%f"), "%I:%M:%S %p")))
plt.subplots_adjust(right=.82)

if show_algorithm_verdict:
    if qc_data.rule_check_dict["Valid Period"]:
        c = 'green'
    if not qc_data.rule_check_dict["Valid Period"]:
        c = 'red'
if not show_algorithm_verdict:
    c = 'black'

ax1.plot(np.arange(0, len(ecg_object.raw)) / fs, ecg_object.raw, color=c, linestyle='-', label='raw')
ax2.plot(np.arange(0, len(ecg_object.filtered)) / fs, ecg_object.filtered, color=c, label=qc_data.template_data)

ax1.legend()
ax2.legend()

ax1.set_ylabel("Voltage")

y_scale = ax1.get_ylim()
if y_scale[1] - y_scale[0] <= 1000:
    ax1.set_ylim(np.mean(ecg_object.raw) - 600, np.mean(ecg_object.raw) + 600)

y_scale = ax2.get_ylim()
if y_scale[1] - y_scale[0] <= 1000:
    ax2.set_ylim(np.mean(ecg_object.filtered) - 600, np.mean(ecg_object.filtered) + 600)

ax1.fill_between(x=[15, 30], y1=ax1.get_ylim()[0], y2=ax1.get_ylim()[1], color='grey', alpha=.25)
ax2.fill_between(x=[15, 30], y1=ax2.get_ylim()[0], y2=ax2.get_ylim()[1], color='grey', alpha=.25)

ax3.plot(np.arange(0, len(ecg_object.accel_vm)) / ecg_object.accel_sample_rate, ecg_object.accel_x,
         color='dodgerblue', label='x')
ax3.plot(np.arange(0, len(ecg_object.accel_vm)) / ecg_object.accel_sample_rate, ecg_object.accel_y,
         color='red', label='y')
ax3.plot(np.arange(0, len(ecg_object.accel_vm)) / ecg_object.accel_sample_rate, ecg_object.accel_z,
         color='black', label='z')
ax3.legend()
ax3.set_ylabel("mG")
ax3.set_ylim(-2000, 2000)
ax3.set_xlabel("Seconds")
ax3.fill_between(x=[15, 30], y1=-2000, y2=2000, color='grey', alpha=.25)

rax = plt.axes([.83, .4, .15, .2])
check = CheckButtons(rax, ("Nonwear", "ValidWear", "InvalidWear", "Unsure"))
check.on_clicked(get_value)

plt.show()
