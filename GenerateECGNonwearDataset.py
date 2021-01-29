from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime
from random import randint
import random
import ImportEDF
from matplotlib.widgets import CheckButtons
import Bittium_Freq_Analysis
import os

# ======================================================= SET UP ======================================================
# WHETHER OR NOT TO SHOW WHAT ALGORITHM DECIDED
show_algorithm_verdict = True

# Full pathway to folder with EDF files
edf_folder = "/Users/kyleweber/Desktop/Data/OND07/EDF/"

# csv file to write to (full path)
data_file = "/Users/kyleweber/Desktop/ECG_Datafile.csv"

# ===================================================== PROCESSING ====================================================
file_list = os.listdir(edf_folder)
file_list = [i for i in file_list if "BF" in i]

rand_sub = random.randint(0, len(file_list) - 1)

start_time, end_time, fs, duration = \
    ImportEDF.check_file(edf_folder + file_list[rand_sub],
                         print_summary=False)
rand_start = randint(0, duration * fs - 45 * fs)

x = Bittium_Freq_Analysis.Data(subj_id=rand_sub, start_index=rand_start, end_index=45 * fs, seg_length=15,
                               filepath=edf_folder + file_list[rand_sub])
x.import_ecg()

x.ecg_fft = x.run_ecg_fft(start=15*250, show_plot=False)
ecg_cutoff = x.plot_ecg_cumulative_fft(threshold=.9, show_plot=False)

x.accel_fft = x.run_accel_fft(start=15*25, show_plot=False)
x.plot_accel_cumulative_fft(threshold=.25, axis="X", show_plot=False)
x.plot_accel_cumulative_fft(threshold=.25, axis="Y", show_plot=False)
x.plot_accel_cumulative_fft(threshold=.25, axis="Z", show_plot=False)

x.complete_parameter_dict()


def get_value(label):
    if label == "Nonwear":
        x.parameters_dict["Visual_nonwear"] = "Nonwear"
        print("Period set as non-wear.")
    if label == "Wear":
        x.parameters_dict["Visual_nonwear"] = "Wear"
        print("Period set as wear.")
    if label == "Unsure":
        x.parameters_dict["Visual_nonwear"] = "Unsure"
        print("Period marked as unsure if non-wear.")

    plt.draw()
    plt.close("all")

    x.append_parameters_dict(data_file)


fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 6), sharex='col')

if show_algorithm_verdict:
    plt.suptitle("{}: {}, {}".format(x.subj_id, x.parameters_dict["Valid_ECG"],
                                     datetime.strftime(datetime.strptime(str(x.ecg.timestamps[0])[:-3],
                                                                         "%Y-%m-%dT%H:%M:%S.%f"), "%I:%M:%S %p")))
if not show_algorithm_verdict:
    plt.suptitle("{}: {}".format(x.subj_id, datetime.strftime(datetime.strptime(str(x.ecg.timestamps[0])[:-3],
                                                                                "%Y-%m-%dT%H:%M:%S.%f"),
                                                              "%I:%M:%S %p")))
plt.subplots_adjust(right=.82)

if show_algorithm_verdict:
    if x.ecg.epoch_validity[0] == 'Valid':
        c = 'green'
    if x.ecg.epoch_validity[0] == 'Invalid':
        c = 'red'
if not show_algorithm_verdict:
    c = 'black'

ax1.plot(np.arange(0, len(x.ecg.raw)) / x.ecg.sample_rate, x.ecg.raw, color=c, linestyle='-', label='raw')
ax2.plot(np.arange(0, len(x.ecg.filtered)) / x.ecg.sample_rate, x.ecg.filtered, color=c, label='filt')

ax1.legend()
ax2.legend()

ax1.set_ylabel("Voltage")

y_scale = ax1.get_ylim()
if y_scale[1] - y_scale[0] <= 1000:
    ax1.set_ylim(np.mean(x.ecg.raw) - 600, np.mean(x.ecg.raw) + 600)

y_scale = ax2.get_ylim()
if y_scale[1] - y_scale[0] <= 1000:
    ax2.set_ylim(np.mean(x.ecg.filtered) - 600, np.mean(x.ecg.filtered) + 600)

ax1.fill_between(x=[15, 30], y1=ax1.get_ylim()[0], y2=ax1.get_ylim()[1], color='grey', alpha=.25)
ax2.fill_between(x=[15, 30], y1=ax2.get_ylim()[0], y2=ax2.get_ylim()[1], color='grey', alpha=.25)

ax3.plot(np.arange(0, len(x.ecg.accel_vm)) / x.ecg.accel_sample_rate, x.ecg.accel_x, color='dodgerblue', label='x')
ax3.plot(np.arange(0, len(x.ecg.accel_vm)) / x.ecg.accel_sample_rate, x.ecg.accel_y, color='red', label='y')
ax3.plot(np.arange(0, len(x.ecg.accel_vm)) / x.ecg.accel_sample_rate, x.ecg.accel_z, color='black', label='z')
ax3.legend()
ax3.set_ylabel("mG")
ax3.set_ylim(-2000, 2000)
ax3.set_xlabel("Seconds")
ax3.fill_between(x=[15, 30], y1=-2000, y2=2000, color='grey', alpha=.25)

rax = plt.axes([.83, .4, .15, .2])
# check = CheckButtons(rax, ("Nonwear", "Wear", "Unsure"), (False, False, False))
check = CheckButtons(rax, ("InvalidNonwear", "ValidWear", "InvalidWear", "Unsure"))
check.on_clicked(get_value)

plt.show()
