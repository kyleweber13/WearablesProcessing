import ECG
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


class EcgComp:

    def __init__(self, subj_id=None, age=0, load_accel=False,
                 stingray_file=None, fastfix_file=None, electrode_file=None):

        self.subj_id = subj_id
        self.age = age
        self.load_accel = load_accel
        self.stingray_file = stingray_file
        self.fastfix_file = fastfix_file
        self.electrode_file = electrode_file

        self.stingray = None
        self.fastfix = None
        self.electrode = None

    def load_data(self):

        if self.stingray_file is not None:
            self.stingray = ECG.ECG(subject_id=self.subj_id, filepath=self.stingray_file,
                                    age=self.age, start_offset=0, end_offset=0,
                                    rest_hr_window=60, n_epochs_rest=30,
                                    epoch_len=15, load_accel=self.load_accel,
                                    filter_data=False, low_f=1, high_f=30, f_type="bandpass",
                                    load_raw=True, from_processed=False)

            self.stingray.epoch_timestamps = [datetime.strptime(str(i)[:-3], "%Y-%m-%dT%H:%M:%S.%f") for
                                              i in self.stingray.epoch_timestamps]
            self.stingray.name = "Stingray"

        if self.fastfix_file is not None:
            self.fastfix = ECG.ECG(subject_id=self.subj_id, filepath=self.fastfix_file,
                                   age=self.age, start_offset=0, end_offset=0,
                                   rest_hr_window=60, n_epochs_rest=30,
                                   epoch_len=15, load_accel=self.load_accel,
                                   filter_data=False, low_f=1, high_f=30, f_type="bandpass",
                                   load_raw=True, from_processed=False)

            self.fastfix.epoch_timestamps = [datetime.strptime(str(i)[:-3], "%Y-%m-%dT%H:%M:%S.%f") for
                                             i in self.fastfix.epoch_timestamps]
            self.fastfix.name = "Fastfix"

        if self.electrode_file is not None:
            self.electrode = ECG.ECG(subject_id=self.subj_id, filepath=self.electrode_file,
                                     age=self.age, start_offset=0, end_offset=0,
                                     rest_hr_window=60, n_epochs_rest=30,
                                     epoch_len=15, load_accel=self.load_accel,
                                     filter_data=False, low_f=1, high_f=30, f_type="bandpass",
                                     load_raw=True, from_processed=False)

            self.electrode.epoch_timestamps = [datetime.strptime(str(i)[:-3], "%Y-%m-%dT%H:%M:%S.%f") for
                                               i in self.electrode.epoch_timestamps]
            self.electrode.name = "Electrode"

    def compare_raw(self, data1, data2):

        print("\nPlotting {} and {} data...".format(data1.name, data2.name))
        fig, (ax1, ax2) = plt.subplots(2, sharex='col')

        ax1.plot(data1.timestamps[::5], data1.raw[::5], color='red', label="{}_Raw".format(data1.name))
        ax1.plot(data1.timestamps[::5], data1.filtered[::5], color='black', label="{}_Filt".format(data1.name))
        ax1.legend()

        ax2.plot(data2.timestamps[::5], data2.raw[::5], color='red', label="{}_Raw".format(data2.name))
        ax2.plot(data2.timestamps[::5], data2.filtered[::5], color='black', label="{}_Filt".format(data2.name))
        ax2.legend()

    def compare_hr(self, data1, data2):

        print("\nPlotting {} and {} heart rate data...".format(data1.name, data2.name))

        fig, (ax1, ax2) = plt.subplots(2, sharex='col')
        ax1.plot(data1.epoch_timestamps, data1.valid_hr, color='red', label=data1.name)
        ax1.set_ylabel("HR (bpm)")
        ax1.legend()

        ax2.plot(data2.epoch_timestamps, data2.valid_hr, color='black', label=data2.name)
        ax2.set_ylabel("HR (bpm)")
        ax2.legend()

    def compare_valid_epochs(self, data1, data2):

        print("\nComparing valid epochs for {} and {} heart rate data...".format(data1.name, data2.name))

        d1 = [data1.valid_hr[i] for i in range(min(len(data1.valid_hr), len(data2.valid_hr))) if
              data1.valid_hr[i] is not None and data2.valid_hr[i] is not None]
        d2 = [data2.valid_hr[i] for i in range(min(len(data1.valid_hr), len(data2.valid_hr))) if
              data1.valid_hr[i] is not None and data2.valid_hr[i] is not None]

        print("Pearson correlation (data1, data2) = {}".format(round(scipy.stats.pearsonr(d1, d2)[0], 3)))

        fig, (ax1, ax2) = plt.subplots(2, sharex='col')
        ax1.plot(d1, color='black', label=data1.name)
        ax1.plot(d2, color='dodgerblue', label=data2.name)
        ax1.legend()
        ax1.set_ylabel("HR")

        d = [d1[i] - d2[i] for i in range(len(d1))]

        ax2.plot(d, color='red', label="Difference (data1 - data2)")
        ax2.axhline(sum(d)/len(d), linestyle='dashed', color='black',
                    label="Mean diff ({} bpm)".format(round(sum(d)/len(d), 1)))
        ax2.set_ylabel("Delta HR (bpm)")
        ax2.legend()

    def compare_valid_epochs_scatter(self, data1, data2):

        print("\nComparing valid epochs for {} and {} heart rate data...".format(data1.name, data2.name))

        d1 = [data1.valid_hr[i] for i in range(min(len(data1.valid_hr), len(data2.valid_hr))) if
              data1.valid_hr[i] is not None and data2.valid_hr[i] is not None]
        d2 = [data2.valid_hr[i] for i in range(min(len(data1.valid_hr), len(data2.valid_hr))) if
              data1.valid_hr[i] is not None and data2.valid_hr[i] is not None]

        plt.scatter(d1, d2, color='dodgerblue')
        plt.plot(np.arange(min(min(d1), min(d2))-5, max(max(d1), max(d2)), 5)+5,
                 np.arange(min(min(d1), min(d2))-5, max(max(d1), max(d2)), 5)+5,
                 linestyle='dashed', color='black', label='Y = X')
        plt.legend()
        plt.xlabel("{} HR".format(data1.name))
        plt.ylabel("{} HR".format(data2.name))

    def bland_altman(self, data1, data2):

        print("\nComparing valid epochs for {} and {} heart rate data "
              "using Bland-Altman...".format(data1.name, data2.name))

        d1 = [data1.valid_hr[i] for i in range(min(len(data1.valid_hr), len(data2.valid_hr))) if
              data1.valid_hr[i] is not None and data2.valid_hr[i] is not None]
        d2 = [data2.valid_hr[i] for i in range(min(len(data1.valid_hr), len(data2.valid_hr))) if
              data1.valid_hr[i] is not None and data2.valid_hr[i] is not None]

        m = [(d1[i] + d2[i])/2 for i in range(len(d1))]
        d = [d1[i] - d2[i] for i in range(len(d1))]
        sd = np.std(d)

        plt.title("Bland-Altman Plot")
        plt.scatter(m, d, color='dodgerblue')

        plt.axhline(1.96*sd + sum(d)/len(d), color='red', linestyle='dashed',
                    label="Upper LOA ({}bpm)".format(round(1.96*sd + sum(d)/len(d), 1)))
        plt.axhline(sum(d) / len(d), label="Bias ({}bpm)".format(round(sum(d) / len(d), 1)),
                    color='black', linestyle='dashed')
        plt.axhline(sum(d)/len(d) - 1.96*sd, color='red', linestyle='dashed',
                    label="Lower LOA ({}bpm)".format(round(sum(d)/len(d) - 1.96*sd, 1)))

        plt.ylabel("Difference ({} - {})".format(data1.name, data2.name))
        plt.xlabel("Mean ({}, {})".format(data1.name, data2.name))
        plt.legend()


ecg = EcgComp(subj_id="007", age=26, load_accel=False,
              electrode_file="/Users/kyleweber/Desktop/007_Test/007_Electrodes.edf",
              stingray_file="/Users/kyleweber/Desktop/007_Test/007_Stingray.edf")

# ecg.load_data()
# ecg.compare_raw(ecg.stingray, ecg.electrode)
# ecg.compare_hr(ecg, ecg.stingray, ecg.electrode)
# ecg.compare_valid_epochs(ecg.stingray, ecg.electrode)
# ecg.compare_valid_epochs_scatter(ecg.stingray, ecg.electrode)
# ecg.bland_altman(ecg.electrode, ecg.stingray)