from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statistics
import scipy.stats as stats
from datetime import datetime
import progressbar
from matplotlib.ticker import PercentFormatter
from random import randint
import matplotlib.dates as mdates
import scipy.fft
from scipy.signal import butter, filtfilt
import random
from ECG import ECG
import ImportEDF
from matplotlib.widgets import CheckButtons
import scipy.stats as stats


class Data:

    def __init__(self, subj_id, start_index=None, end_index=None, seg_length=15):
        self.subj_id = subj_id
        self.start_index = start_index
        self.end_index = end_index
        self.seg_length = seg_length

        self.filepath = "/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_BF.edf".format(subj_id)
        self.ecg = None
        self.nonwear = None
        self.ecg_cutoffs = None

        self.ecg_fft = None
        self.accel_fft = None

        self.start_ind = None

        self.gs_log = None

        self.parameters_dict = {"ID": self.subj_id,
                                "ECG_Index": 0, "Valid_ECG": 0, "ECG_volt_range": 0, "ECG_dom_f": 0, "ECG_power": 0,
                                "SD_X": 0, "SD_Y": 0, "SD_Z": 0, "SVM": 0, "Accel_dom_f": 0,
                                "Accel_power_X": 0, "Accel_power_Y": 0, "Accel_power_Z": 0,
                                "Visual_nonwear": 0, "DataLength": self.seg_length, "VisualLength": 0}

    def import_ecg(self):

        self.ecg = ECG(subject_id=self.subj_id, filepath=self.filepath,
                       output_dir=None, processed_folder=None,
                       processed_file=None,
                       age=0, start_offset=self.start_index, end_offset=self.end_index,
                       rest_hr_window=60, n_epochs_rest=10,
                       epoch_len=15, load_accel=True,
                       filter_data=False, low_f=1, high_f=30, f_type="bandpass",
                       load_raw=True, from_processed=False)

    def import_gold_standard_log(self):

        gs_log = pd.read_excel("/Users/kyleweber/Desktop/ECG Non-Wear/OND07_VisuallyInspectedECG_Nonwear.xlsx")
        self.gs_log = gs_log.loc[gs_log["ID"] == self.ecg.subject_id]

    def plot_raw(self, use_timestamps=True, write_dict=False):

        if use_timestamps:
            fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 6))
            ax1.plot(self.ecg.timestamps[::5], self.ecg.raw[::5], color='red')
            ax2.plot(self.ecg.timestamps[::10], self.ecg.accel_vm, color='black')
            xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
            ax2.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45, fontsize=8)

        if not use_timestamps:
            fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 6))
            ax1.plot(np.arange(0, len(self.ecg.raw[::5])) / 2, self.ecg.raw[::5], color='red')
            ax2.plot(np.arange(0, len(self.ecg.accel_vm)), self.ecg.accel_vm, color='black')
            plt.xticks(rotation=45, fontsize=8)

        plt.show()

        if write_dict:
            self.append_parameters_dict("/Users/kyleweber/Desktop/ECG Non-Wear/ECG_Nonwear_Parameters.csv")

    def run_ecg_fft(self, start=None, show_plot=True):

        print("\nPerforming FFT on ECG data starting at index {} using {}-second windows...".format(start,
                                                                                                    self.seg_length))

        if start is None:
            start = random.randint(0, len(self.ecg.raw) - self.seg_length * self.ecg.sample_rate)

        end = start + self.seg_length * self.ecg.sample_rate

        self.start_ind = start

        raw_fft = scipy.fft.fft(self.ecg.raw[start:end])
        filt_fft = scipy.fft.fft(self.ecg.filtered[start:end])

        xf = np.linspace(0.0, 1.0 / (2.0 * (1 / self.ecg.sample_rate)), (self.seg_length * self.ecg.sample_rate) // 2)

        if show_plot:
            fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 8))
            plt.suptitle("Index = {}".format(start))
            plt.subplots_adjust(hspace=.35)

            ax1.plot(np.arange(0, self.seg_length * self.ecg.sample_rate) / self.ecg.sample_rate,
                     self.ecg.raw[start:end],
                     color='red', label='Raw')
            ax1.plot(np.arange(0, self.seg_length * self.ecg.sample_rate) / self.ecg.sample_rate,
                     self.ecg.filtered[start:end],
                     color='black', label='Filt')
            ax1.legend()
            ax1.set_xlabel("Seconds")
            ax1.set_ylabel("Voltage")

            ax2.plot(xf, 2.0 / (self.seg_length * self.ecg.sample_rate) / 2 *
                     np.abs(raw_fft[0:(self.seg_length * self.ecg.sample_rate) // 2]),
                     color='red', label="Raw")

            ax2.set_ylabel("Power")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.legend()

            ax3.plot(xf, 2.0 / (self.seg_length * self.ecg.sample_rate) / 2 *
                     np.abs(filt_fft[0:(self.seg_length * self.ecg.sample_rate) // 2]),
                     color='black', label="Filt")
            ax3.fill_between(x=[self.ecg.low_f, self.ecg.high_f], y1=plt.ylim()[0], y2=plt.ylim()[1],
                             color='green', alpha=.5, label="Bandpass")
            ax3.set_ylabel("Power")
            ax3.set_xlabel("Frequency (Hz)")
            ax3.legend()

        df_raw_fft = pd.DataFrame(list(zip(xf,
                                           2.0 / (self.seg_length * self.ecg.sample_rate) /
                                           2 * np.abs(raw_fft[0:(self.seg_length * self.ecg.sample_rate) // 2]))),
                                  columns=["Frequency", "Power"])

        df = df_raw_fft.loc[df_raw_fft["Frequency"] >= .05]
        dom_f = round(df.loc[df["Power"] == df["Power"].max()]["Frequency"].values[0], 3)
        # dom_f = round(df_raw_fft.loc[df_raw_fft["Power"] == df_raw_fft["Power"].max()]["Frequency"].values[0], 3)
        print("-Dominant frequency = {} Hz".format(dom_f))
        self.parameters_dict["ECG_dom_f"] = dom_f

        return df_raw_fft

    def run_accel_fft(self, start=None, show_plot=True):

        if start is not None:
            print("\nPerforming FFT on accelerometer data from index {} using "
                  "{}-second epochs...".format(start, self.seg_length))
        if start is None:
            print("\nPerforming FFT on accelerometer data from random segment using "
                  "{}-second epochs...".format(self.seg_length))

        if start is None:
            start = random.randint(0, len(self.ecg.accel_x) - self.seg_length * self.ecg.accel_sample_rate)

        end = start + self.seg_length * self.ecg.accel_sample_rate

        fft_x = scipy.fft.fft(self.ecg.accel_x[start:end])
        fft_y = scipy.fft.fft(self.ecg.accel_y[start:end])
        fft_z = scipy.fft.fft(self.ecg.accel_z[start:end])
        fft_vm = scipy.fft.fft(self.ecg.accel_vm[start:end])

        xf = np.linspace(0.0, 1.0 / (2.0 * (1 / self.ecg.accel_sample_rate)),
                         (self.seg_length * self.ecg.accel_sample_rate) // 2)

        df_accel_fft = pd.DataFrame(list(zip(xf,
                                             2.0 / (self.seg_length * self.ecg.accel_sample_rate) /
                                             2 * np.abs(fft_x[0:(self.seg_length * self.ecg.accel_sample_rate) // 2]),
                                             2.0 / (self.seg_length * self.ecg.accel_sample_rate) /
                                             2 * np.abs(fft_y[0:(self.seg_length * self.ecg.accel_sample_rate) // 2]),
                                             2.0 / (self.seg_length * self.ecg.accel_sample_rate) /
                                             2 * np.abs(fft_z[0:(self.seg_length * self.ecg.accel_sample_rate) // 2]),
                                             2.0 / (self.seg_length * self.ecg.accel_sample_rate) /
                                             2 * np.abs(fft_vm[0:(self.seg_length * self.ecg.accel_sample_rate) // 2])
                                             )),
                                    columns=["Frequency", "Power_X", "Power_Y", "Power_Z", "Power_VM"])

        df_accel_fft = df_accel_fft.loc[df_accel_fft["Frequency"] >= .05]

        dom_fx = round(df_accel_fft.loc[df_accel_fft["Power_X"] ==
                                        df_accel_fft["Power_X"].max()]["Frequency"].values[0], 3)
        dom_fy = round(df_accel_fft.loc[df_accel_fft["Power_Y"] ==
                                        df_accel_fft["Power_Y"].max()]["Frequency"].values[0], 3)
        dom_fz = round(df_accel_fft.loc[df_accel_fft["Power_Z"] ==
                                        df_accel_fft["Power_Z"].max()]["Frequency"].values[0], 3)
        dom_fvm = round(df_accel_fft.loc[df_accel_fft["Power_VM"] ==
                                        df_accel_fft["Power_VM"].max()]["Frequency"].values[0], 3)

        self.parameters_dict["Accel_dom_f"] = [dom_fx, dom_fy, dom_fz, dom_fvm]
        print("-Dominant frequencies: X = {}Hz, Y = {}Hz, Z = {}Hz.".format(dom_fx, dom_fy, dom_fz))

        if show_plot:
            fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
            plt.suptitle("Index = {}".format(start))
            plt.subplots_adjust(hspace=.35)

            ax1.plot(np.arange(0, self.seg_length * self.ecg.accel_sample_rate) / self.ecg.accel_sample_rate,
                     self.ecg.accel_x[start:end],
                     color='red', label='X')
            ax1.plot(np.arange(0, self.seg_length * self.ecg.accel_sample_rate) / self.ecg.accel_sample_rate,
                     self.ecg.accel_y[start:end],
                     color='black', label='Y')
            ax1.plot(np.arange(0, self.seg_length * self.ecg.accel_sample_rate) / self.ecg.accel_sample_rate,
                     self.ecg.accel_z[start:end],
                     color='dodgerblue', label='Z')
            ax1.legend()
            ax1.set_xlabel("Seconds")
            ax1.set_ylabel("mG")

            ax2.plot(df_accel_fft["Frequency"], df_accel_fft["Power_X"],
                     color='red', label="X")

            ax2.plot(df_accel_fft["Frequency"], df_accel_fft["Power_Y"],
                     color='black', label="Y")

            ax2.plot(df_accel_fft["Frequency"], df_accel_fft["Power_Z"],
                     color='dodgerblue', label="Z")

            ax2.plot(df_accel_fft["Frequency"], df_accel_fft["Power_VM"],
                     color='grey', label="VM")

            ax2.set_ylabel("Power")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.legend()

        return df_accel_fft

    def plot_ecg_cumulative_fft(self, threshold=.9, show_plot=True):

        # Plots cumulative power and FFT
        self.ecg_fft["Normalized_CumulativePower"] = [i / self.ecg_fft["Power"].sum() for
                                                      i in self.ecg_fft["Power"].cumsum()]

        # Calculates frequency that corresponds to cumulative power at threshold
        cutoff_freq = round(self.ecg_fft.loc[self.ecg_fft["Normalized_CumulativePower"]
                                             >= threshold].iloc[0].values[0], 1)

        # Calculates frequency that corresponds to 0, 10, 20, 30%...power
        power_decades = [round(self.ecg_fft.loc[self.ecg_fft["Normalized_CumulativePower"]
                                                >= threshold].iloc[0].values[0], 1) for threshold in
                         np.arange(0, 1.0, .1)]

        self.parameters_dict["ECG_power"] = power_decades

        if show_plot:
            fig, (ax1, ax2) = plt.subplots(2)
            ax1.set_title("ECG Cumulative Power FFT: index = {}".format(self.start_ind))
            ax1.plot(self.ecg_fft["Frequency"], self.ecg_fft["Power"], label="FFT", color='green')
            ax1.set_ylabel("Power")
            ax1.legend()
            ax2.plot(self.ecg_fft["Frequency"], [i / self.ecg_fft["Power"].sum() for i in self.ecg_fft["Power"].cumsum()],
                     label='Cumulative power', color='black')
            ax2.axvline(x=cutoff_freq, linestyle='dashed', color='red',
                        label="{}% of power by {} Hz".format(threshold * 100, cutoff_freq))
            ax2.legend()
            ax2.set_ylabel("Normalized power")
            ax2.set_xlabel("Hz")

        return cutoff_freq

    def plot_accel_cumulative_fft(self, threshold=.9, axis="Z", start=None, show_plot=True):

        if start is None:
            start = int(self.start_ind / 10)

        # Plots cumulative power and FFT
        self.accel_fft["Normalized_CumulativePower"] = [i / self.accel_fft["Power_{}".format(axis)].sum() for
                                                        i in self.accel_fft["Power_{}".format(axis)].cumsum()]

        cutoff_freq = round(self.accel_fft.loc[self.accel_fft["Normalized_CumulativePower"]
                                               >= threshold].iloc[0].values[0], 1)

        # Calculates frequency that corresponds to 0, 10, 20, 30%...power
        power_decades = []

        for i, threshold in enumerate(np.arange(0, 1.0, .1)):
            try:
                val = round(self.accel_fft.loc[self.accel_fft["Power_{}".format(axis)]
                                               >= threshold].iloc[0].values[0], 1)
                power_decades.append(val)
            except IndexError:
                power_decades.append(power_decades[i-1])

        """
        power_decades = [round(self.accel_fft.loc[self.accel_fft["Power_{}".format(axis)]
                                                  >= threshold].iloc[0].values[0], 1) for threshold in
                           np.arange(0, 1.0, .1)]"""

        self.parameters_dict["Accel_power_{}".format(axis)] = power_decades

        if show_plot:
            fig, (ax1, ax2) = plt.subplots(2)
            ax1.set_title("Accelerometer Cumulative Power FFT")
            ax1.plot(self.accel_fft["Frequency"], self.accel_fft["Power_{}".format(axis)],
                     label="FFT", color='dodgerblue')
            ax1.set_ylabel("Power")
            ax1.legend()

            ax2.plot(self.accel_fft["Frequency"], [i / self.accel_fft["Power_{}".format(axis)].sum() for
                                                   i in self.accel_fft["Power_{}".format(axis)].cumsum()],
                     label='Cumulative power', color='black')
            ax2.axvline(x=cutoff_freq, linestyle='dashed', color='red',
                        label="{}% of power by {} Hz".format(threshold * 100, cutoff_freq))
            ax2.legend()
            ax2.set_ylabel("Normalized power")
            ax2.set_xlabel("Hz")

    def compare_accel_ffts(self, ind1, ind2):

        fig, (ax1, ax2) = plt.subplots(2, sharex='col')
        df_accel_fft = self.run_accel_fft(ind1, show_plot=False)
        ax1.plot(df_accel_fft["Frequency"],
                 [i / df_accel_fft["Power_Z"].sum() for i in df_accel_fft["Power_Z"].cumsum()],
                 label="Z_Sleep", color='black')
        ax1.fill_between(x=[1 / (60 / 12), 1 / (60 / 20)], y1=0, y2=1, color='green', alpha=.5,
                         label='Sleep breath rate')
        ax1.set_ylabel("Power")
        ax1.legend()

        df_accel_fft = self.run_accel_fft(ind2, show_plot=False)
        ax2.plot(df_accel_fft["Frequency"],
                 [i / df_accel_fft["Power_Z"].sum() for i in df_accel_fft["Power_Z"].cumsum()],
                 label="Z_Nonwear", color='black')
        ax2.fill_between(x=[1 / (60 / 12), 1 / (60 / 20)], y1=0, y2=1, color='green', alpha=.5,
                         label='Sleep breath rate')
        ax2.set_ylabel("Power")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.legend()

    def calculate_nonwear(self, plot_data=True):

        def find_nonwear():
            # First accel check: SD and range below threshold calculations -------------------------------------------
            print("\nPerforming non-wear detection algorithm...")

            accel_nw = []

            for i in np.arange(0, len(self.ecg.accel_x), self.ecg.accel_sample_rate * self.seg_length):
                sd_x = np.std(self.ecg.accel_x[i:i + self.ecg.accel_sample_rate * self.seg_length])
                sd_y = np.std(self.ecg.accel_y[i:i + self.ecg.accel_sample_rate * self.seg_length])
                sd_z = np.std(self.ecg.accel_z[i:i + self.ecg.accel_sample_rate * self.seg_length])
                axes_below_thresh = int(sd_x <= 3) + int(sd_y <= 3) + int(sd_z <= 3)

                range_x = max(self.ecg.accel_x[i:i + self.ecg.accel_sample_rate * self.seg_length]) - \
                          min(self.ecg.accel_x[i:i + self.ecg.accel_sample_rate * self.seg_length])
                range_y = max(self.ecg.accel_y[i:i + self.ecg.accel_sample_rate * self.seg_length]) - \
                          min(self.ecg.accel_y[i:i + self.ecg.accel_sample_rate * self.seg_length])
                range_z = max(self.ecg.accel_z[i:i + self.ecg.accel_sample_rate * self.seg_length]) - \
                          min(self.ecg.accel_z[i:i + self.ecg.accel_sample_rate * self.seg_length])

                axes_below_range = int(range_x <= 50) + int(range_y <= 50) + int(range_z <= 50)

                if axes_below_range >= 2 or axes_below_thresh >= 2:
                    accel_nw.append("Nonwear")
                else:
                    accel_nw.append("Wear")

            # Combines accelerometer and ECG non-wear characteristics: epoch-by-epoch ---------------------------------
            df_ecg = pd.DataFrame(list(zip(self.ecg.epoch_timestamps, self.ecg.epoch_validity,
                                           self.ecg.avg_voltage, self.ecg.svm, accel_nw)),
                                  columns=["Stamp", "Validity", "VoltRange", "SVM", "AccelNW"])

            nw = []
            for epoch in df_ecg.itertuples():
                if epoch.Validity == "Invalid" and epoch.AccelNW == "Nonwear" and epoch.VoltRange <= 400:
                    nw.append("Nonwear")
                else:
                    nw.append("Wear")

            # 5-minute windows ----------------------------------------------------------------------------------------
            t0 = datetime.now()
            final_nw = np.zeros(len(nw))
            for i in range(len(nw)):

                if final_nw[i] == "Wear" or final_nw[i] == "Nonwear":
                    pass

                if nw[i:i + 20].count("Nonwear") >= 19:
                    final_nw[i:i + 20] = 1

                    for j in range(i, len(nw)):
                        if nw[j] == "Nonwear":
                            pass
                        if nw[j] == "Wear":
                            stop_ind = j
                            if j > i:
                                final_nw[i:stop_ind] = 1

                else:
                    final_nw[i] = 0

            final_nw = ["Nonwear" if i == 1 else "Wear" for i in final_nw]
            t1 = datetime.now()
            print("Algorithm time = {} seconds.".format(round((t1 - t0).total_seconds(), 1)))

            return final_nw

        if self.nonwear is None:
            final_nw = find_nonwear()

        if self.nonwear is not None:
            print("Data already exists. Using previous data.")
            final_nw = self.nonwear

        if plot_data:

            print("Generating plot...")

            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 7))
            plt.suptitle(self.ecg.subject_id)
            ax1.plot(self.ecg.timestamps[::int(5 * self.ecg.sample_rate / 250)],
                     self.ecg.raw[::int(5 * self.ecg.sample_rate / 250)], color='black')
            ax1.set_ylabel("ECG Voltage")

            ax2.plot(self.ecg.timestamps[::int(10 * self.ecg.sample_rate / 250)], self.ecg.accel_x, color='dodgerblue')
            ax2.set_ylabel("Accel VM")

            ax3.plot(self.ecg.epoch_timestamps[0:min([len(self.ecg.epoch_timestamps), len(self.ecg.epoch_validity)])],
                     self.ecg.epoch_validity[0:min([len(self.ecg.epoch_timestamps), len(self.ecg.epoch_validity)])],
                     color='black')
            ax3.fill_between(x=self.ecg.epoch_timestamps[0:min([len(self.ecg.epoch_timestamps), len(final_nw)])],
                             y1="Wear", y2=final_nw, color='grey')

            if self.gs_log.shape[0] >= 1:
                for row in self.gs_log.itertuples():
                    ax1.fill_between(x=[row.Start, row.Stop],
                                     y1=min(self.ecg.filtered[::5]), y2=max(self.ecg.filtered[::5]),
                                     color='red', alpha=.5)
                    ax2.fill_between(x=[row.Start, row.Stop], y1=min(self.ecg.accel_x), y2=max(self.ecg.accel_x),
                                     color='red', alpha=.5)

            xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")

            ax3.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45, fontsize=8)

            # plt.savefig("/Users/kyleweber/Desktop/ECG Non-Wear/{}.png".format(self.ecg.subject_id))

        return final_nw

    def calculate_freq_cutoffs(self, epoch_len=15):

        ecg_cutoffs = []

        t0 = datetime.now()

        # Loops through all data
        for i in np.arange(0, len(self.ecg.raw), epoch_len * self.ecg.sample_rate):

            epoch_ind = int(i / self.ecg.sample_rate / epoch_len)

            try:
                if self.nonwear[epoch_ind] == "Wear":
                    ecg_cutoffs.append(None)

                if self.nonwear[epoch_ind] == "Nonwear":

                    self.ecg_fft = self.run_ecg_fft(start=i, show_plot=False)
                    self.ecg_fft["Normalized_CumulativePower"] = [i / self.ecg_fft["Power"].sum() for
                                                                  i in self.ecg_fft["Power"].cumsum()]

                    cutoff_freq = round(self.ecg_fft.loc[self.ecg_fft["Normalized_CumulativePower"]
                                                         >= .9].iloc[0].values[0], 1)

                    ecg_cutoffs.append(cutoff_freq)

            except IndexError:
                ecg_cutoffs.append(None)

        t1 = datetime.now()
        print("Time = ", round((t1 - t0).total_seconds(), 1), " seconds")

        return ecg_cutoffs

    def complete_parameter_dict(self):

        self.parameters_dict["ECG_Index"]  = self.start_index + 15 * self.ecg.sample_rate
        self.parameters_dict["Valid_ECG"] = self.ecg.epoch_validity[0]
        self.parameters_dict["ECG_volt_range"] = self.ecg.avg_voltage[0]

        self.parameters_dict["SD_X"] = round(np.std(self.ecg.accel_x), 2)
        self.parameters_dict["SD_Y"] = round(np.std(self.ecg.accel_y), 2)
        self.parameters_dict["SD_Z"] = round(np.std(self.ecg.accel_z), 2)
        self.parameters_dict["SVM"] = round(sum(self.ecg.accel_vm), 2)

        self.parameters_dict["VisualLength"] = int(self.end_index / self.ecg.sample_rate)

    def append_parameters_dict(self, data_file):

        from csv import DictWriter

        with open(data_file, "a", newline="\n") as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.parameters_dict.keys())

            dict_writer.writerow(self.parameters_dict)
            print("\nNew data appended to {}".format(data_file))

        df = pd.read_csv(data_file, usecols=["ID", "VisualNonwear"])

        print("-File contains {} records "
              "({} non-wear periods).".format(df.shape[0], df.loc[df["VisualNonwear"]=="Nonwear"].shape[0]))


rand_sub = random.randint(3002, 3044)
start_time, end_time, fs, duration = \
    ImportEDF.check_file("/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_{}_01_BF.edf".format(rand_sub),
                         print_summary=False)
# rand_start = randint(0, duration * fs - 15 * fs)
rand_start = randint(0, duration * fs - 45 * fs)

x = Data(subj_id=rand_sub, start_index=rand_start, end_index=45 * fs, seg_length=15)
x.import_ecg()

x.ecg_fft = x.run_ecg_fft(start=15*250, show_plot=False)
ecg_cutoff = x.plot_ecg_cumulative_fft(threshold=.9, show_plot=False)

x.accel_fft = x.run_accel_fft(start=15*25, show_plot=False)
x.plot_accel_cumulative_fft(threshold=.25, axis="X", show_plot=True)
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

    x.append_parameters_dict("/Users/kyleweber/Desktop/ECG Non-Wear/ECG_Nonwear_Parameters.csv")


fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 6), sharex='col')
plt.suptitle("{}: {}, {}".format(x.subj_id, x.parameters_dict["Valid_ECG"],
                                 datetime.strftime(datetime.strptime(str(x.ecg.timestamps[0])[:-3],
                                                                     "%Y-%m-%dT%H:%M:%S.%f"), "%I:%M:%S %p")))
if x.ecg.epoch_validity[0] == 'Valid':
    ax1.plot(np.arange(0, len(x.ecg.raw)) / x.ecg.sample_rate, x.ecg.raw, color='green', linestyle='-', label='raw')
    ax2.plot(np.arange(0, len(x.ecg.filtered)) / x.ecg.sample_rate, x.ecg.filtered, color='green', label='filt')
if x.ecg.epoch_validity[0] == 'Invalid':
    ax1.plot(np.arange(0, len(x.ecg.raw)) / x.ecg.sample_rate, x.ecg.raw, color='red', linestyle='-', label='raw')
    ax2.plot(np.arange(0, len(x.ecg.filtered)) / x.ecg.sample_rate, x.ecg.filtered, color='red', label='filt')
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

rax = plt.axes([.9, .5, .1, .15])
check = CheckButtons(rax, ("Nonwear", "Wear", "Unsure"), (False, False, False))
check.on_clicked(get_value)

plt.show()

"""
x = Data(3028)
x.import_ecg()
x.import_gold_standard_log()
x.nonwear = x.calculate_nonwear(epoch_len=15, plot_data=True)

# ECG processing
x.ecg_fft = x.run_ecg_fft(start=int(1.22e7), show_plot=True) # 3028, nonwear
x.ecg_fft = x.run_ecg_fft(start=int(14745000), show_plot=True) # 3028, sleep, wear, clean
cumulative_cutoff = x.plot_ecg_cumulative_fft(.9)

# Accelerometer processing
x.accel_fft = x.run_accel_fft(start=1474500, show_plot=True) # 3028, wear, sleep (1474500)
x.accel_fft = x.run_accel_fft(start=1200000, show_plot=True)  # 3028, nonwear (120000)
x.accel_fft = x.run_accel_fft(start=200000, show_plot=True)  # 3028, wear, awake (200000)
plot_accel_cumulative_fft(x, .25, axis="Z", start=200000)

x.compare_accel_ffts(ind1=1200000, ind2=1474500)  # 1200000 = nonwear; 1474500 = sleep

"""

"""
fig, (ax1, ax2) = plt.subplots(2, sharex='col')
ax1.plot(x.ecg.timestamps[::5], x.ecg.raw[::5], color='red')
ax1.set_ylabel("Voltage")
ax2.plot(x.ecg.timestamps[::15*250], ecg_cutoffs, color='green')
ax2.set_ylabel("Cutoff Freq. (90% power)")
ax2.axhline(y=70, linestyle='dashed', color='black')

df = pd.DataFrame(list(zip(x.ecg.timestamps[::15*250], x.ecg.epoch_validity, ecg_cutoffs, x.nonwear)),
                  columns=["Timestamp", "Validity", "Cutoff", "Nonwear"])

print(df.groupby("Nonwear").describe()["Cutoff"][["mean", "std"]])
df_valid = df.groupby("Nonwear").get_group("Wear")["Cutoff"]
df_invalid = df.groupby("Nonwear").get_group("Nonwear")["Cutoff"]

plt.hist(df_invalid, bins=np.arange(0, 125, 5), edgecolor='black', color='red',
         weights=np.ones(df_invalid.shape[0]) / df_invalid.shape[0] * 100, alpha=.5, label="Nonwear")
plt.hist(df_valid, bins=np.arange(0, 125, 5), edgecolor='black', color='green',
         weights=np.ones(df_valid.shape[0]) / df_valid.shape[0] * 100, alpha=.5, label="Wear")
plt.legend()
plt.ylabel("%")
plt.xlabel("Hz")
"""


class GoldStandardReview:

    def __init__(self, file="/Users/kyleweber/Desktop/ECG Non-Wear/ECG_Nonwear_Parameters.csv"):

        self.file = file

        self.df = None
        self.df_nw = None
        self.df_valid = None
        self.df_invalid_nonwear = None
        self.df_invalid_wear = None

        self.import_file()
        self.format_accel_dom_f()

    def import_file(self):

        self.df = pd.read_csv(self.file)

        self.df["Accel_avg"] = (self.df["SD_X"] + self.df["SD_Y"] + self.df["SD_Z"]) / 3
        self.df_nw = self.df.loc[self.df["VisualNonwear"] == "Nonwear"]
        self.df_valid = self.df.loc[self.df["Valid_ECG"] == "Valid"]
        self.df_invalid_wear = self.df.loc[(self.df["Valid_ECG"] == "Invalid") &
                                           (self.df["VisualNonwear"] == "Wear")]

    def format_accel_dom_f(self):
        self.df["Accel_dom_f_X"] = [self.df.iloc[i]["Accel_dom_f"][1:-1] for i in range(self.df.shape[0])]

        l = [self.df["Accel_dom_f_X"].iloc[i].split(",") for i in range(self.df.shape[0])]
        df = pd.DataFrame(l, columns=["Accel_dom_f_X", "Accel_dom_f_Y", "Accel_dom_f_Z"], dtype='float')
        self.df["Accel_dom_f_X"] = df["Accel_dom_f_X"]
        self.df["Accel_dom_f_Y"] = df["Accel_dom_f_Y"]
        self.df["Accel_dom_f_Z"] = df["Accel_dom_f_Z"]

        dead_data = self.df.drop("Accel_dom_f", axis=1)

    def generate_boxplot(self, col_name="ECG_volt_range"):

        boxprops = dict(linewidth=1.5)
        medianprops = dict(linewidth=1.5)

        units_dict = {"ECG_volt_range": "Voltage", "ECG_dom_f": "Hz", "SVM": "Counts", "Accel_avg": "Avg SD (mg)",
                      "Accel_dom_f_X": "Hz", "Accel_dom_f_Y": "Hz", "Accel_dom_f_Z": "Hz", }

        self.df.loc[self.df["VisualNonwear"] != "Unsure"].boxplot(column=[col_name],
                                                                  by=["Valid_ECG", "VisualNonwear"],
                                                                  grid=False,
                                                                  boxprops=boxprops, medianprops=medianprops,
                                                                  showfliers=False, showmeans=False,
                                                                  figsize=(10, 6))
        plt.ylabel(units_dict[col_name])

    def plot_descriptive_stats(self, data="ECG_volt_range", error_bars="SD"):

        tally = self.df.iloc[:, 2:].groupby(["VisualNonwear"]).count().transpose()
        means = self.df.iloc[:, 2:].groupby(["VisualNonwear"]).mean().transpose()
        stds = self.df.iloc[:, 2:].groupby(["VisualNonwear"]).std().transpose()

        import scipy.stats

        scipy.stats.t.ppf()
        if error_bars == "SEM":
            plt.bar(means.columns, means.loc[data], color=["darkgrey", "dodgerblue", "green"],
                    edgecolor='black',
                    alpha=.7, yerr=stds.loc[data] / np.sqrt(tally.loc[data]), capsize=4)

        if error_bars == "STD" or error_bars == "SD":
            plt.bar(means.columns, means.loc[data], color=["darkgrey", "dodgerblue", "green"],
                    edgecolor='black',
                    alpha=.7, yerr=stds.loc[data], capsize=4)


# x = GoldStandardReview()
# x.generate_boxplot("Accel_dom_f_Z")
