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
import pingouin as pg
import sklearn.metrics


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
                       age=0, start_offset=self.start_index if self.start_index is not None else 0,
                       end_offset=self.end_index if self.end_index is not None else 0,
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

    def run_ecg_fft(self, start=None, show_plot=True, print_data=True):

        if print_data:
            print("\nPerforming FFT on ECG data starting at index {} "
                  "using {}-second windows...".format(start, self.seg_length))

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

    def run_accel_fft(self, start=None, show_plot=True, print_data=True):

        if start is not None:
            if print_data:
                print("\nPerforming FFT on accelerometer data from index {} using "
                      "{}-second epochs...".format(start, self.seg_length))
        if start is None:
            if print_data:
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

    def calculate_nonwear2(self, volt_thresh=350, ecg_dom_f_thresh=5, accel_dom_f_thresh=0.5, accel_sd_thresh=25,
                           n_conditions=3, plot_data=True):
        """volt_thresh=350, ecg_dom_f_thresh=5, accel_dom_f_thresh=0.5, accel_sd_thresh=25, n_conditions=3"""

        print("\nPerforming non-wear detection algorithm...")

        nonwear_status = []

        # Loops through epochs
        for epoch in np.arange(0, len(self.ecg.epoch_validity)):

            # Condition check booleans
            volt_nonwear = False
            ecg_f_nonwear = False
            accel_f_nonwear = False
            accel_sd_nonwear = False

            # Calculates corresponding accelerometer data indexes
            accel_epoch_len = int(self.ecg.sample_rate / self.ecg.accel_sample_rate)
            raw_accel_ind = int(epoch * accel_epoch_len * self.ecg.accel_sample_rate)

            # Sets epoch as wear if ECG signal was valid
            if self.ecg.epoch_validity[epoch] == "Valid":
                nonwear_status.append("Wear")

            # Runs additional analyses if ECG signal is invalid =======================================================
            if self.ecg.epoch_validity[epoch] == "Invalid":

                # Calculates average axes SD if invalid ECG period ----------------------------------------------------
                sd_x = np.std(self.ecg.accel_x[raw_accel_ind:raw_accel_ind + accel_epoch_len])
                sd_y = np.std(self.ecg.accel_y[raw_accel_ind:raw_accel_ind + accel_epoch_len])
                sd_z = np.std(self.ecg.accel_z[raw_accel_ind:raw_accel_ind + accel_epoch_len])

                avg = (sd_x + sd_y + sd_z) / 3
                accel_sd_nonwear = True if avg <= accel_sd_thresh else False

                # Calculates dominant frequency in accel signal -------------------------------------------------------
                accel_fft = self.run_accel_fft(start=raw_accel_ind, show_plot=False)
                dom_f_x = accel_fft.loc[accel_fft["Power_X"] == max(accel_fft["Power_X"])]["Frequency"].iloc[0]

                accel_f_nonwear = True if dom_f_x >= accel_dom_f_thresh else False

                # Calculates dominant frequency in ECG signal ---------------------------------------------------------
                ecg_fft = self.run_ecg_fft(start=int(epoch * self.ecg.sample_rate), show_plot=False)
                dom_f_ecg = ecg_fft.loc[ecg_fft["Power"] == max(ecg_fft["Power"])]["Frequency"].iloc[0]

                ecg_f_nonwear = True if dom_f_ecg >= ecg_dom_f_thresh else False

                # Checks ECG voltage range ----------------------------------------------------------------------------
                volt_nonwear = True if self.ecg.avg_voltage[epoch] <= volt_thresh else False

                # Rule pass tally -------------------------------------------------------------------------------------
                if volt_nonwear + accel_sd_nonwear + accel_f_nonwear + ecg_f_nonwear >= n_conditions:
                    nonwear_status.append("Nonwear")

                if volt_nonwear + accel_sd_nonwear + accel_f_nonwear + ecg_f_nonwear < n_conditions:
                    nonwear_status.append("Wear")

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
            ax3.fill_between(x=self.ecg.epoch_timestamps[0:min([len(self.ecg.epoch_timestamps), len(nonwear_status)])],
                             y1="Wear", y2=nonwear_status, color='grey')

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

        return nonwear_status

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


"""
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

"""
x = Data(3028)
x.import_ecg()
x.import_gold_standard_log()
# x.nonwear = x.calculate_nonwear(plot_data=True)
x.nonwear = x.calculate_nonwear2(volt_thresh=350, ecg_dom_f_thresh=5, accel_dom_f_thresh=0.5,
                                 accel_sd_thresh=25, n_conditions=3, plot_data=True)
"""

"""
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

        self.df_ecg_power = None
        self.df_accel_power = None

        self.df_nw = None
        self.df_valid = None
        self.df_invalid_wear = None

        self.anova = None
        self.posthoc = None

        self.import_file()

    def import_file(self):

        print("\nImporting data...")

        self.df = pd.read_csv(self.file)

        self.df["Accel_avg"] = (self.df["SD_X"] + self.df["SD_Y"] + self.df["SD_Z"]) / 3

        self.df_nw = self.df.loc[self.df["VisualNonwear"] == "Nonwear"]
        self.df_valid = self.df.loc[self.df["Valid_ECG"] == "Valid"]
        self.df_invalid_wear = self.df.loc[(self.df["Valid_ECG"] == "Invalid") &
                                           (self.df["VisualNonwear"] == "Wear")]

        self.df["Group"] = self.df["Valid_ECG"] + self.df["VisualNonwear"]

        self.format_accel_dom_f()
        self.format_ecg_power()

    def format_accel_dom_f(self):
        """Converts strings to list of floats."""

        self.df["Accel_dom_f_X"] = [self.df.iloc[i]["Accel_dom_f"][1:-1] for i in range(self.df.shape[0])]

        l = [self.df["Accel_dom_f_X"].iloc[i].split(",") for i in range(self.df.shape[0])]

        for window in l:
            if len(window) == 3:
                window.append(None)

        df = pd.DataFrame(l, columns=["Accel_dom_f_X", "Accel_dom_f_Y", "Accel_dom_f_Z", "Accel_dom_f_VM"],
                          dtype='float')

        self.df["Accel_dom_f_X"] = df["Accel_dom_f_X"]
        self.df["Accel_dom_f_Y"] = df["Accel_dom_f_Y"]
        self.df["Accel_dom_f_Z"] = df["Accel_dom_f_Z"]
        self.df["Accel_dom_f_VM"] = df["Accel_dom_f_VM"]

        self.df["Accel_dom_f_avg"] = df["Accel_dom_f_X"] + df["Accel_dom_f_Y"] + df["Accel_dom_f_Z"]

        dead_data = self.df.drop("Accel_dom_f", axis=1)

    def format_ecg_power(self):
        """Converts strings to list of floats. Makes new df of power data"""

        print("\nFormatting ECG signal power data...")

        data = []
        for row in range(self.df.shape[0]):
            l = [float(i) for i in list(self.df.iloc[row]["ECG_power"][1:-1].split(","))]

            data.append(l)

        self.df["ECG_power"] = data

        self.df_ecg_power = pd.DataFrame([i for i in self.df["ECG_power"]],
                                         columns=["{}%".format(p) for p in np.arange(0, 100, 10)])
        self.df_ecg_power["VisualNonwear"] = self.df["VisualNonwear"]
        self.df_ecg_power["Valid_ECG"] = self.df["Valid_ECG"]
        self.df_ecg_power["Group"] = self.df["Group"]

    def analyze_ecg_power(self, error_range="95%CI"):
        """Analyzes cumulative ECG frequency power data."""

        print("\nAnalyzing ECG cumulative frequency power data...")

        df = self.df_ecg_power

        validwear = df.groupby("Group").get_group("ValidWear")
        invalidwear = df.groupby("Group").get_group("InvalidWear")
        nonwear = df.groupby("Group").get_group("InvalidNonwear")

        nw_desc = nonwear.describe().loc[["mean", "std", "count"]].transpose()
        nw_desc["95%CI"] = nw_desc["std"] / np.sqrt(nw_desc["count"]) * scipy.stats.t.ppf(.95, nw_desc["count"] - 1)

        wear_desc = validwear.describe().loc[["mean", "std", "count"]].transpose()
        wear_desc["95%CI"] = wear_desc["std"] / np.sqrt(wear_desc["count"]) * \
                             scipy.stats.t.ppf(.95, wear_desc["count"] - 1)

        invalidwear_desc = invalidwear.describe().loc[["mean", "std", "count"]].transpose()
        invalidwear_desc["95%CI"] = invalidwear_desc["std"] / np.sqrt(invalidwear_desc["count"]) * \
                                    scipy.stats.t.ppf(.95, invalidwear_desc["count"] - 1)

        plt.fill_between(x=np.arange(0, 100, 10),
                         y1=nw_desc['mean'] - nw_desc[error_range], y2=nw_desc['mean'],
                         label="Nonwear", color='red', alpha=.25)
        plt.fill_between(x=np.arange(0, 100, 10),
                         y1=nw_desc['mean'], y2=nw_desc['mean'] + nw_desc[error_range],
                         color='red', alpha=.25)

        plt.fill_between(x=np.arange(0, 100, 10),
                         y1=wear_desc['mean'] - wear_desc[error_range], y2=wear_desc['mean'],
                         label="Wear", color='green', alpha=.25)
        plt.fill_between(x=np.arange(0, 100, 10),
                         y1=wear_desc['mean'], y2=wear_desc['mean'] + wear_desc[error_range], color='green', alpha=.25)

        plt.fill_between(x=np.arange(0, 100, 10),
                         y1=invalidwear_desc['mean'] - invalidwear_desc[error_range], y2=invalidwear_desc['mean'],
                         label="Invalid", color='orange', alpha=.25)
        plt.fill_between(x=np.arange(0, 100, 10),
                         y1=invalidwear_desc['mean'], y2=invalidwear_desc['mean'] + invalidwear_desc[error_range],
                         color='orange', alpha=.25)

        plt.legend(loc='upper left')
        plt.title("Cumulative ECG Frequency Power ({})".format(error_range))
        plt.xlabel("% of signal power ≤ value")
        plt.ylabel("Hz")
        plt.xlim(0, 100)
        plt.xticks(np.arange(0, 110, 10))
        plt.yticks(np.arange(0, plt.ylim()[1], 25))

    def analyze_accel_power(self, error_range="95%CI"):
        """Analyzes cumulative Accelerometer frequency power data."""

        print("\nAnalyzing Accelerometer cumulative frequency power data...")

        df = self.df_accel_power

        validwear = df.groupby("Group").get_group("ValidWear")
        invalidwear = df.groupby("Group").get_group("InvalidWear")
        nonwear = df.groupby("Group").get_group("InvalidNonwear")

        nw_desc = nonwear.describe().loc[["mean", "std", "count"]].transpose()
        nw_desc["95%CI"] = nw_desc["std"] / np.sqrt(nw_desc["count"]) * scipy.stats.t.ppf(.95, nw_desc["count"] - 1)

        wear_desc = validwear.describe().loc[["mean", "std", "count"]].transpose()
        wear_desc["95%CI"] = wear_desc["std"] / np.sqrt(wear_desc["count"]) * \
                             scipy.stats.t.ppf(.95, wear_desc["count"] - 1)

        invalidwear_desc = invalidwear.describe().loc[["mean", "std", "count"]].transpose()
        invalidwear_desc["95%CI"] = invalidwear_desc["std"] / np.sqrt(invalidwear_desc["count"]) * \
                                    scipy.stats.t.ppf(.95, invalidwear_desc["count"] - 1)

        plt.fill_between(x=np.arange(0, 100, 10),
                         y1=nw_desc['mean'] - nw_desc[error_range], y2=nw_desc['mean'],
                         label="Nonwear", color='red', alpha=.25)
        plt.fill_between(x=np.arange(0, 100, 10),
                         y1=nw_desc['mean'], y2=nw_desc['mean'] + nw_desc[error_range],
                         color='red', alpha=.25)

        plt.fill_between(x=np.arange(0, 100, 10),
                         y1=wear_desc['mean'] - wear_desc[error_range], y2=wear_desc['mean'],
                         label="Wear", color='green', alpha=.25)
        plt.fill_between(x=np.arange(0, 100, 10),
                         y1=wear_desc['mean'], y2=wear_desc['mean'] + wear_desc[error_range], color='green', alpha=.25)

        plt.fill_between(x=np.arange(0, 100, 10),
                         y1=invalidwear_desc['mean'] - invalidwear_desc[error_range], y2=invalidwear_desc['mean'],
                         label="Invalid", color='orange', alpha=.25)
        plt.fill_between(x=np.arange(0, 100, 10),
                         y1=invalidwear_desc['mean'], y2=invalidwear_desc['mean'] + invalidwear_desc[error_range],
                         color='orange', alpha=.25)

        plt.legend(loc='upper left')
        plt.title("Cumulative ECG Frequency Power ({})".format(error_range))
        plt.xlabel("% of signal power ≤ value")
        plt.ylabel("Hz")
        plt.xlim(0, 100)
        plt.xticks(np.arange(0, 110, 10))
        plt.yticks(np.arange(0, plt.ylim()[1], 25))

    def generate_boxplot(self, col_name):

        boxprops = dict(linewidth=1.5)
        medianprops = dict(linewidth=1.5)

        units_dict = {"ECG_volt_range": "Voltage", "ECG_dom_f": "Hz", "SVM": "Counts", "Accel_avg": "Avg SD (mg)",
                      "Accel_dom_f_X": "Hz", "Accel_dom_f_Y": "Hz", "Accel_dom_f_Z": "Hz", "Accel_dom_f_VM": "Hz"}

        self.df.loc[self.df["VisualNonwear"] != "Unsure"].boxplot(column=[col_name],
                                                                  by=["Valid_ECG", "VisualNonwear"],
                                                                  grid=False,
                                                                  boxprops=boxprops, medianprops=medianprops,
                                                                  showfliers=False, showmeans=False,
                                                                  figsize=(10, 6))
        plt.ylabel(units_dict[col_name])

    def perform_anova(self, data, error_bars="SD", plot_type="bar", save_dir=None):

        # Runs one-way ANOVA ------------------------------------------------------------------------------------------

        df = self.df
        self.anova = pg.anova(data=df, dv=data, between="Group", detailed=True)
        print("\nOne-way ANOVA results:")
        print(self.anova.round(3)[["Source", "F", "p-unc"]])

        validwear = df.groupby("Group").get_group("ValidWear")[data]
        invalidwear = df.groupby("Group").get_group("InvalidWear")[data]
        nonwear = df.groupby("Group").get_group("InvalidNonwear")[data]

        print("\nUnpaired T-Test results:")
        pair1 = pg.ttest(validwear, invalidwear, paired=False)[["T", "dof", "p-val", "CI95%", "cohen-d", "power"]]
        pair2 = pg.ttest(validwear, nonwear, paired=False)[["T", "dof", "p-val", "CI95%", "cohen-d", "power"]]
        pair3 = pg.ttest(nonwear, invalidwear, paired=False)[["T", "dof", "p-val", "CI95%", "cohen-d", "power"]]

        df_ttest = pair1.append(other=(pair2, pair3))
        df_ttest.index = ["ValidWear-InvalidWear", "ValidWear-Nonwear", "Nonwear-InvalidWear"]
        df_ttest.insert(loc=0, column="Variable", value=[data, data, data])
        print(df_ttest.round(3)[["T", "p-val", "cohen-d"]])

        self.posthoc = df_ttest

        # PLOTTING ---------------------------------------------------------------------------------------------------

        tally = df.iloc[:, 2:].groupby(["Group"]).count().transpose()
        means = df.iloc[:, 2:].groupby(["Group"]).mean().transpose()
        stds = df.iloc[:, 2:].groupby(["Group"]).std().transpose()

        units_dict = {"ECG_volt_range": "Voltage", "ECG_dom_f": "Hz", "SVM": "Counts", "Accel_avg": "Avg SD (mg)",
                      "Accel_dom_f_X": "Hz", "Accel_dom_f_Y": "Hz", "Accel_dom_f_Z": "Hz",
                      "Accel_dom_f_VM": "Hz", "Accel_dom_f_avg": "Hz"}

        import scipy.stats
        plt.close("all")

        if error_bars == "SEM":
            plt.bar(means.columns, means.loc[data], color=["red", "grey", "darkorange", "green"],
                    edgecolor='black',
                    alpha=.7, yerr=stds.loc[data] / np.sqrt(tally.loc[data]), capsize=4)

        if error_bars == "STD" or error_bars == "SD":
            plt.bar(means.columns, means.loc[data], color=["red", "grey", "darkorange", "green"],
                    edgecolor='black',
                    alpha=.7, yerr=stds.loc[data], capsize=4)

        if "CI" in error_bars or "ci" in error_bars:
            t_crits = [scipy.stats.t.ppf(q=int(error_bars.split("%")[0])/100, df=i-1) for
                       i in [tally.loc[data].iloc[j] for j in range(tally.shape[1])]]

            cis = [stds.loc[data].iloc[i] / np.sqrt(tally.loc[data].iloc[i]) * t_crits[i]
                   for i in range(tally.shape[1])]

            if plot_type == "bar":
                plt.bar(means.columns, means.loc[data], color=["red", "grey", "darkorange", "green"],
                        edgecolor='black',
                        alpha=.7, yerr=cis, capsize=4)

        plt.title("{} data (mean ± {})".format(data, error_bars))
        plt.ylabel(units_dict[data])
        plt.xlabel("Data Description")

        if save_dir is not None:
            plt.savefig(save_dir + "ANOVA_Output_{}.png".format(data))
            self.anova.to_excel(save_dir + "ANOVA_{}.xlsx".format(data), index=False)
            self.posthoc.to_excel(save_dir + "Posthoc_{}.xlsx".format(data))

    def recalculate_nonwear(self, volt_thresh=None, ecg_dom_f_thresh=None, accel_dom_f_thresh=None,
                            accel_sd_thresh=None, ecg_perc_thresh=None, ecg_f_thresh=None,
                            n_conditions=None, print_data=True):

        if print_data:
            print("\nRunning algorithm:")
            if volt_thresh is not None:
                print("-Voltage range threshold = {} mV".format(volt_thresh))
            if ecg_dom_f_thresh is not None:
                print("-ECG dominant frequency threshold = {} Hz".format(ecg_dom_f_thresh))
            if ecg_perc_thresh is not None and ecg_f_thresh is not None:
                print("-ECG cumulative frequency threshold = {}% and {} Hz".format(ecg_perc_thresh, ecg_f_thresh))
            if accel_sd_thresh is not None:
                print("-Accelerometer SD threshold = {}".format(accel_sd_thresh))
            if accel_dom_f_thresh is not None:
                print("-Accelerometer dominant threshold = {} Hz".format(accel_dom_f_thresh))
            print("-Requires {} conditions to be classified as nonwear".format(n_conditions))

        var_use_dict = {"ECG_volt_range": True if volt_thresh is not None else False,
                        "ECG_dom_f": True if ecg_dom_f_thresh is not None else False,
                        "Accel_dom_f": True if accel_dom_f_thresh is not None else False,
                        "Accel_avg": True if accel_sd_thresh is not None else False,
                        "ECG_cf_perc": True if ecg_perc_thresh is not None else False,
                        "ECG_cf_f": True if ecg_f_thresh is not None else False}

        df = self.df.loc[self.df["Group"] != "InvalidUnsure"]
        df = df[["Valid_ECG", "VisualNonwear", "Group", "ECG_volt_range", "ECG_dom_f", "Accel_avg", "Accel_dom_f_avg"]]

        if ecg_perc_thresh is not None:
            df["ECG_cf_f"] = self.df_ecg_power[ecg_perc_thresh]
        if ecg_perc_thresh is None:
            df["ECG_cf_f"] = [None for i in range(df.shape[0])]

        outcome = []
        for tup in df.itertuples():

            var_dict = {"ECG_volt_range": True if volt_thresh is not None else False,
                        "ECG_dom_f": True if ecg_dom_f_thresh is not None else False,
                        "Accel_dom_f": True if accel_dom_f_thresh is not None else False,
                        "Accel_SD": True if accel_sd_thresh is not None else False,
                        "ECG_cf_perc": True if ecg_perc_thresh is not None else False,
                        "ECG_cf_f": True if ecg_f_thresh is not None else False}

            if volt_thresh is not None:
                if tup.ECG_volt_range > volt_thresh:
                    var_dict["ECG_volt_range"] = False

            if ecg_dom_f_thresh is not None:
                if tup.ECG_dom_f < ecg_dom_f_thresh:
                    var_dict["ECG_dom_f"] = False

            if accel_dom_f_thresh is not None:
                if tup.Accel_dom_f_avg < accel_dom_f_thresh:
                    var_dict["Accel_dom_f"] = False

            if accel_sd_thresh is not None:
                if tup.Accel_avg > accel_sd_thresh:
                    var_dict["Accel_SD"] = False

            if ecg_perc_thresh is not None and ecg_f_thresh is not None:
                if ecg_f_thresh <= tup.ECG_cf_f:
                    var_dict["ECG_cf_f"] = False

            # Determines how many conditions indicate non-wear
            if [i for i in var_dict.values()].count(True) >= n_conditions:
                outcome.append("Nonwear")
            if [i for i in var_dict.values()].count(True) < n_conditions:
                outcome.append("Wear")

        df["Outcome"] = outcome

        sens, spec = self.calculate_algorithm_performance(df=df, print_data=print_data)

        tp = df.loc[(df["VisualNonwear"] == "Wear") & (df["Outcome"] == "Wear")].shape[0]
        tn = df.loc[(df["VisualNonwear"] == "Nonwear") & (df["Outcome"] == "Nonwear")].shape[0]
        correct = tp + tn
        perc_accuracy = round(100 * correct / df.shape[0], 1)

        performance_dict = {"N conditions": n_conditions,
                            "ECG_volt_range": volt_thresh, "ECG_dom_f": ecg_dom_f_thresh,
                            "Accel_dom_f_avg": accel_dom_f_thresh, "Accel_avg": accel_sd_thresh,
                            "ECG_cf_perc": ecg_perc_thresh, "ECG_cf_f": ecg_f_thresh,
                            "% Accuracy": perc_accuracy,
                            "Sensitivity": sens, "Specificity": spec,
                            "AUC": round(sklearn.metrics.roc_auc_score(y_true=[0 if i == "Wear" else 1 for
                                                                               i in df["VisualNonwear"]],
                                                                       y_score=[0 if i == "Wear" else 1 for
                                                                                i in df["Outcome"]]), 3),
                            "Distance": np.sqrt((1 - sens) ** 2 + (1 - spec) ** 2),
                            "Youden": round(sens + spec - 1, 1)}

        return performance_dict, df

    @staticmethod
    def calculate_algorithm_performance(df, print_data=True):
        """Calculates sensitivity and specificity. A nonwear period is considered a 'positive'."""

        # False positive: algorithm says nonwear; gold standard says wear
        fp = df.loc[(df["VisualNonwear"] == "Wear") & (df["Outcome"] == "Nonwear")].shape[0]

        # False negative: algorithm says wear; gold standard says nonwear
        fn = df.loc[(df["VisualNonwear"] == "Nonwear") & (df["Outcome"] == "Wear")].shape[0]

        # True positive: algorithm says nonwear; gold standard says nonwear
        tp = df.loc[(df["VisualNonwear"] == "Nonwear") & (df["Outcome"] == "Nonwear")].shape[0]

        # True negative: algorithm says wear; gold standard says wear
        tn = df.loc[(df["VisualNonwear"] == "Wear") & (df["Outcome"] == "Wear")].shape[0]

        sens = round(tp / (tp+fn), 3)
        spec = round(tn / (tn+fp), 3)
        youden = round(sens + spec - 1, 3)
        perc_acc = round(100 * (tp + tn) / df.shape[0], 1)

        if print_data:
            print("\nAlgorithm performance on gold standard dataset:")
            print("-Accuracy = {}% \n-Sensitivity = {} "
                  "\n-Specificity = {} \n-Youden's index = {}".format(perc_acc, sens, spec, youden))

        return sens, spec

    def perform_roc(self, volt_ranges=np.arange(50, 750, 100), ecg_f=np.arange(2.5, 27.5, 2.5),
                    accel_f=np.arange(0.5, 3.5, .25), accel_sd=np.arange(1, 30, 3), n_conditions=3):
        """Loops through all combinations of parameters given as arguments.
           Returns df of best AUC and minimum distance to perfect accuracy.
        """

        print("\nRunning ROC analysis...Please wait a while...")

        t0 = datetime.now()

        n_combos = len(volt_ranges) * len(ecg_f) * len(accel_f) * len(accel_sd)

        data = []
        tally = 0
        percent_markers = np.arange(0, n_combos * 1.1, n_combos/10)

        for v in volt_ranges:
            for e in ecg_f:
                for accel in accel_f:
                    for sd in accel_sd:
                        tally += 1

                        if tally in percent_markers:
                            print("{}% done ({} seconds)...".format(int(100*tally/n_combos),
                                                                    round((datetime.now() - t0).total_seconds(), 1)))

                        perf, results = self.recalculate_nonwear(volt_thresh=v, ecg_dom_f_thresh=e,
                                                                 accel_dom_f_thresh=accel, accel_sd_thresh=sd,
                                                                 n_conditions=n_conditions, print_data=False)

                        data.append([i for i in perf.values()])

        df = pd.DataFrame(data, columns=[i for i in perf.keys()])

        best_auc = df.loc[df["AUC"] == max(df["AUC"])]
        best_dist = df.loc[df["Distance"] == min(df["Distance"])]
        best_youden = df.loc[df["Youden"] == max(df["Youden"])]

        t1 = datetime.now()
        print("\n=====================================================================================================")
        print("TOTAL PROCESSING TIME: {} seconds.".format(round((t1-t0).total_seconds(), 1)))
        print("-Tested {} combinations.".format(df.shape[0]))
        print("-Required {}/4 conditions to be met for non-wear.".format(n_conditions))
        print("    -Best AUC = {}, accuracy = {}%, "
              "sensitivity = {}, specificity = {}".format(best_auc.iloc[0]["AUC"].round(3),
                                                          best_auc.iloc[0]["% Accuracy"].round(3),
                                                          best_auc.iloc[0]["Sensitivity"].round(3),
                                                          best_auc.iloc[0]["Specificity"].round(3)))
        print("    -Best distance = {}, accuracy = {}%, "
              "sensitivity = {}, specificity = {}".format(best_dist.iloc[0]["Distance"].round(3),
                                                          best_dist.iloc[0]["% Accuracy"].round(3),
                                                          best_dist.iloc[0]["Sensitivity"].round(3),
                                                          best_dist.iloc[0]["Specificity"].round(3)))
        print("    -Best Youden's index = {}, accuracy = {}%, "
              "sensitivity = {}, specificity = {}".format(best_youden.iloc[0]["Youden"].round(3),
                                                          best_youden.iloc[0]["% Accuracy"].round(3),
                                                          best_youden.iloc[0]["Sensitivity"].round(3),
                                                          best_youden.iloc[0]["Specificity"].round(3)))

        return best_auc, best_dist, best_youden


d = GoldStandardReview()
# d.generate_boxplot("Accel_avg")
# d.perform_anova(data="SVM", error_bars="95%CI", save_dir="/Users/kyleweber/Desktop/")
# d.perform_anova(data="ECG_volt_range", error_bars="95%CI", save_dir=None)
# d.analyze_ecg_power(error_range="95%CI")

"""
# BEST PARAMETERS SO FAR
perf, results = d.recalculate_nonwear(volt_thresh=350, ecg_dom_f_thresh=5, accel_dom_f_thresh=0.5, accel_sd_thresh=25,
                                      ecg_perc_thresh=None, ecg_f_thresh=None, n_conditions=3)

perf, results = d.recalculate_nonwear(volt_thresh=350, ecg_dom_f_thresh=5, accel_dom_f_thresh=.5,
                                      accel_sd_thresh=None, ecg_perc_thresh="80%", ecg_f_thresh=30, n_conditions=2)
"""
# auc, dist, youden = d.perform_roc(n_conditions=3)

# Make perform_roc able to pick which variables to use/not use
