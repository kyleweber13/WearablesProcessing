import ImportEDF

from ecgdetectors import Detectors
# https://github.com/luishowell/ecg-detectors

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statistics
import scipy.stats as stats
from datetime import datetime
import csv
import progressbar
from matplotlib.ticker import PercentFormatter
from random import randint
import matplotlib.dates as mdates
from ImportEDF import Bittium


# --------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------- ECG CLASS OBJECT ------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


class ECG:

    def __init__(self, subject_id=None, filepath=None, output_dir=None, processed_folder=None,
                 processed_file=None,
                 age=0, start_offset=0, end_offset=0,
                 rest_hr_window=60, n_epochs_rest=10,
                 epoch_len=15, load_accel=False,
                 filter_data=False, low_f=1, high_f=30, f_type="bandpass",
                 load_raw=False, from_processed=True):
        """Class that contains raw and processed ECG data.

        :argument
        DATA IMPORT
        -filepath: full pathway to EDF file
        -load_raw: boolean of whether to load raw ECG data. Can be used in addition to from_processed = True
        -from_processed: boolean of whether to read in already processed data
                         (epoch timestamps, epoch HR, quality control check)
        -output_dir: where files are written to OR where processed data files are read in from
        -start_offset, end_offset: indexes used to crop data to match other devices

        DATA EPOCHING
        -rest_hr_window: number of seconds over which HR is averaged when calculating resting HR
            -Creates a rolling average of HR over this many seconds (rounded to match epoch length)
        -n_epochs_rest: number of epochs used in the resting HR calculation
                        (averages HR over the n_epochs_rest lower HRs)
        -epoch_len: time period over which data is processed, seconds

        FILTERING
        -filter: whether or not to filter the data
        -low_f, high_1: cut-off frequencies for the filter. Set to None if irrelevant. In Hz.
        -f_type: type of filter; "lowpass", "highpass", "bandpass"

        OTHER
        -age: participant age in years. Needed for HRmax calculation.
        """

        print()
        print("============================================= ECG DATA ==============================================")

        self.filepath = filepath
        self.processed_file = processed_file
        self.subject_id = subject_id
        self.output_dir = output_dir
        self.processed_folder = processed_folder
        self.age = age
        self.epoch_len = epoch_len
        self.rest_hr_window = rest_hr_window
        self.n_epochs_rest = n_epochs_rest
        self.start_offset = start_offset
        self.end_offset = end_offset

        self.filter_data = filter_data
        self.low_f = low_f
        self.high_f = high_f
        self.f_type = f_type

        self.load_raw = load_raw
        self.load_accel = load_accel
        self.from_processed = from_processed

        self.accel_sample_rate = 1
        self.accel_x = None
        self.accel_y = None
        self.accel_z = None
        self.accel_vm = None
        self.svm = []

        # Raw data
        if self.load_raw:
            self.ecg = ImportEDF.Bittium(filepath=self.filepath, load_accel=self.load_accel,
                                         start_offset=self.start_offset, end_offset=self.end_offset,
                                         low_f=self.low_f, high_f=self.high_f, f_type=self.f_type)

            self.sample_rate = self.ecg.sample_rate
            self.accel_sample_rate = self.ecg.accel_sample_rate
            self.raw = self.ecg.raw
            self.filtered = self.ecg.filtered
            self.timestamps = self.ecg.timestamps
            self.epoch_timestamps = self.ecg.epoch_timestamps

            self.accel_x, self.accel_y, self.accel_z, self.accel_vm = self.ecg.x, self.ecg.y, self.ecg.z, self.ecg.vm

            del self.ecg

        if self.load_accel:
            self.epoch_accel()

        # Performs quality control check on raw data and epochs data
        if self.from_processed:
            self.epoch_validity, self.epoch_hr = None, None
        if not self.from_processed:
            self.epoch_validity, self.epoch_hr, self.avg_voltage, self.rr_sd, self.r_peaks = self.check_quality()

        # Loads epoched data from existing file
        if self.from_processed:
            self.load_processed()

        # List of epoched heart rates but any invalid epoch is marked as None instead of 0 (as is self.epoch_hr)
        self.valid_hr = [self.epoch_hr[i] if self.epoch_validity[i] == 0
                         else None for i in range(len(self.epoch_hr))]

        self.quality_report = self.generate_quality_report()

        self.rolling_avg_hr = None
        self.rest_hr = None
        self.perc_hrr = None
        self.epoch_intensity = None
        self.epoch_intensity_totals = None

        self.nonwear = None

    def epoch_accel(self):

        for i in range(0, len(self.accel_vm), int(self.accel_sample_rate * self.epoch_len)):

            if i + self.epoch_len * self.accel_sample_rate > len(self.accel_vm):
                break

            vm_sum = sum(self.accel_vm[i:i + self.epoch_len * self.accel_sample_rate])

            self.svm.append(round(vm_sum, 5))

    def check_quality(self):
        """Performs quality check using Orphanidou et al. (2015) algorithm that has been tweaked to factor in voltage
           range as well.

           This function runs a loop that creates object from the class CheckQuality for each epoch in the raw data.
        """

        print("\n" + "Running quality check with Orphanidou et al. (2015) algorithm...")

        t0 = datetime.now()

        validity_list = []  # window's validity (binary; 1 = invalid)
        epoch_hr = []  # window's HRs
        avg_voltage = []  # window's voltage range
        rr_sd = []  # window's RR SD
        r_peaks = []  # all R peak indexes

        bar = progressbar.ProgressBar(maxval=len(self.raw),
                                      widgets=[progressbar.Bar('>', '', '|'), ' ',
                                               progressbar.Percentage()])
        bar.start()

        for start_index in range(0, int(len(self.raw)), self.epoch_len * self.sample_rate):
            bar.update(start_index + 1)

            qc = CheckQuality(ecg_object=self, start_index=start_index, epoch_len=self.epoch_len)

            avg_voltage.append(qc.volt_range)

            if qc.valid_period:
                validity_list.append("Valid")
                epoch_hr.append(round(qc.hr, 2))
                rr_sd.append(qc.rr_sd)

                for peak in qc.r_peaks_index_all:
                    r_peaks.append(peak)
                for peak in qc.removed_peak:
                    r_peaks.append(peak + start_index)

                r_peaks = sorted(r_peaks)

            if not qc.valid_period:
                validity_list.append("Invalid")
                epoch_hr.append(0)
                rr_sd.append(0)

        bar.finish()

        t1 = datetime.now()
        proc_time = (t1 - t0).seconds
        print("\n" + "Quality check complete ({} seconds).".format(round(proc_time, 2)))
        print("-Processing time of {} seconds per "
              "hour of data.".format(round(proc_time / (len(self.raw)/self.sample_rate/3600)), 2))

        return validity_list, epoch_hr, avg_voltage, rr_sd, r_peaks

    def generate_quality_report(self):
        """Calculates how much of the data was usable. Returns values in dictionary."""

        invalid_epochs = self.epoch_validity.count("Invalid")  # number of invalid epochs
        hours_lost = round(invalid_epochs / (60 / self.epoch_len) / 60, 2)  # hours of invalid data
        perc_invalid = round(invalid_epochs / len(self.epoch_validity) * 100, 1)  # percent of invalid data

        quality_report = {"Invalid epochs": invalid_epochs, "Hours lost": hours_lost,
                          "Percent invalid": perc_invalid,
                          "Average valid duration (minutes)": None}

        print("-{}% of the data is valid.".format(round(100 - perc_invalid), 3))

        return quality_report

    def load_processed(self):

        df = pd.read_csv(self.processed_file)

        self.epoch_timestamps = [i for i in pd.to_datetime(df["Timestamps"])]
        self.epoch_validity = [i for i in df["ECG_Validity"]]
        self.epoch_hr = [i for i in df["HR"]]

    def find_resting_hr(self, window_size, n_windows, sleep_status=None, start_index=None, end_index=None):
        """Function that calculates resting HR based on inputs.

        :argument
        -window_size: size of window over which rolling average is calculated, seconds
        -n_windows: number of epochs over which resting HR is averaged (lowest n_windows number of epochs)
        -sleep_status: data from class Sleep that corresponds to asleep/awake epochs
        """

        if start_index is not None and end_index is not None:
            epoch_hr = np.array(self.epoch_hr[start_index:end_index])
        else:
            epoch_hr = self.epoch_hr

        # Sets integer for window length based on window_size and epoch_len
        window_len = int(window_size / self.epoch_len)

        try:
            rolling_avg = [statistics.mean(epoch_hr[i:i + window_len]) if 0 not in epoch_hr[i:i + window_len]
                           else None for i in range(len(epoch_hr))]
        except statistics.StatisticsError:
            print("No data points found.")
            rolling_avg = []

        # Calculates resting HR during waking hours if sleep_log available --------------------------------------------
        if sleep_status is not None:
            print("\n" + "Calculating resting HR from periods of wakefulness...")

            awake_hr = [rolling_avg[i] for i in range(0, min([len(sleep_status), len(rolling_avg)]))
                        if sleep_status[i] == 0 and rolling_avg[i] is not None]

            sorted_hr = sorted(awake_hr)

            if len(sorted_hr) < n_windows:
                resting_hr = "N/A"

            if len(sorted_hr) >= n_windows:
                resting_hr = round(sum(sorted_hr[0:n_windows]) / n_windows, 1)

            print("Resting HR (average of {} lowest {}-second periods while awake) is {} bpm.".format(n_windows,
                                                                                                      window_size,
                                                                                                      resting_hr))

        # Calculates resting HR during all hours if sleep_log not available -------------------------------------------
        if sleep_status is None:
            # print("\n" + "Calculating resting HR from periods of all data (sleep data not available)...")

            awake_hr = None

            valid_hr = [i for i in rolling_avg if i is not None]

            sorted_hr = sorted(valid_hr)

            resting_hr = round(sum(sorted_hr[:n_windows]) / n_windows, 1)

            print("Resting HR (sleep not removed; average of {} lowest "
                  "{}-second periods) is {} bpm.".format(n_windows, window_size, resting_hr))

        return rolling_avg, resting_hr, awake_hr

    def calculate_percent_hrr(self):
        """Calculates HR as percent of heart rate reserve using resting heart rate and predicted HR max using the
           equation from Tanaka et al. (2001).
           Removes negative %HRR values which are possible due to how resting HR is defined.
        """

        hr_max = 208 - 0.7 * self.age

        perc_hrr = [round(100 * (hr - self.rest_hr) / (hr_max - self.rest_hr), 2) if hr
                    is not None else None for hr in self.valid_hr]

        # A single epoch's HR can be below resting HR based on how it's defined
        # Changes any negative values to 0, maintains Nones and positive values
        # Can't figure out how to do this as a list comprehension - don't judge
        perc_hrr_final = []

        for i in perc_hrr:
            if i is not None:
                if i >= 0:
                    perc_hrr_final.append(i)
                if i < 0:
                    perc_hrr_final.append(0)
            if i is None:
                perc_hrr_final.append(None)

        return perc_hrr_final

    def hr_to_perchrr(self, hr):
        """Calculates a HR as a percent of HRR."""

        net_hr = hr - self.rest_hr
        hrr = (208 - 0.7 * self.age) - self.rest_hr

        hrr_percent = round(100 * (net_hr / hrr), 1)

        return hrr_percent

    def perchrr_to_hr(self, perc_hrr):
        """Calculates percent HRR as a raw HR (bpm)"""

        hrr = (208 - 0.7 * self.age) - self.rest_hr

        hr = round(perc_hrr * hrr / 100 + self.rest_hr, 1)

        return hr

    def calculate_intensity(self):
        """Calculates intensity category based on %HRR ranges.
           Sums values to determine total time spent in each category.

        :returns
        -intensity: epoch-by-epoch categorization by intensity. 0=sedentary, 1=light, 2=moderate, 3=vigorous
        -intensity_minutes: total minutes spent at each intensity, dictionary
        """

        # INTENSITIY DEFINITIONS
        # Sedentary = %HRR < 30, light = 30 < %HRR <= 40, moderate = 40 < %HRR <= 60, vigorous = %HRR >= 60

        intensity = []

        for hrr in self.perc_hrr:
            if hrr is None:
                intensity.append(None)

            if hrr is not None:
                if hrr < 30:
                    intensity.append(0)
                if 30 <= hrr < 40:
                    intensity.append(1)
                if 40 <= hrr < 60:
                    intensity.append(2)
                if hrr >= 60:
                    intensity.append(3)

        n_valid_epochs = len(self.valid_hr) - self.quality_report["Invalid epochs"]

        if n_valid_epochs == 0:
            n_valid_epochs = len(self.valid_hr)

        # Calculates time spent in each intensity category
        intensity_totals = {"Sedentary": intensity.count(0) / (60 / self.epoch_len),
                            "Sedentary%": round(intensity.count(0) / n_valid_epochs, 3),
                            "Light": intensity.count(1) / (60 / self.epoch_len),
                            "Light%": round(intensity.count(1) / n_valid_epochs, 3),
                            "Moderate": intensity.count(2) / (60 / self.epoch_len),
                            "Moderate%": round(intensity.count(2) / n_valid_epochs, 3),
                            "Vigorous": intensity.count(3) / (60 / self.epoch_len),
                            "Vigorous%": round(intensity.count(3) / n_valid_epochs, 3)}

        print("\n" + "HEART RATE MODEL SUMMARY")
        print("Sedentary: {} minutes ({}%)".format(intensity_totals["Sedentary"],
                                                   round(intensity_totals["Sedentary%"] * 100, 3)))

        print("Light: {} minutes ({}%)".format(intensity_totals["Light"],
                                               round(intensity_totals["Light%"] * 100, 3)))

        print("Moderate: {} minutes ({}%)".format(intensity_totals["Moderate"],
                                                  round(intensity_totals["Moderate%"] * 100, 3)))

        print("Vigorous: {} minutes ({}%)".format(intensity_totals["Vigorous"],
                                                  round(intensity_totals["Vigorous%"] * 100, 3)))

        return intensity, intensity_totals

    def plot_histogram(self):
        """Generates a histogram of heart rates over the course of the collection with a bin width of 5 bpm.
           Marks calculated average and resting HR."""

        # Data subset: only valid HRs
        valid_heartrates = [i for i in self.valid_hr if i is not None]
        avg_hr = sum(valid_heartrates) / len(valid_heartrates)

        # Bins of width 5bpm between 40 and 180 bpm
        n_bins = np.arange(40, 180, 5)

        plt.figure(figsize=(10, 7))
        plt.hist(x=valid_heartrates, weights=np.ones(len(valid_heartrates)) / len(valid_heartrates), bins=n_bins,
                 edgecolor='black', color='grey')
        plt.axvline(x=avg_hr, color='red', linestyle='dashed', label="Average HR ({} bpm)".format(round(avg_hr, 1)))
        plt.axvline(x=self.rest_hr, color='green', linestyle='dashed',
                    label='Calculated resting HR ({} bpm)'.format(round(self.rest_hr, 1)))

        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

        plt.ylabel("% of Epochs")
        plt.xlabel("HR (bpm)")
        plt.title("Heart Rate Histogram")
        plt.legend(loc='upper left')
        plt.show()

    def plot_qc_segment(self, input_index=None, template_data='filtered', plot_steps=True, plot_template=False):
        """Method that generates a random 10-minute sample of data. Overlays filtered data with quality check output.

        :argument
        -start_index: able to input desired start index. If None, randomly generated
        """

        # Generates random start index
        if input_index is not None:
            start_index = input_index
        if input_index is None:
            start_index = randint(0, len(self.filtered) - self.epoch_len * self.sample_rate)

        # Rounds random start to an index that corresponds to start of an epoch
        start_index -= start_index % (self.epoch_len * self.sample_rate)

        print("\n" + "Index {}.".format(start_index))

        # End index: one epoch
        end_index = start_index + self.epoch_len * self.sample_rate

        # Data point index converted to seconds
        seconds_seq_raw = np.arange(0, self.epoch_len * self.sample_rate) / self.sample_rate

        # Epoch's quality check
        validity_data = CheckQuality(ecg_object=self, start_index=start_index,
                                     epoch_len=self.epoch_len, template_data=template_data)

        print()
        print("Valid HR: {} (passed {}/5 conditions)".format(validity_data.rule_check_dict["Valid Period"],
                                                             validity_data.rule_check_dict["HR Valid"] +
                                                             validity_data.rule_check_dict["Max RR Interval Valid"] +
                                                             validity_data.rule_check_dict["RR Ratio Valid"] +
                                                             validity_data.rule_check_dict["Voltage Range Valid"] +
                                                             validity_data.rule_check_dict["Correlation Valid"]))

        print("-HR range ({} bpm): {}".format(validity_data.rule_check_dict["HR"],
                                              validity_data.rule_check_dict["HR Valid"]))
        print("-Max RR interval ({} sec): {}".format(validity_data.rule_check_dict["Max RR Interval"],
                                                     validity_data.rule_check_dict["Max RR Interval Valid"]))
        print("-RR ratio ({}): {}".format(validity_data.rule_check_dict["RR Ratio"],
                                          validity_data.rule_check_dict["RR Ratio Valid"]))
        print("-Voltage range ({} uV): {}".format(validity_data.rule_check_dict["Voltage Range"],
                                                  validity_data.rule_check_dict["Voltage Range Valid"]))
        print("-Correlation (r={}): {}".format(validity_data.rule_check_dict["Correlation"],
                                               validity_data.rule_check_dict["Correlation Valid"]))

        # Plot

        if plot_template:
            plt.close("all")

            fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 7))

            valid_period = "Valid" if validity_data.rule_check_dict["Valid Period"] else "Invalid"

            ax1.set_title("Participant {}: {} (index = {})".format(self.subject_id, valid_period, start_index))

            # Filtered ECG data
            ax1.plot(seconds_seq_raw, self.raw[start_index:end_index], color='black', label="Raw ECG")
            ax1.set_ylabel("Voltage")
            ax1.legend(loc='upper left')

            # Wavelet data
            ax2.plot(np.arange(0, len(validity_data.wavelet)) / self.sample_rate, validity_data.wavelet,
                     color='green', label="Wavelet")
            ax2.plot(validity_data.r_peaks / self.sample_rate,
                     [validity_data.wavelet[peak] for peak in validity_data.r_peaks],
                     linestyle="", marker="x", color='black')
            ax2.set_ylabel("Voltage")
            ax2.legend()

            for peak in validity_data.removed_peak:
                ax2.plot(np.arange(0, len(validity_data.wavelet))[peak] / self.sample_rate,
                         validity_data.wavelet[peak], marker="x", color='red')

            for i, window in enumerate(validity_data.ecg_windowed):
                ax3.plot(np.arange(0, len(window))/self.sample_rate, window, color='black')

            ax3.plot(np.arange(0, len(validity_data.average_qrs))/self.sample_rate, validity_data.average_qrs,
                     label="QRS template ({} data; r={})".format(template_data, validity_data.average_r),
                     color='red', linestyle='dashed')

            ax3.legend()
            ax3.set_ylabel("Voltage")
            ax3.set_xlabel("Seconds")

        if plot_steps:
            validity_data.plot_steps()

        return validity_data

    def calculate_nonwear(self, epoch_len=15, plot_data=True):

        # First accel check: SD and range below threshold calculations ------------------------------------------------
        accel_nw = []

        for i in np.arange(0, len(self.accel_x), self.accel_sample_rate * epoch_len):
            sd_x = np.std(x.accel_x[i:i + self.accel_sample_rate * epoch_len])
            sd_y = np.std(x.accel_y[i:i + self.accel_sample_rate * epoch_len])
            sd_z = np.std(x.accel_z[i:i + self.accel_sample_rate * epoch_len])
            axes_below_thresh = int(sd_x <= 3) + int(sd_y <= 3) + int(sd_z <= 3)

            range_x = max(x.accel_x[i:i + self.accel_sample_rate * epoch_len]) - \
                      min(self.accel_x[i:i + self.accel_sample_rate * epoch_len])
            range_y = max(x.accel_y[i:i + self.accel_sample_rate * epoch_len]) - \
                      min(self.accel_y[i:i + self.accel_sample_rate * epoch_len])
            range_z = max(x.accel_z[i:i + self.accel_sample_rate * epoch_len]) - \
                      min(self.accel_z[i:i + self.accel_sample_rate * epoch_len])

            axes_below_range = int(range_x <= 50) + int(range_y <= 50) + int(range_z <= 50)

            if axes_below_range >= 2 or axes_below_thresh >= 2:
                accel_nw.append("Nonwear")
            else:
                accel_nw.append("Wear")

        # Combines accelerometer and ECG non-wear characteristics: epoch-by-epoch -------------------------------------
        df_ecg = pd.DataFrame(list(zip(self.epoch_timestamps, self.epoch_validity,
                                       self.avg_voltage, self.svm, accel_nw)),
                              columns=["Stamp", "Validity", "VoltRange", "SVM", "AccelNW"])

        nw = []
        for epoch in df_ecg.itertuples():
            if epoch.Validity == "Invalid" and epoch.AccelNW == "Nonwear" and epoch.VoltRange <= 400:
                nw.append("Nonwear")
            else:
                nw.append("Wear")

        # 5-minute windows --------------------------------------------------------------------------------------------
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

        if plot_data:

            print("Generating plot...")

            manual_log = pd.read_excel("/Users/kyleweber/Desktop/BittiumFF_Nonwear.xlsx")
            manual_log = manual_log.loc[manual_log["ID"] == self.subject_id]

            fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 7))
            plt.suptitle(self.subject_id)
            ax1.plot(self.timestamps[::int(5 * self.sample_rate / 250)],
                     self.raw[::int(5 * self.sample_rate / 250)], color='black')
            ax1.set_ylabel("ECG Voltage")

            ax2.plot(self.timestamps[::int(10 * self.sample_rate / 250)], self.accel_x, color='dodgerblue')
            ax2.set_ylabel("Accel VM")

            ax3.plot(self.epoch_timestamps[0:min([len(self.epoch_timestamps), len(self.epoch_validity)])],
                     self.epoch_validity[0:min([len(self.epoch_timestamps), len(self.epoch_validity)])], color='black')
            ax3.fill_between(x=self.epoch_timestamps[0:min([len(self.epoch_timestamps), len(final_nw)])],
                             y1="Wear", y2=final_nw, color='grey')

            if manual_log.shape[0] >= 1:
                for row in manual_log.itertuples():
                    ax1.fill_between(x=[row.Start, row.Stop], y1=min(self.filtered[::5]), y2=max(self.filtered[::5]),
                                     color='red', alpha=.5)
                    ax2.fill_between(x=[row.Start, row.Stop], y1=min(self.accel_x), y2=max(self.accel_x),
                                     color='red', alpha=.5)

            xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")

            ax3.xaxis.set_major_formatter(xfmt)
            plt.xticks(rotation=45, fontsize=8)

        return final_nw


class DetectAllPeaks:

    def __init__(self, data=None, sample_rate=1, algorithm="wavelet"):

        self.r_peaks = None
        self.filtered = None
        self.filt_squared = None
        self.algorithm = algorithm

        self.sample_rate = sample_rate
        self.data = data

    def detect_peaks(self):
        t0 = datetime.now()
        print("\nRunning {} peak detection on entire dataset. Please wait a while...".format(self.algorithm))

        detectors = Detectors(self.sample_rate)

        if self.algorithm == "wavelet":
            self.r_peaks, self.filtered, self.filt_squared = detectors.swt_detector(unfiltered_ecg=self.data)

        if self.algorithm == "Hamilton":
            self.r_peaks, self.filtered = detectors.hamilton_detector(unfiltered_ecg=self.data)

        t1 = datetime.now()
        proc_time = round((t1-t0).seconds, 1)

        print("Complete. Took {} seconds.".format(proc_time))

    def plot_all_peaks(self, downsample_ratio=3):

        plt.plot(np.arange(len(self.data))[::downsample_ratio]/self.sample_rate/60,
                 self.data[::downsample_ratio], color='black')

        plt.plot([peak/self.sample_rate/60 for peak in self.r_peaks], [self.data[peak] for peak in self.r_peaks],
                 marker="x", color='red', linestyle="")

# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- Quality Check ----------------------------------------------
# --------------------------------------------------------------------------------------------------------------------


class CheckQuality:
    """Class method that implements the Orphanidou ECG signal quality assessment algorithm on raw ECG data.

       Orphanidou, C. et al. (2015). Signal-Quality Indices for the Electrocardiogram and Photoplethysmogram:
       Derivation and Applications to Wireless Monitoring. IEEE Journal of Biomedical and Health Informatics.
       19(3). 832-838.
    """

    def __init__(self, ecg_object, start_index, template_data='filtered', voltage_thresh=250, epoch_len=15):
        """Initialization method.

        :param
        -ecg_object: EcgData class instance created by ImportEDF script
        -random_data: runs algorithm on randomly-generated section of data; False by default.
                      Takes priority over start_index.
        -start_index: index for windowing data; 0 by default
        -epoch_len: window length in seconds over which algorithm is run; 15 seconds by default
        """

        self.voltage_thresh = voltage_thresh
        self.epoch_len = epoch_len
        self.fs = ecg_object.sample_rate
        self.start_index = start_index
        self.template_data = template_data

        self.ecg_object = ecg_object

        self.raw_data = ecg_object.raw[self.start_index:self.start_index+self.epoch_len*self.fs]
        self.filt_data = ecg_object.filtered[self.start_index:self.start_index+self.epoch_len*self.fs]
        self.wavelet = None
        self.filt_squared = None

        self.index_list = np.arange(0, len(self.raw_data), self.epoch_len*self.fs)

        self.rule_check_dict = {"Valid Period": False,
                                "HR Valid": False, "HR": None,
                                "Max RR Interval Valid": False, "Max RR Interval": None,
                                "RR Ratio Valid": False, "RR Ratio": None,
                                "Voltage Range Valid": False, "Voltage Range": None,
                                "Correlation Valid": False, "Correlation": None,
                                "Accel Counts": None}

        # prep_data parameters
        self.r_peaks = None
        self.r_peaks_index_all = None
        self.rr_sd = None
        self.removed_peak = []
        self.enough_beats = True
        self.hr = 0
        self.delta_rr = []
        self.removal_indexes = []
        self.rr_ratio = None
        self.volt_range = 0

        # apply_rules parameters
        self.valid_hr = None
        self.valid_rr = None
        self.valid_ratio = None
        self.valid_range = None
        self.valid_corr = None
        self.rules_passed = None

        # adaptive_filter parameters
        self.median_rr = None
        self.ecg_windowed = []
        self.average_qrs = None
        self.average_r = 0

        # calculate_correlation parameters
        self.beat_ppmc = []
        self.valid_period = None

        """RUNS METHODS"""
        # Peak detection and basic outcome measures
        self.prep_data()

        # Runs rules check if enough peaks found
        if self.enough_beats:
            self.adaptive_filter(template_data=self.template_data)
            self.calculate_correlation()
            self.apply_rules()

        if self.valid_period:
            self.r_peaks_index_all = [peak + start_index for peak in self.r_peaks]

    def prep_data(self):
        """Function that:
        -Initializes ecgdetector class instance
        -Runs stationary wavelet transform peak detection
            -Implements 0.1-10Hz bandpass filter
            -DB3 wavelet transformation
            -Pan-Tompkins peak detection thresholding
        -Calculates RR intervals
        -Removes first peak if it is within median RR interval / 2 from start of window
        -Calculates average HR in the window
        -Determines if there are enough beats in the window to indicate a possible valid period
        """

        # Initializes Detectors class instance with sample rate
        detectors = Detectors(self.fs)

        # Runs peak detection on raw data ----------------------------------------------------------------------------
        # Uses ecgdetectors package -> stationary wavelet transformation + Pan-Tompkins peak detection algorithm
        self.r_peaks, self.wavelet, self.filt_squared = detectors.swt_detector(unfiltered_ecg=self.filt_data)

        # Checks to see if there are enough potential peaks to correspond to correct HR range ------------------------
        # Requires number of beats in window that corresponds to ~40 bpm to continue
        # Prevents the math in the self.hr calculation from returning "valid" numbers with too few beats
        # i.e. 3 beats in 3 seconds (HR = 60bpm) but nothing detected for rest of epoch
        if len(self.r_peaks) >= np.floor(40/60*self.epoch_len):
            self.enough_beats = True

            n_beats = len(self.r_peaks)  # number of beats in window
            delta_t = (self.r_peaks[-1] - self.r_peaks[0]) / self.fs  # time between first and last beat, seconds
            self.hr = 60 * (n_beats-1) / delta_t  # average HR, bpm

        # Stops function if not enough peaks found to be a potential valid period
        # Threshold corresponds to number of beats in the window for a HR of 40 bpm
        if len(self.r_peaks) < np.floor(40/60*self.epoch_len):
            self.enough_beats = False
            self.valid_period = False
            return

        # Calculates RR intervals in seconds -------------------------------------------------------------------------
        for peak1, peak2 in zip(self.r_peaks[:], self.r_peaks[1:]):
            rr_interval = (peak2 - peak1) / self.fs
            self.delta_rr.append(rr_interval)

        # Approach 1: median RR characteristics ----------------------------------------------------------------------
        # Calculates median RR-interval in seconds
        median_rr = np.median(self.delta_rr)

        # SD of RR intervals in ms
        self.rr_sd = np.std(self.delta_rr) * 1000

        # Converts median_rr to samples
        self.median_rr = int(median_rr * self.fs)

        # Removes any peak too close to start/end of data section: affects windowing later on ------------------------
        # Peak removed if within median_rr/2 samples of start of window
        # Peak removed if within median_rr/2 samples of end of window
        for i, peak in enumerate(self.r_peaks):
            # if peak < (self.median_rr/2 + 1) or (self.epoch_len*self.fs - peak) < (self.median_rr/2 + 1):
            if peak < (self.median_rr / 2 + 1) or (self.epoch_len * self.fs - peak) < (self.median_rr / 2 + 1):
                self.removed_peak.append(self.r_peaks.pop(i))
                self.removal_indexes.append(i)

        # Removes RR intervals corresponding to
        if len(self.removal_indexes) != 0:
            self.delta_rr = [self.delta_rr[i] for i in range(len(self.r_peaks)) if i not in self.removal_indexes]

        # Calculates range of ECG voltage ----------------------------------------------------------------------------
        self.volt_range = max(self.raw_data) - min(self.raw_data)

    def adaptive_filter(self, template_data="filtered"):
        """Method that runs an adaptive filter that generates the "average" QRS template for the window of data.

        - Calculates the median RR interval
        - Generates a sub-window around each peak, +/- RR interval/2 in width
        - Deletes the final beat sub-window if it is too close to end of data window
        - Calculates the "average" QRS template for the window
        """

        # Approach 1: calculates median RR-interval in seconds  -------------------------------------------------------
        # See previous method

        # Approach 2: takes a window around each detected R-peak of width peak +/- median_rr/2 ------------------------
        for peak in self.r_peaks:
            if template_data == "raw":
                window = self.raw_data[peak - int(self.median_rr / 2):peak + int(self.median_rr / 2)]
            if template_data == "filtered":
                window = self.filt_data[peak - int(self.median_rr / 2):peak + int(self.median_rr / 2)]
            if template_data == "wavelet":
                window = self.wavelet[peak - int(self.median_rr / 2):peak + int(self.median_rr / 2)]

            self.ecg_windowed.append(window)  # Adds window to list of windows

        # Approach 3: determine average QRS template ------------------------------------------------------------------
        self.ecg_windowed = np.asarray(self.ecg_windowed)[1:]  # Converts list to np.array; omits first empty array

        # Calculates "correct" length (samples) for each window (median_rr number of datapoints)
        correct_window_len = 2*int(self.median_rr/2)

        # Removes final beat's window if its peak is less than median_rr/2 samples from end of window
        # Fixes issues when calculating average_qrs waveform
        if len(self.ecg_windowed[-1]) != correct_window_len:
            self.removed_peak.append(self.r_peaks.pop(-1))
            self.ecg_windowed = self.ecg_windowed[:-2]

        # Calculates "average" heartbeat using windows around each peak
        try:
            self.average_qrs = np.mean(self.ecg_windowed, axis=0)
        except ValueError:
            print("Failed to calculate mean QRS template.")

    def calculate_correlation(self):
        """Method that runs a correlation analysis for each beat and the average QRS template.

        - Runs a Pearson correlation between each beat and the QRS template
        - Calculates the average individual beat Pearson correlation value
        - The period is deemed valid if the average correlation is >= 0.66, invalid is < 0.66
        """

        # Calculates correlation between each beat window and the average beat window --------------------------------
        for beat in self.ecg_windowed:
            r = stats.pearsonr(x=beat, y=self.average_qrs)
            self.beat_ppmc.append(abs(r[0]))

        self.average_r = float(np.mean(self.beat_ppmc))
        self.average_r = round(self.average_r, 3)

    def apply_rules(self):
        """First stage of algorithm. Checks data against three rules to determine if the window is potentially valid.
        -Rule 1: HR needs to be between 40 and 180bpm
        -Rule 2: no RR interval can be more than 3 seconds
        -Rule 3: the ratio of the longest to shortest RR interval is less than 2.2
        -Rule 4: the amplitude range of the raw ECG voltage must exceed n microV (approximate range for non-wear)
        -Rule 5: the average correlation coefficient between each beat and the "average" beat must exceed 0.66
        -Verdict: all rules need to be passed
        """

        # Rule 1: "The HR extrapolated from the sample must be between 40 and 180 bpm" -------------------------------
        if 40 <= self.hr <= 180:
            self.valid_hr = True
        else:
            self.valid_hr = False

        # Rule 2: "the maximum acceptable gap between successive R-peaks is 3s ---------------------------------------
        for rr_interval in self.delta_rr:
            if rr_interval < 3:
                self.valid_rr = True

            if rr_interval >= 3:
                self.valid_rr = False
                break

        # Rule 3: "the ratio of the maximum beat-to-beat interval to the minimum beat-to-beat interval... ------------
        # should be less than 2.5"
        self.rr_ratio = max(self.delta_rr) / min(self.delta_rr)

        if self.rr_ratio >= 2.5:
            self.valid_ratio = False

        if self.rr_ratio < 2.5:
            self.valid_ratio = True

        # Rule 4: the range of the raw ECG signal needs to be >= 250 microV ------------------------------------------
        if self.volt_range <= self.voltage_thresh:
            self.valid_range = False

        if self.volt_range > self.voltage_thresh:
            self.valid_range = True

        # Rule 5: Determines if average R value is above threshold of 0.66 -------------------------------------------
        if self.average_r >= 0.66:
            self.valid_corr = True

        if self.average_r < 0.66:
            self.valid_corr = False

        # FINAL VERDICT: valid period if all rules are passed --------------------------------------------------------
        if self.valid_hr and self.valid_rr and self.valid_ratio and self.valid_range and self.valid_corr:
            self.valid_period = True
        else:
            self.valid_period = False

        self.rule_check_dict = {"Valid Period": self.valid_period,
                                "HR Valid": self.valid_hr, "HR": round(self.hr, 1),
                                "Max RR Interval Valid": self.valid_rr, "Max RR Interval": round(max(self.delta_rr), 1),
                                "RR Ratio Valid": self.valid_ratio, "RR Ratio": round(self.rr_ratio, 1),
                                "Voltage Range Valid": self.valid_range, "Voltage Range": round(self.volt_range, 1),
                                "Correlation Valid": self.valid_corr, "Correlation": self.average_r,
                                "Accel Flatline": None}

        if self.ecg_object.load_accel:
            accel_start = int(self.start_index / (self.ecg_object.sample_rate / self.ecg_object.accel_sample_rate))
            accel_end = accel_start + self.ecg_object.accel_sample_rate * self.epoch_len

            svm = sum(self.ecg_object.accel_vm[accel_start:accel_end])
            self.rule_check_dict["Accel Counts"] = round(svm, 2)

            flatline = True if max(self.ecg_object.accel_vm[accel_start:accel_end]) - \
                               min(self.ecg_object.accel_vm[accel_start:accel_end]) <= .05 else False
            self.rule_check_dict["Accel Flatline"] = flatline

            sd = np.std(self.ecg_object.accel_vm[accel_start:accel_end])
            self.rule_check_dict["Accel SD"] = sd

    def plot_steps(self):

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex="col", figsize=(10, 6))
        plt.suptitle("ECG Quality Check Processing Steps "
                     "({} period)".format("Valid" if self.valid_period else "Invalid"))

        # Raw ECG
        ax1.plot(np.arange(len(self.raw_data))/self.ecg_object.sample_rate, self.raw_data,
                 color='red', label="Raw")
        ax1.set_ylabel("Voltage")
        ax1.set_xlim(-.5, self.epoch_len * 1.25)
        ax1.legend()

        # Filtered ECG
        ax2.plot(np.arange(len(self.filt_data))/self.ecg_object.sample_rate, self.filt_data,
                 color='blue', label="Filtered")
        ax2.set_ylabel("Voltage")
        ax2.legend()

        # Wavelet ECG
        ax3.plot(np.arange(len(self.wavelet)) / self.ecg_object.sample_rate, self.wavelet,
                 color='green', label="Wavelet")
        ax3.set_ylabel("Voltage")
        ax3.legend()

        # Wavelet squared + filtered
        ax4.plot(np.arange(len(self.filt_squared))/self.ecg_object.sample_rate, self.filt_squared,
                 color='dodgerblue', label="Squared")
        ax4.plot([np.arange(len(self.filt_squared))[i]/self.ecg_object.sample_rate for i in self.r_peaks],
                 [self.filt_squared[i] for i in self.r_peaks], linestyle="", marker="x", color='black')
        ax4.fill_between(x=[0, self.median_rr / 2 / self.ecg_object.sample_rate],
                         y1=min(self.filt_squared), y2=max(self.filt_squared), color='grey', alpha=.5)
        ax4.fill_between(x=[self.epoch_len - self.median_rr / 2 / self.ecg_object.sample_rate, self.epoch_len],
                         y1=min(self.filt_squared), y2=max(self.filt_squared), color='grey', alpha=.5,
                         label="Peak removed")
        ax4.set_ylabel("Voltage")
        ax4.set_xlabel("Time (s)")
        ax4.legend()

# qc = CheckQuality(ecg_object=x.ecg, start_index=8887500, template_data="raw")
# x.ecg.plot_qc_segment(input_index=None, template_data='wavelet')

# Figure out what to do with first beats that get missed
    # Thought: run peak detection on entire file (separate from quality check), include all peak indexes, and then
    # remove ones that fall in invalid regions


"""
x = ECG(subject_id=3035, filepath="/Users/kyleweber/Desktop/Data/OND07/EDF/OND07_WTL_3035_01_BF.edf",
        output_dir=None, processed_folder=None,
        processed_file=None,
        age=0, start_offset=0, end_offset=0,
        rest_hr_window=60, n_epochs_rest=10,
        epoch_len=15, load_accel=True,
        filter_data=False, low_f=1, high_f=30, f_type="bandpass",
        load_raw=True, from_processed=False)


fig, (ax1, ax2) = plt.subplots(2, sharex='col')
ax1.plot(x.timestamps[::5], x.raw[::5], color='red')
ax2.plot(x.timestamps[::10], x.accel_x, color='black')
xfmt = mdates.DateFormatter("%Y/%m/%d\n%H:%M:%S")
ax2.xaxis.set_major_formatter(xfmt)
plt.xticks(rotation=45, fontsize=8)

x.nonwear = x.calculate_nonwear(epoch_len=15, plot_data=True)
"""