import pandas as pd
import math
import numpy as np
import scipy.fft
import matplotlib.pyplot as plt


class CircadianRhythm:

    def __init__(self, subject_obj=None):

        self.subj_obj = subject_obj
        self.interday_stability = None
        self.intraday_variability = None

        if subject_obj.epoch_len != 5:
            self.df = self.recalculate_epochs()

        if subject_obj.epoch_len == 5:
            self.df = pd.DataFrame(list(zip(self.subj_obj.wrist.epoch.timestamps,
                                            self.subj_obj.wrist.epoch.svm)), columns=["Timestamp", "SVM"])

    def recalculate_epochs(self):

        print("\nRecalculating from {}-second epochs to 5-second epochs...".format(self.subj_obj.epoch_len))

        timestamps = self.subj_obj.wrist.raw.timestamps[::self.subj_obj.epoch_len *
                                                          self.subj_obj.wrist.raw.sample_rate]

        raw_data = self.subj_obj.wrist.raw
        epoch_len = self.subj_obj.epoch_len

        # Calculates gravity-subtracted vector magnitude
        raw_data.vm = [round(abs(math.sqrt(math.pow(raw_data.x[i], 2) + math.pow(raw_data.y[i], 2) +
                                           math.pow(raw_data.z[i], 2)) - 1), 5) for i in range(len(raw_data.x))]

        # Calculates activity counts
        svm = []
        for i in range(0, len(raw_data.vm), int(raw_data.sample_rate * epoch_len)):

            if i + epoch_len * raw_data.sample_rate > len(raw_data.vm):
                break

            vm_sum = sum(raw_data.vm[i:i + epoch_len * raw_data.sample_rate])

            # Bug handling: when we combine multiple EDF files they are zero-padded
            # When vector magnitude is calculated, it is 1
            # Any epoch where the values were all the epoch length * sampling rate (i.e. a VM of 1 for each data point)
            # becomes 0
            if vm_sum == epoch_len * raw_data.sample_rate:
                vm_sum = 0

            svm.append(round(vm_sum, 5))

        df = pd.DataFrame(list(zip(timestamps, svm)), columns=["Timestamp", "SVM"])

        print("Complete.")

        return df

    def calculate_interday(self):

        print("\nCalculating interday stability...")

        # Gets value for which hour of day
        hours = []
        for row in self.df.itertuples():
            h = str(row.Timestamp.hour)

            hours.append(h)

        self.df["Hour"] = hours

        # Calculates mean SVM by hour of day
        mean_svms = c.df.groupby("Hour")["SVM"].describe()["mean"]

        # Calculates overall SVM mean value
        mean_svm = c.df["SVM"].mean()

        # Equation from Suibkitwanchai et al. 2020 paper
        numerator = sum([(hourly - mean_svm) ** 2 / 24 for hourly in mean_svms])
        denominator = sum([(value - mean_svm) ** 2 / c.df.shape[0] for value in c.df["SVM"]])

        interday_stability = round(numerator / denominator, 4)

        print("-Interday stability = {}".format(interday_stability))

        return interday_stability

    def calculate_fft(self):

        f = scipy.fft.fft([i for i in self.df["SVM"]])

        xf = np.linspace(0.0, 1.0 / (2.0 * (1 / 0.2)), self.df.shape[0] // 2)

        plt.plot(xf, 2.0 / self.df.shape[0] / 2 * np.abs(f[0:self.df.shape[0] // 2]), color='black')
        plt.axvline(x=1/86400, color='green', label="1/24H")
        plt.axvline(x=1/86400*2, color='green', linestyle='dashed', label="1/12H")
        plt.axvline(x=1/86400*3, color='green', linestyle='dashed', label="1/8H")
        plt.axvline(x=1/86400*4, color='green', linestyle='dashed', label="1/6H")
        plt.axvline(x=1/86400*5, color='green', linestyle='dashed', label="1/3H")

        plt.legend()
        plt.xlim(0, 1/86400*6)


c = CircadianRhythm(x)
c.interday = c.calculate_interday()
