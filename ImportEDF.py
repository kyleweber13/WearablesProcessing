import pyedflib
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import Filtering
import matplotlib.dates as mdates


class GENEActiv:

    def __init__(self, filepath, load_raw, start_offset=0, end_offset=0):

        self.filepath = filepath
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.load_raw = load_raw

        # Accelerometer data
        self.x = None
        self.y = None
        self.z = None
        self.vm = None  # Vector Magnitudes
        self.timestamps = None

        # Details
        self.sample_rate = 75  # default value
        self.starttime = None
        self.file_dur = None

        # IMPORTS GENEActiv FILE
        if self.load_raw:
            self.import_file()

    def import_file(self):

        t0 = datetime.now()  # Gets current time

        print("Importing {}...".format(self.filepath))

        # READS IN ACCELEROMETER DATA ================================================================================
        file = pyedflib.EdfReader(self.filepath)

        if self.end_offset != 0:
            print("Importing file from index {} to {}...".format(self.start_offset, self.end_offset))

            self.x = file.readSignal(chn=0, start=self.start_offset, n=self.end_offset)
            self.y = file.readSignal(chn=1, start=self.start_offset, n=self.end_offset)
            self.z = file.readSignal(chn=2, start=self.start_offset, n=self.end_offset)

        if self.end_offset == 0:
            print("Importing file from index {} to the end...".format(self.start_offset))

            self.x = file.readSignal(chn=0, start=self.start_offset)
            self.y = file.readSignal(chn=1, start=self.start_offset)
            self.z = file.readSignal(chn=2, start=self.start_offset)

        # Calculates gravity-subtracted vector magnitude
        # Negative values become zero
        self.vm = np.sqrt(np.square(np.array([self.x, self.y, self.z])).sum(axis=0)) - 1
        self.vm[self.vm < 0] = 0

        self.sample_rate = file.getSampleFrequencies()[1]  # sample rate
        self.starttime = file.getStartdatetime() + timedelta(seconds=self.start_offset/self.sample_rate)
        self.file_dur = round(file.getFileDuration() / 3600, 3)  # Seconds --> hours

        # TIMESTAMP GENERATION ========================================================================================
        t0_stamp = datetime.now()

        print("\n" + "Creating timestamps...")

        end_time = self.starttime + timedelta(seconds=len(self.x) / self.sample_rate)
        self.timestamps = np.asarray(pd.date_range(start=self.starttime, end=end_time, periods=len(self.x)))

        t1_stamp = datetime.now()
        stamp_time = (t1_stamp-t0_stamp).seconds
        print("Complete ({} seconds).".format(round(stamp_time, 2)))

        t1 = datetime.now()
        proc_time = (t1 - t0).seconds
        print("Import complete ({} seconds).".format(round(proc_time, 2)))


class GENEActivTemperature:

    def __init__(self, filepath, from_processed=False, start_offset=0, end_offset=0):

        self.filepath = filepath
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.from_processed = from_processed

        # Accelerometer data
        self.temperature = None
        self.timestamps = None

        # Details
        self.sample_rate = 0
        self.starttime = None
        self.file_dur = None

        # IMPORTS GENEACTIV FILE
        if not self.from_processed and self.filepath is not None:
            self.import_file()

    def import_file(self):

        t0 = datetime.now()  # Gets current time

        print("Importing {}...".format(self.filepath))

        # READS IN ACCELEROMETER DATA ================================================================================
        file = pyedflib.EdfReader(self.filepath)

        self.temperature = file.readSignal(chn=0)

        self.sample_rate = file.getSampleFrequencies()[0]  # sample rate

        if self.sample_rate == 0:
            self.sample_rate = 1
        self.starttime = file.getStartdatetime() + timedelta(seconds=self.start_offset/self.sample_rate)
        self.file_dur = round(file.getFileDuration() / 3600, 3)  # Seconds --> hours

        # TIMESTAMP GENERATION ========================================================================================
        t0_stamp = datetime.now()

        print("\n" + "Creating timestamps...")

        end_time = self.starttime + timedelta(seconds=len(self.temperature) * 4 / self.sample_rate)
        self.timestamps = np.asarray(pd.date_range(start=self.starttime, end=end_time, periods=len(self.temperature)))

        t1_stamp = datetime.now()
        stamp_time = (t1_stamp-t0_stamp).seconds
        print("Complete ({} seconds).".format(round(stamp_time, 2)))

        t1 = datetime.now()
        proc_time = (t1 - t0).seconds
        print("Import complete ({} seconds).".format(round(proc_time, 2)))


class GENEActivLight:

    def __init__(self, filepath, from_processed=False, start_offset=0, end_offset=0):

        self.filepath = filepath
        self.from_processed = from_processed
        self.start_offset = start_offset
        self.end_offset = end_offset

        self.light = None
        self.light_avg = None
        self.timestamps = None
        self.epoch_timestamps = None
        self.sample_rate = 1

        self.import_file()

    def import_file(self):

        t0 = datetime.now()  # Gets current time

        print("Importing {}...".format(self.filepath))

        # READS IN ACCELEROMETER DATA ================================================================================
        file = pyedflib.EdfReader(self.filepath)

        self.light = file.readSignal(chn=0)

        self.sample_rate = file.getSampleFrequencies()[0]  # sample rate

        if self.sample_rate == 0:
            self.sample_rate = 1
        self.starttime = file.getStartdatetime() + timedelta(seconds=self.start_offset/self.sample_rate)
        self.file_dur = round(file.getFileDuration() / 3600, 3)  # Seconds --> hours

        # TIMESTAMP GENERATION ========================================================================================
        t0_stamp = datetime.now()

        print("\n" + "Creating timestamps...")

        end_time = self.starttime + timedelta(seconds=len(self.light) / self.sample_rate)
        self.timestamps = np.asarray(pd.date_range(start=self.starttime, end=end_time, periods=len(self.light)))

        t1_stamp = datetime.now()
        stamp_time = (t1_stamp-t0_stamp).seconds
        print("Complete ({} seconds).".format(round(stamp_time, 2)))

        # DATA EPOCHING ===============================================================================================
        self.light_avg = [sum(self.light[i:i+15*self.sample_rate])/(15 * self.sample_rate) for
                          i in range(0, len(self.light), 15 * self.sample_rate)]
        self.epoch_timestamps = self.timestamps[::self.sample_rate * 15]

        t1 = datetime.now()
        proc_time = (t1 - t0).seconds
        print("Import complete ({} seconds).".format(round(proc_time, 2)))


class Bittium:

    def __init__(self, filepath, start_offset=0, end_offset=0, epoch_len=15, load_accel=False,
                 low_f=1, high_f=30, f_type="bandpass"):

        self.filepath = filepath
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.epoch_len = epoch_len
        self.load_accel = load_accel

        # Filter details
        self.low_f = low_f
        self.high_f = high_f
        self.f_type = f_type

        # ECG data
        self.raw = None
        self.filtered = None
        self.timestamps = None
        self.epoch_timestamps = None

        # Accel data
        self.accel_sample_rate = 1  # default value
        self.x = None
        self.y = None
        self.z = None
        self.vm = None  # Vector Magnitudes

        # Details
        self.sample_rate = None
        self.starttime = None
        self.file_dur = None

        # RUNS METHODS
        self.import_file()

    def import_file(self):
        """Method that loads voltage channel, sample rate, starttime, and file duration.
        Creates timestamp for each data point."""

        t0 = datetime.now()

        print("\n" + "Importing {}...".format(self.filepath))

        file = pyedflib.EdfReader(self.filepath)

        self.sample_rate = file.getSampleFrequencies()[0]
        self.accel_sample_rate = file.getSampleFrequencies()[1]

        # READS IN ECG DATA ===========================================================================================
        if self.end_offset == 0:
            print("Importing file from index {} to the end...".format(self.start_offset))
            self.raw = file.readSignal(chn=0, start=self.start_offset)

            if self.load_accel:
                self.x = file.readSignal(chn=1, start=int(self.start_offset *
                                                          self.accel_sample_rate / self.sample_rate))
                self.y = file.readSignal(chn=2, start=int(self.start_offset *
                                                          self.accel_sample_rate / self.sample_rate))
                self.z = file.readSignal(chn=3, start=int(self.start_offset *
                                                          self.accel_sample_rate / self.sample_rate))

        if self.end_offset != 0:
            print("Importing file from index {} to {}...".format(self.start_offset,
                                                                 self.start_offset + self.end_offset))
            self.raw = file.readSignal(chn=0, start=self.start_offset, n=self.end_offset)

            if self.load_accel:
                self.x = file.readSignal(chn=1,
                                         start=int(self.start_offset * self.accel_sample_rate / self.sample_rate),
                                         n=int(self.end_offset * self.accel_sample_rate / self.sample_rate))

                self.y = file.readSignal(chn=2,
                                         start=int(self.start_offset * self.accel_sample_rate / self.sample_rate),
                                         n=int(self.end_offset * self.accel_sample_rate / self.sample_rate))

                self.z = file.readSignal(chn=3,
                                         start=int(self.start_offset * self.accel_sample_rate / self.sample_rate),
                                         n=int(self.end_offset * self.accel_sample_rate / self.sample_rate))

        # Calculates gravity-subtracted vector magnitude. Converts from mg to G
        # Negative values become zero
        if self.load_accel:
            self.vm = (np.sqrt(np.square(np.array([self.x, self.y, self.z])).sum(axis=0)) - 1000) / 1000
            self.vm[self.vm < 0] = 0

        print("ECG data import complete.")

        self.starttime = file.getStartdatetime() + timedelta(seconds=self.start_offset/self.sample_rate)
        self.file_dur = round(file.getFileDuration() / 3600, 3)

        # Data filtering
        self.filtered = Filtering.filter_signal(data=self.raw, low_f=self.low_f, high_f=self.high_f,
                                                filter_type=self.f_type, sample_f=self.sample_rate, filter_order=3)

        # TIMESTAMP GENERATION ========================================================================================
        t0_stamp = datetime.now()

        print("\n" + "Creating timestamps...")

        # Timestamps
        end_time = self.starttime + timedelta(seconds=len(self.raw)/self.sample_rate)
        self.timestamps = np.asarray(pd.date_range(start=self.starttime, end=end_time, periods=len(self.raw)))
        self.epoch_timestamps = self.timestamps[::self.epoch_len * self.sample_rate]

        t1_stamp = datetime.now()
        stamp_time = (t1_stamp - t0_stamp).seconds
        print("Complete ({} seconds).".format(round(stamp_time, 2)))

        t1 = datetime.now()
        proc_time = (t1 - t0).seconds
        print("\n" + "Import complete ({} seconds).".format(round(proc_time, 2)))


def check_file(filepath, print_summary=True):
    """Calculates file duration with start and end times. Prints results to console."""

    if filepath is None:
        return None, None, None

    edf_file = pyedflib.EdfReader(filepath)

    duration = edf_file.getFileDuration()
    start_time = edf_file.getStartdatetime()
    end_time = start_time + timedelta(seconds=edf_file.getFileDuration())

    if print_summary:
        print("\n", filepath)
        print("-Sample rate: {}Hz".format(edf_file.getSampleFrequency(0)))
        print("-Start time: ", start_time)
        print("-End time:", end_time)
        print("-Duration: {} hours".format(round(duration/3600, 2)))

    return start_time, end_time, edf_file.getSampleFrequency(0), duration


def import_subject(id):
    accel = GENEActiv(filepath="/Volumes/Kyle's External HD/OND06 LWrist Data/Accelerometer/OND06_SBH_{}_"
                               "GNAC_ACCELEROMETER_LWrist.edf".format(id),
                      load_raw=True, start_offset=0, end_offset=0)
    light = GENEActivLight("/Volumes/Kyle's External HD/OND06 LWrist Data/Light/OND06_SBH_{}_"
                           "GNAC_LIGHT_LWrist.edf".format(id))
    temp = GENEActivTemperature("/Volumes/Kyle's External HD/OND06 LWrist Data/Temperature/"
                                "OND06_SBH_{}_GNAC_TEMPERATURE_LWrist.edf".format(id))

    nw = pd.read_csv("/Users/kyleweber/Desktop/Data/OND06/ReMiNDDNonWearReformatted_GAgoldstandarddataset_6Nov2020.csv")
    nw = nw.loc[nw["ID"] == id]
    nw = nw.loc[nw["location"] == "LW"]
    nw["start_time"] = pd.to_datetime(nw["start_time"])
    nw["end_time"] = pd.to_datetime(nw["end_time"])

    accel.svm = [sum(accel.vm[i:i+accel.sample_rate*15]) for i in range(0, len(accel.vm), 15 * accel.sample_rate)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 7))
    plt.suptitle(id)
    plt.subplots_adjust(bottom=.12)
    ax1.plot(accel.timestamps[::accel.sample_rate * 15], accel.svm, color='black')
    ax1.set_ylabel("Counts")

    for row in nw.itertuples():
        ax1.fill_between(x=[row.start_time, row.end_time], y1=0, y2=max(accel.svm), color='red', alpha=.5)

    ax2.plot(temp.timestamps, temp.temperature, color='red')
    ax2.set_ylabel("Degrees")
    ax3.plot(light.epoch_timestamps, light.light_avg, color='orange')
    ax3.set_ylabel("Avg Lux")

    xfmt = mdates.DateFormatter("%Y-%m-%d %H:%M:%S")
    ax3.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=45, fontsize=6)

    return accel, light, temp, nw


def check_log(id):

    df = pd.read_excel('/Users/kyleweber/Desktop/Data/OND06/OND06_LWrist_VisualSleepDetection.xlsx',
                       usecols=["Subject", "Sleep", "Wake"])
    df = df.loc[df["Subject"]==id]

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='col', figsize=(10, 7))
    plt.suptitle(id)
    plt.subplots_adjust(bottom=.12)
    ax1.plot(accel.timestamps[::accel.sample_rate * 15], accel.svm, color='black')
    ax1.set_ylabel("Counts")
    ax2.plot(temp.timestamps, temp.temperature, color='red')
    ax2.set_ylabel("Degrees")
    ax3.plot(light.epoch_timestamps, light.light_avg, color='orange')
    ax3.set_ylabel("Avg Lux")

    xfmt = mdates.DateFormatter("%Y-%m-%d %H:%M:%S")
    ax3.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=45, fontsize=6)

    for row in df.itertuples():
        ax1.fill_between(x=[row.Sleep, row.Wake], y1=0, y2=max(accel.svm), color='green', alpha=.5)
        ax2.fill_between(x=[row.Sleep, row.Wake], y1=min(temp.temperature), y2=max(temp.temperature),
                         color='green', alpha=.5)
        ax3.fill_between(x=[row.Sleep, row.Wake], y1=0, y2=max(light.light_avg), color='green', alpha=.5)


# accel, light, temp, nw = import_subject(7595)
# check_log(7595)

