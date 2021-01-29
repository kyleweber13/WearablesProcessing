import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import os


class Sleep:
    """Imports a participant's sleep log data from .csv. Creates a list corresponding to epochs where participant was
       awake vs. napping vs. asleep (overnight). Calculates total time spent in each state. Plots sleep periods
       on ankle/wrist/HR data, if available.

    :argument
    -subject_object: object of class Subject
    -file_loc: folder that contains sleep logs
    """

    def __init__(self, subject_object):

        print()
        print("========================================= SLEEP LOG DATA ===========================================")

        self.file_loc = subject_object.sleeplog_file
        self.subject_object = subject_object
        self.subject_id = self.subject_object.subject_id

        self.epoch_timestamps = None

        self.log = None
        self.data = None
        self.status = None
        self.report = {"SleepDuration": 0, "Sleep%": 0,
                       "OvernightSleepDuration": 0, "OvernightSleepDurations": 0,
                       "OvernightSleep%": 0, "AvgSleepDuration": 0,
                       "NapDuration": 0, "NapDurations": 0, "Nap%": 0, "AvgNapDuration": 0}

        # RUNS METHODS ===============================================================================================
        self.set_timestamps()
        self.log = self.import_sleeplog()

        if self.file_loc is not None and os.path.exists(self.file_loc):
            self.data, self.epoch_indexes = self.format_sleeplog()

            # Determines which epochs were asleep
            self.status = self.mark_sleep_epochs()

            # Sleep report
            self.report = self.generate_sleep_report()

    def set_timestamps(self):

        # MODEL DATA =================================================================================================
        # Sets length of data (number of epochs) and timestamps based on any data that is available
        try:
            self.epoch_timestamps = self.subject_object.ankle.epoch.timestamps
        except AttributeError:
            try:
                self.epoch_timestamps = self.subject_object.wrist.epoch.timestamps
            except AttributeError:
                self.epoch_timestamps = self.subject_object.ecg.epoch_timestamps

    def import_sleeplog(self):
        """Imports sleep log from .csv. Only keeps values associated with Subject. Returns as ndarray.
           Column names: SUBJECT, DATE, TIME_OUT_BED, NAP_START, NAP_END, TIME_IN_BED
        """

        if self.file_loc is not None and os.path.exists(self.file_loc):

            # Imports sleep log data
            if "csv" in self.file_loc:
                sleep_log = pd.read_csv(self.file_loc)
            if "xlsx" in self.file_loc:
                sleep_log = pd.read_excel(self.file_loc)

            try:
                subj_log = sleep_log.loc[sleep_log["SUBJECT"] == self.subject_object.wrist.filename.split("_0")[0]]
            except AttributeError:
                try:
                    subj_log = sleep_log.loc[sleep_log["SUBJECT"] == self.subject_object.ankle.filename.split("_0")[0]]
                except AttributeError:
                    subj_log = sleep_log.loc[sleep_log["SUBJECT"] == self.subject_object.ecg.filename.split("_0")[0]]

            return subj_log

        if self.file_loc is None or not os.path.exists(self.file_loc):
            self.status = np.zeros(self.subject_object.data_len)  # Pretends participant did not sleep

    def format_sleeplog(self):

        all_data = []
        all_epoch_indexes = []

        for day in range(self.log.shape[0]):
            day_data = []
            epoch_indexes = []

            row_data = self.log.iloc[day]

            date = datetime.strptime(row_data["DATE"], "%Y%b%d").date()

            for colname in ["TIME_WAKE", "NAP_START", "NAP_END", "TIME_IN_BED"]:
                try:
                    # Combines date + time
                    datestamp = datetime.strptime(str(date) + " " + str(row_data[colname]), "%Y-%m-%d %H:%M")

                    # Extracts hour of day (integer)
                    hour_of_day = datetime.strptime(str(row_data[colname]), "%H:%M").hour

                    # Changes date to next day if went to bed after midnight and before 6am
                    if colname == "TIME_IN_BED" and hour_of_day < 6:
                        datestamp += timedelta(days=1)

                    # Epoch index relative to start of collection
                    epoch_index = (datestamp - self.subject_object.start_timestamp).total_seconds() / \
                                  self.subject_object.epoch_len

                except ValueError:
                    datestamp = "N/A"
                    epoch_index = "N/A"

                day_data.append(datestamp)
                epoch_indexes.append(epoch_index)

            all_data.append(day_data)
            all_epoch_indexes.append(epoch_indexes)

        sleep_df = pd.DataFrame(all_data, columns=["TIME_WAKE", "NAP_START", "NAP_END", "TIME_SLEEP"])
        index_df = pd.DataFrame(all_epoch_indexes, columns=["TIME_WAKE", "NAP_START", "NAP_END", "TIME_SLEEP"])

        return sleep_df, index_df

    def mark_sleep_epochs(self):
        """Creates a list of len(epoch_timestamps) where awake is coded as 0, naps coded as 1, and
           overnight sleep coded as 2"""

        print("\nMarking epochs as asleep or awake...")

        # Creates list of 0s corresponding to each epoch
        epoch_list = np.zeros(self.subject_object.data_len + 1)

        # Converts sleeplog timestamps to epoch index relative to start of collection ---------------------------------
        overnight_indexes = []
        nap_indexes = []
        for row in range(self.epoch_indexes.shape[0]):
            # Overnight sleep
            for val in [i for i in self.epoch_indexes.iloc[row][["TIME_WAKE", "TIME_SLEEP"]].values]:
                if val != "N/A":
                    overnight_indexes.append(int(val))
            # Nap times
            for val in [i for i in self.epoch_indexes.iloc[row][["NAP_START", "NAP_END"]].values]:
                if val != "N/A":
                    nap_indexes.append(int(val))

        # Sets sections of epoch_list to correct values ---------------------------------------------------------------
        for start, stop in zip(overnight_indexes[::2], overnight_indexes[1::2]):
            epoch_list[start:stop] = 2

        for start, stop in zip(nap_indexes[::2], nap_indexes[1::2]):
            epoch_list[start:stop] = 1

        print("Done.")

        return epoch_list

    def generate_sleep_report(self):
        """Generates summary sleep measures in minutes.
        Column names: SUBJECT, DATE, TIME_OUT_BED, NAP_START, NAP_END, TIME_IN_BED
        """

        sleep_durations = ["N/A" for i in range(self.data.shape[0])]
        nap_durations = ["N/A" for i in range(self.data.shape[0])]

        epoch_to_mins = 60 / self.subject_object.epoch_len

        for row in range(self.data.shape[0] - 1):
            asleep = self.data.iloc[row]
            awake = self.data.iloc[row + 1]

            # Overnight sleep
            if asleep["TIME_SLEEP"] != "N/A" and awake["TIME_WAKE"] != "N/A":
                sleep_durations[row] = (round((awake["TIME_WAKE"] - asleep["TIME_SLEEP"]).seconds / 60, 2))

            # Naps
            if asleep["NAP_START"] != "N/A" and asleep["NAP_END"] != "N/A":
                nap_durations[row] = (round((asleep["NAP_END"] - asleep["NAP_START"]).seconds / 60, 2))

        report = {"SleepDuration": np.sum(self.status > 0) / epoch_to_mins,
                  "Sleep%": round(100 * np.sum(self.status > 0) / len(self.epoch_timestamps), 1),
                  "OvernightSleepDuration": np.sum(self.status == 2) / epoch_to_mins,
                  "OvernightSleepDurations": sleep_durations,
                  "OvernightSleep%": round(100 * np.sum(self.status == 2) / len(self.epoch_timestamps), 1),
                  "AvgSleepDuration": round(sum([i for i in sleep_durations if i != "N/A"]) /
                                            len([i for i in sleep_durations if i != "N/A"]), 1)
                  if len([i for i in sleep_durations if i != "N/A"]) > 0 else 0,
                  "NapDuration": np.sum(self.status == 1) / epoch_to_mins,
                  "NapDurations": [i for i in nap_durations if i != "N/A"],
                  "Nap%": round(100 * np.sum(self.status == 1) / len(self.epoch_timestamps), 1),
                  "AvgNapDuration": round(sum([i for i in nap_durations if i != "N/A"]) /
                                          len([i for i in nap_durations if i != "N/A"]), 1) if
                  len([j for j in nap_durations if j != "N/A"]) > 0 else 0}

        print("\n" + "SLEEP REPORT")

        print("-Total time asleep: {} minutes ({}%)".format(report["SleepDuration"], report["Sleep%"]))

        print("\n" + "-Total overnight sleep: {} minutes ({}%)".format(report["OvernightSleepDuration"],
                                                                       report["OvernightSleep%"]))
        print("-Overnight sleep durations: {} minutes".format(report["OvernightSleepDurations"][:-1]))
        print("-Average overnight sleep duration: {} minutes".format(report["AvgSleepDuration"]))

        print("\n" + "-Total napping time: {} minutes ({}%)".format(report["NapDuration"], report["Nap%"]))
        print("-Nap durations: {} minutes".format(report["NapDurations"]))
        print("-Average nap duration: {} minutes".format(report["AvgNapDuration"]))

        # Updates values data df
        self.data["NIGHT_DURATION"] = sleep_durations
        self.data["NAP_DURATION"] = nap_durations

        return report
