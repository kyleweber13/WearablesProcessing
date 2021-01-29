import os
import numpy as np
import pandas as pd


class NonwearLog:

    def __init__(self, subject_object):

        print("")
        print("=================================== ACCELEROMETER NONWEAR DATA ======================================")

        self.subject_object = subject_object
        self.subject_id = subject_object.subject_id
        self.epoch_timestamps = None

        self.file_loc = subject_object.nonwear_file
        self.status = []
        self.nonwear_log = None
        self.nonwear_dict = {"Minutes": 0, "Number of Removals": 0, "Average Duration (Mins)": 0, "Percent": 0}

        # =============================================== RUNS METHODS ================================================
        self.prep_data()
        self.import_nonwearlog()
        self.mark_nonwear_epochs()

    def prep_data(self):

        # Sets epoched timestamps based on available data
        if self.subject_object.load_wrist:
            try:
                self.epoch_timestamps = self.subject_object.wrist.epoch.timestamps
            except (AttributeError, TypeError):
                self.epoch_timestamps = self.subject_object.ankle.epoch.timestamps

        try:
            self.epoch_timestamps = self.subject_object.ankle.epoch.timestamps
        except AttributeError:
            try:
                self.epoch_timestamps = self.subject_object.wrist.epoch.timestamps
            except AttributeError:
                self.epoch_timestamps = self.subject_object.ecg.epoch_timestamps

    def import_nonwearlog(self):

        if self.file_loc is not None and os.path.exists(self.file_loc):

            # Imports sleep log data
            if "xlsx" in self.file_loc:
                nonwear_log = pd.read_excel(io=self.file_loc, columns=["ID", "DEVICE OFF", "DEVICE ON"])

            if "csv" in self.file_loc:
                nonwear_log = pd.read_csv(io=self.file_loc, columns=["ID", "DEVICE OFF", "DEVICE ON"])

            nonwear_log = nonwear_log.loc[nonwear_log["ID"] == self.subject_id]

            nonwear_log["DEVICE OFF"] = pd.to_datetime(nonwear_log["DEVICE OFF"], format="%Y%b%d %H:%M")
            nonwear_log["DEVICE ON"] = pd.to_datetime(nonwear_log["DEVICE ON"], format="%Y%b%d %H:%M")

            nonwear_log = nonwear_log.fillna(self.epoch_timestamps[-1])
            nonwear_durs = nonwear_log["DEVICE ON"] - nonwear_log["DEVICE OFF"]

            self.nonwear_log = nonwear_log

            self.nonwear_log["PERIOD DURATION"] = nonwear_durs

            self.nonwear_log["PERIOD DURATION (MINS)"] = \
                [self.nonwear_log.iloc[i]["PERIOD DURATION"].total_seconds()/60
                 for i in range(self.nonwear_log.shape[0])]

            # re-does column names
            self.nonwear_log.columns = ["ID", "DEVICEOFF", "DEVICEON", "ENDCOLLECION",
                                        "PERIODDURATION", "PERIODDURATIONMINS"]

            self.nonwear_dict["Average Duration (Mins)"] = round(self.nonwear_log.describe()
                                                                 ["PERIODDURATION"]['mean'].total_seconds() / 60, 1)

            self.nonwear_dict["Number of Removals"] = self.nonwear_log.shape[0]

            print("\nNon-wear log data imported. Found {} removals.".format(self.nonwear_log.shape[0]))

        """if self.file_loc is None or not os.path.exists(self.file_loc):
            
            self.status = np.zeros(self.subject_object.data_len)  # Pretends participant did not remove device

            if self.file_loc is not None:
                if not os.path.exists(self.file_loc):
                    print("Non-wear log filepath given was not valid.")"""

        if self.file_loc is None or (self.file_loc is not None and not os.path.exists(self.file_loc)):
            self.status = np.zeros(self.subject_object.data_len)  # Pretends participant did not remove device
        if self.file_loc is not None and not os.path.exists(self.file_loc):
            print("Non-wear log filepath is not valid.")

    def mark_nonwear_epochs(self):
        """Creates a list of len(epoch_timestamps) where worn is coded as 0 and non-wear coded as 1"""

        if self.file_loc is None or not os.path.exists(self.file_loc):
            print("\nNo log found. Skipping non-wear epoch marking...")
            return None

        print("\nMarking non-wear epochs...")

        # Puts pd.df data into list --> mucho faster
        off_stamps = [i for i in self.nonwear_log["DEVICEOFF"]]
        on_stamps = [i for i in self.nonwear_log["DEVICEON"]]

        # Creates list of 0s corresponding to each epoch
        epoch_list = np.zeros(self.subject_object.data_len)

        for i, epoch_stamp in enumerate(self.epoch_timestamps):
            for off, on in zip(off_stamps, on_stamps):
                if off <= epoch_stamp <= on:
                    epoch_list[i] = 1

        self.status = epoch_list

        self.nonwear_dict["Minutes"] = (np.count_nonzero(self.status)) / (60 / self.subject_object.epoch_len)
        self.nonwear_dict["Percent"] = round(100 * self.nonwear_dict["Minutes"] *
                                             (60 / self.subject_object.epoch_len) /
                                             len(self.status), 2)

        print("Complete. Found {} hours, {} minutes of "
              "non-wear time.".format(np.floor(self.nonwear_dict["Minutes"]/60), self.nonwear_dict["Minutes"] % 60))
