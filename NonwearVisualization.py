from ImportEDF import *
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ImportEDF
import os


class Data:

    def __init__(self, subj_id, location, file_folder, log_file, study_code):
        """Class with ability to read in full accelerometer + temperature file and plot data or
           subsection of all devices for one participant.
           Reads in non-wear logs and highlights existing non-wear regions on plots.

            :argument
            -subj_id: 4 digit ID code
            -location: body segment of interest (for whole file read-in)
                       -"RA", "LA", "RW" or "LW"
            -file_folder: pathway to folder containing all EDF files
            -log_file pathway to nonwear log
            -study_code: "OND06" or "OND07"
        """

        self.accel = None
        self.temp = None
        self.subj_id = subj_id
        self.location = location
        self.log_file = log_file
        self.file_folder = file_folder
        self.study_code = study_code

        # Imports log
        self.import_log()

        self.df_all = None
        self.df_all_temp = None

        self.timestamp = None
        self.duration = None

        if location == "LW":
            self.long_loc = "LWrist"
        if location == "RW":
            self.long_loc = "RWrist"
        if location == "LA":
            self.long_loc = "LAnkle"
        if location == "RA":
            self.long_loc = "RAnkle"

        if self.study_code == "OND06":
            self.filename = self.file_folder + \
                            "OND06_SBH_{}_GNAC_ACCELEROMETER_{}.edf".format(self.subj_id, self.long_loc)

            self.filename_blank = self.file_folder + \
                                  "OND06_SBH_" + str(self.subj_id) + "_GNAC_ACCELEROMETER_{}.edf"

            self.temp_filename = self.file_folder + \
                                 "OND06_SBH_{}_GNAC_TEMPERATURE_{}.edf".format(self.subj_id, self.long_loc)

            self.temp_filename_blank = self.file_folder + "OND06_SBH_" + \
                                       str(self.subj_id) + "_GNAC_TEMPERATURE_{}.edf"

        if self.study_code == "OND07":
            self.filename = self.file_folder + \
                            "OND07_WTL_{}_02_GA_{}_Accelerometer.edf".format(self.subj_id, self.long_loc)

            self.filename_blank = self.file_folder + \
                            "OND07_WTL_" + str(self.subj_id) + "_02_GA_{}_Accelerometer.edf"

            self.temp_filename = self.file_folder + \
                                 "OND07_WTL_{}_GA_02_{}_Temperature.edf".format(self.subj_id, self.long_loc)
            self.temp_filename_blank = self.file_folder + \
                                       "OND07_WTL_" + str(self.subj_id) + \
                                       "_GA_02_{}_Temperature.edf"

    def import_log(self):
        """Imports nonwear log and saves participant's data as dataframe."""

        nw_log = pd.read_csv(self.log_file)
        self.nw_log = nw_log.loc[(nw_log["ID"] == self.subj_id)]
        self.nw_log["start_time"] = pd.to_datetime(self.nw_log["start_time"])
        self.nw_log["end_time"] = pd.to_datetime(self.nw_log["end_time"])

    def import_data(self):
        """Imports entire accelerometer and temperature file for wear location specified by 'location'"""

        """if self.study_code == "OND06" or self.study_code == "06":
            self.accel = GENEActiv(filepath=self.file_folder +
                                            "OND06_SBH_{}_GNAC_ACCELEROMETER_{}.edf".format(str(self.subj_id),
                                                                                            self.long_loc),
                                   load_raw=True, start_offset=0, end_offset=0)

            self.temp = GENEActivTemperature(filepath=self.file_folder +
                                                      "OND06_SBH_{}_GNAC_TEMPERATURE_{}.edf".format(str(self.subj_id),
                                                                                                    self.long_loc),
                                             from_processed=False, start_offset=0, end_offset=0)

        if self.study_code == "OND07" or self.study_code == "07":
            self.accel = GENEActiv(filepath=self.file_folder +
                                            "OND07_WTL_{}_02_{}_Accelerometer.edf".format(str(self.subj_id),
                                                                                            self.long_loc),
                                   load_raw=True, start_offset=0, end_offset=0)

            self.temp = GENEActivTemperature(filepath=self.file_folder +
                                                      "OND07_WTL_{}_02_{}_Temperature.edf".format(str(self.subj_id),
                                                                                                  self.long_loc),
                                             from_processed=False, start_offset=0, end_offset=0)"""

        if os.path.exists(self.filename):
            self.accel = GENEActiv(filepath=self.filename,
                                   load_raw=True, start_offset=0, end_offset=0)
        if not os.path.exists(self.filename):
            self.accel = GENEActiv(filepath=None,
                                   load_raw=True, start_offset=0, end_offset=0)

        if os.path.exists(self.temp_filename):
            self.temp = GENEActivTemperature(filepath=self.temp_filename,
                                             from_processed=False, start_offset=0, end_offset=0)
        if not os.path.exists(self.temp_filename):
            self.temp = GENEActivTemperature(filepath=None,
                                             from_processed=False, start_offset=0, end_offset=0)

    def plot_data(self):
        """Plots accel_x and accel_y over temperature for full data file.
           Shades areas deemed non-wear from all devices."""

        fig, (ax1, ax2) = plt.subplots(2, sharex='col', figsize=(10, 6))
        plt.suptitle("Subject {}: {}".format(str(self.subj_id), self.long_loc))
        plt.subplots_adjust(bottom=.2)

        ax1.set_title("LW = blue; RW = red; LA = green; RA = purple")
        ax1.plot(self.accel.timestamps[::8], self.accel.x[::8], color='black', label="x")
        ax1.plot(self.accel.timestamps[::8], self.accel.y[::8], color='grey', label="y")
        # ax1.plot(self.accel.timestamps[::5], self.accel.z[::5], color='dodgerblue', label="z")
        ax1.set_ylabel("G")
        ax1.legend()

        if self.temp.timestamps is not None:
            ax2.plot(self.temp.timestamps, self.temp.temperature, color='black')

        ax2.set_ylabel("Celcius")

        if self.nw_log is not None:
            for i in range(self.nw_log.dropna().shape[0]):

                # if self.nw_log.iloc[i]["location"] != self.location and self.nw_log.iloc[i]["location"] == "LW":
                if self.nw_log.iloc[i]["location"] == "LW":
                    fill_colour = "dodgerblue"
                    ylims_a = [-8, -4]
                    ylims_t = [20, 24]
                    if self.nw_log.iloc[i]["Confident"]:
                        alpha = 1
                    if not self.nw_log.iloc[i]["Confident"]:
                        alpha = .25

                #if self.nw_log.iloc[i]["location"] != self.location and self.nw_log.iloc[i]["location"] == "RW":
                if self.nw_log.iloc[i]["location"] == "RW":
                    fill_colour = "red"
                    ylims_a = [-4, 0]
                    ylims_t = [24, 28]
                    if self.nw_log.iloc[i]["Confident"]:
                        alpha = 1
                    if not self.nw_log.iloc[i]["Confident"]:
                        alpha = .25

                # if self.nw_log.iloc[i]["location"] != self.location and self.nw_log.iloc[i]["location"] == "LA":
                if self.nw_log.iloc[i]["location"] == "LA":
                    fill_colour = "green"
                    ylims_a = [0, 4]
                    ylims_t = [28, 32]
                    if self.nw_log.iloc[i]["Confident"]:
                        alpha = 1
                    if not self.nw_log.iloc[i]["Confident"]:
                        alpha = .35

                # if self.nw_log.iloc[i]["location"] != self.location and self.nw_log.iloc[i]["location"] == "RA":
                if self.nw_log.iloc[i]["location"] == "RA":
                    fill_colour = "purple"
                    ylims_a = [4, 8]
                    ylims_t = [32, 36]
                    if self.nw_log.iloc[i]["Confident"]:
                        alpha = 1
                    if not self.nw_log.iloc[i]["Confident"]:
                        alpha = .35

                try:
                    ax1.fill_betweenx(x1=self.nw_log.iloc[i]["start_time"], x2=self.nw_log.iloc[i]["end_time"], y=ylims_a,
                                      color=fill_colour, alpha=alpha)
                    ax2.fill_betweenx(x1=self.nw_log.iloc[i]["start_time"], x2=self.nw_log.iloc[i]["end_time"], y=ylims_t,
                                      color=fill_colour, alpha=alpha)
                except (UnboundLocalError, TypeError):
                    pass

        xfmt = mdates.DateFormatter("%Y/%m/%d %H:%M:%S")

        ax2.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)
        ax2.axhline(y=20, color='red', linestyle='dashed')
        ax2.axhline(y=36, color='red', linestyle='dashed')

    def import_all_accels(self, timestamp=None, duration=120):
        """Imports all available accelerometer and temperature data from participants in specified time period.
           Crops data to start 20 minutes before 'timestamp' and reads in 'duration' number of minutes."""

        # Timestamp formatting ----------------------------------------------------------------------------------------
        if timestamp is None:
            timestamp = self.nw_log.iloc[0]["start_time"]

        if timestamp is not None:
            try:
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

            except (TypeError, ValueError):
                try:
                    timestamp = timestamp + timedelta(minutes=-10)
                except TypeError:
                    timestamp = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S") + timedelta(minutes=-20)

        start_time, end_time, fs, dur = ImportEDF.check_file(self.filename, print_summary=False)

        self.timestamp = timestamp
        self.duration = duration

        start_index = int((timestamp - start_time).total_seconds() * fs)
        stop_index = int(duration * fs * 60)

        # Left ankle -------------------------------------------------------------------------------------------------
        try:
            la = GENEActiv(filepath=self.filename_blank.format("LAnkle"),
                           load_raw=True, start_offset=start_index, end_offset=stop_index)

            la_temp = GENEActivTemperature(filepath=self.temp_filename_blank.format("LAnkle"),
                                           start_offset=0, end_offset=0)

            df = pd.DataFrame(list(zip(la_temp.timestamps, la_temp.temperature)), columns=["Time", "Temp"])

            df = df.loc[(df["Time"] >= timestamp) & (df["Time"] <= timestamp + timedelta(minutes=duration))]

            la_time = la.timestamps
            lax = la.x
            lay = la.y
            laz = la.z
            lat = df["Temp"]
            la_temp_time = df["Time"]

        except OSError:
            print("-LAnkle file not found.")
            la_time = [start_time + timedelta(seconds=i/fs) for i in range(stop_index)]
            lax = [0 for i in range(stop_index)]
            lay = [0 for i in range(stop_index)]
            laz = [0 for i in range(stop_index)]
            lat = [0 for i in range(stop_index)]

        # Right ankle -------------------------------------------------------------------------------------------------
        try:
            ra = GENEActiv(filepath=self.filename_blank.format("RAnkle"),
                           load_raw=True, start_offset=start_index, end_offset=stop_index)

            ra_temp = GENEActivTemperature(filepath=self.temp_filename_blank.format("RAnkle"),
                                           start_offset=0, end_offset=0)

            df = pd.DataFrame(list(zip(ra_temp.timestamps, ra_temp.temperature)), columns=["Time", "Temp"])
            df = df.loc[(df["Time"] >= timestamp) & (df["Time"] <= timestamp + timedelta(minutes=duration))]

            ra_time = ra.timestamps
            rax = ra.x
            ray = ra.y
            raz = ra.z
            rat = df["Temp"]
            ra_temp_time = df["Time"]

        except OSError:
            print("-RAnkle file not found.")
            ra_time = [start_time + timedelta(seconds=i/fs) for i in range(stop_index)]
            rax = [0 for i in range(stop_index)]
            ray = [0 for i in range(stop_index)]
            raz = [0 for i in range(stop_index)]
            rat = [0 for i in range(stop_index)]

        # Left wrist --------------------------------------------------------------------------------------------------
        try:
            lw = GENEActiv(filepath=self.filename_blank.format("LWrist"),
                           load_raw=True, start_offset=start_index, end_offset=stop_index)

            lw_temp = GENEActivTemperature(filepath=self.temp_filename_blank.format("LWrist"),
                                           start_offset=0, end_offset=0)

            df = pd.DataFrame(list(zip(lw_temp.timestamps, lw_temp.temperature)), columns=["Time", "Temp"])
            df = df.loc[(df["Time"] >= timestamp) & (df["Time"] <= timestamp + timedelta(minutes=duration))]

            lw_time = lw.timestamps
            lwx = lw.x
            lwy = lw.y
            lwz = lw.z
            lwt = df["Temp"]
            lw_temp_time = df['Time']

        except OSError:
            print("-LWrist file not found.")
            lw_time = [start_time + timedelta(seconds=i/fs) for i in range(stop_index)]
            lwx = [0 for i in range(stop_index)]
            lwy = [0 for i in range(stop_index)]
            lwz = [0 for i in range(stop_index)]
            lwt = [0 for i in range(stop_index)]

        # Right wrist -------------------------------------------------------------------------------------------------
        try:
            rw = GENEActiv(filepath=self.filename_blank.format("RWrist"),
                           load_raw=True, start_offset=start_index, end_offset=stop_index)

            rw_temp = GENEActivTemperature(filepath=self.temp_filename_blank.format("RWrist"),
                                           start_offset=0, end_offset=0)

            df = pd.DataFrame(list(zip(rw_temp.timestamps, rw_temp.temperature)), columns=["Time", "Temp"])
            df = df.loc[(df["Time"] >= timestamp) & (df["Time"] <= timestamp + timedelta(minutes=duration))]

            rw_time = rw.timestamps
            rwx = rw.x
            rwy = rw.y
            rwz = rw.z
            rwt = df["Temp"]
            rw_temp_time = df["Time"]

        except OSError:
            print("-RWrist file not found.")
            rw_time = [start_time + timedelta(seconds=i/fs) for i in range(stop_index)]
            rwx = [0 for i in range(stop_index)]
            rwy = [0 for i in range(stop_index)]
            rwz = [0 for i in range(stop_index)]
            rwt = [0 for i in range(stop_index)]

        df = pd.DataFrame(list(zip(la_time, lax, lay, laz,
                                   ra_time, rax, ray, raz,
                                   lw_time, lwx, lwy, lwz,
                                   rw_time, rwx, rwy, rwz)),
                          columns=["LA_time", "LA_x", "LA_y", "LA_z",
                                   "RA_time", "RA_x", "RA_y", "RA_z",
                                   "LW_time", "LW_x", "LW_y", "LW_z",
                                   "RW_time", "RW_x", "RW_y", "RW_z"])

        df_temp = pd.DataFrame(list(zip(la_temp_time, lat,
                                        ra_temp_time, rat,
                                        lw_temp_time, lwt,
                                        rw_temp_time, rwt)),
                               columns=["LA_time", "LA_temp", "RA_time", "RA_temp",
                                        "LW_time", "LW_temp", "RW_time", "RW_temp"])

        print("\nIMPORTED DATA FROM {} to {}.".format(start_time, start_time + timedelta(minutes=duration)))

        return df, df_temp

    def plot_all(self):
        """Plots all available data from specified time period. Shades nonwear periods."""

        # Re-loads log
        self.import_log()

        plt.close("all")

        log = self.nw_log.loc[(self.nw_log["start_time"] >= self.timestamp) &
                              (self.nw_log["end_time"] <= self.timestamp + timedelta(minutes=self.duration))]

        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex='col', figsize=(10, 8))
        plt.suptitle("Participant {}: {} ({} minutes)".format(self.subj_id, self.timestamp, self.duration))

        ax1.set_title("Shaded areas: green = confident, red = not confident")
        ax1.plot(self.df_all["LA_time"], self.df_all["LA_x"], color='black')
        ax1.plot(self.df_all["LA_time"], self.df_all["LA_y"], color='grey')
        ax1.set_ylabel("Left ankle (G)")

        ax2.plot(self.df_all["RA_time"], self.df_all["RA_x"], color='black')
        ax2.plot(self.df_all["RA_time"], self.df_all["RA_y"], color='dodgerblue')
        ax2.set_ylabel("Right ankle (G)")

        ax3.plot(self.df_all["LW_time"], self.df_all["LW_x"], color='black')
        ax3.plot(self.df_all["LW_time"], self.df_all["LW_y"], color='green')
        ax3.set_ylabel("Left wrist (G)")

        ax4.plot(self.df_all["RW_time"], self.df_all["RW_x"], color='black')
        ax4.plot(self.df_all["RW_time"], self.df_all["RW_y"], color='red')
        ax4.set_ylabel("Right wrist (G)")

        ax5.plot(self.df_all_temp["LA_time"], self.df_all_temp["LA_temp"], color='grey', label='LA')
        ax5.plot(self.df_all_temp["RA_time"], self.df_all_temp["RA_temp"], color='dodgerblue', label='RA')
        ax5.plot(self.df_all_temp["LW_time"], self.df_all_temp["LW_temp"], color='green', label='LW')
        ax5.plot(self.df_all_temp["RW_time"], self.df_all_temp["RW_temp"], color='red', label='RW')
        ax5.set_ylabel("ºC")

        xfmt = mdates.DateFormatter("%H:%M:%S")
        ax5.xaxis.set_major_formatter(xfmt)
        plt.xticks(rotation=45, fontsize=8)
        ax5.set_ylim(18, 36)

        for i in range(log.shape[0]):

            # Left ankle -----------------------------
            if log.iloc[i]["location"] == "LA":
                if log.iloc[i]["Confident"]:
                    col = 'green'
                if not log.iloc[i]["Confident"]:
                    col = 'red'
                label = "csv row " + str(log.iloc[i].name + 2)

                ax1.fill_betweenx(x1=log.iloc[i]["start_time"], x2=log.iloc[i]["end_time"], y=ax1.get_ylim(),
                                  color=col, alpha=.25, label=label)

            # Right ankle ----------------------------
            if log.iloc[i]["location"] == "RA":
                if log.iloc[i]["Confident"]:
                    col = 'green'
                if not log.iloc[i]["Confident"]:
                    col = 'red'
                label = "csv row " + str(log.iloc[i].name + 2)

                ax2.fill_betweenx(x1=log.iloc[i]["start_time"], x2=log.iloc[i]["end_time"], y=ax2.get_ylim(),
                                  color=col, alpha=.25, label=label)

            # Left wrist ----------------------------
            if log.iloc[i]["location"] == "LW":
                if log.iloc[i]["Confident"]:
                    col = 'green'
                if not log.iloc[i]["Confident"]:
                    col = 'red'
                label = "csv row " + str(log.iloc[i].name + 2)

                ax3.fill_betweenx(x1=log.iloc[i]["start_time"], x2=log.iloc[i]["end_time"], y=ax3.get_ylim(),
                                  color=col, alpha=.25, label=label)

            # Right wrist ----------------------------
            if log.iloc[i]["location"] == "RW":
                if log.iloc[i]["Confident"]:
                    col = 'green'
                if not log.iloc[i]["Confident"]:
                    col = 'red'
                label = "csv row " + str(log.iloc[i].name + 2)

                ax4.fill_betweenx(x1=log.iloc[i]["start_time"], x2=log.iloc[i]["end_time"], y=ax4.get_ylim(),
                                  color=col, alpha=.25, label=label)

        ax1.legend(loc='upper right')
        ax2.legend(loc='upper right')
        ax3.legend(loc='upper right')
        ax4.legend(loc='upper right')
        ax5.legend(loc='upper right')


x = Data(subj_id=2891, location="LA", file_folder="/Users/kyleweber/Desktop/",
         log_file="/Users/kyleweber/Desktop/ReMiNDDNonWearReformatted_vt_25AUG2020.csv", study_code="OND06")
# x.import_data()  # Imports one full file
# x.plot_data()  # Plots one full file with all NW marked

# x.df_all, x.df_all_temp = x.import_all_accels(timestamp=None, duration=120)
# x.plot_all()
