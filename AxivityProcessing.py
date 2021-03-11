try:
    from read_cwa.ax_to_edf import ax_to_edf
except ImportError:
    print("Script requires code from https://github.com/arsalikhov/read_cwa")

import pyedflib
import pandas as pd
import matplotlib.pyplot as plt
import os

"""============================================== CWA TO EDF CONVERSION ============================================"""

"""
input_file = "C:/Users/ksweber/Desktop/007_Axivity_Test/007_AxTest.cwa"
save_loc = "C:/Users/ksweber/Desktop/"

ax_to_edf(input_file_path=input_file, accelerometer_dir=save_loc)
"""

""""==================================================== EDF IMPORT ================================================"""


class Axivity:

    def __init__(self, accel_edf_path=None, temp_edf_path=None, temperature_period=1):

        self.acc = None
        self.temp = None
        self.temp_per = temperature_period

        self.df_acc, self.fs_acc = self.import_files(file=accel_edf_path, file_type="accel")
        self.df_temp, self.fs_temp = self.import_files(file=temp_edf_path, file_type="temperature",
                                                       temp_sample_int=self.temp_per)

    @staticmethod
    def import_files(file, file_type="accel", temp_sample_int=1):

        if file is not None and not os.path.exists(file):
            print("\nFile {} does not exists.".format(file))
            return None, None

        if file is None:
            return None, None

        if file is not None and os.path.exists(file):
            print("\nImporting {}...".format(file))

            if file_type == "accel":

                acc_file = pyedflib.EdfReader(file)
                h = acc_file.getSignalHeader(0)
                s = acc_file.getStartdatetime()
                fs = h["sample_rate"]

                df = pd.DataFrame([acc_file.readSignal(0), acc_file.readSignal(1), acc_file.readSignal(2)]).transpose()
                df.columns = ["x", "y", "z"]
                df.insert(loc=0, column="Timestamp",
                          value=pd.date_range(start=s, periods=df.shape[0], freq="{}ms".format(1000/fs)))

            if file_type == "temperature":
                temp_file = pyedflib.EdfReader(file)
                h = temp_file.getSignalHeader(0)
                s = temp_file.getStartdatetime()
                fs = h["sample_rate"]

                df = pd.DataFrame(temp_file.readSignal(0)[::int(temp_sample_int*fs)], columns=["Temperature"])
                df.insert(loc=0, column="Timestamp",
                          value=pd.date_range(start=s, freq="{}S".format(temp_sample_int), periods=df.shape[0]))

                fs = 1 / temp_sample_int

            print("Complete.")

        return df, fs

    def plot_data(self):

        fig, axes = plt.subplots(2, sharex='col', figsize=(12, 7))

        axes[0].set_title("Accelerometer")
        axes[0].plot(self.df_acc["Timestamp"], self.df_acc["x"], color='black', label="X")
        axes[0].plot(self.df_acc["Timestamp"], self.df_acc["y"], color='red', label="Y")
        axes[0].plot(self.df_acc["Timestamp"], self.df_acc["z"], color='dodgerblue', label="Z")
        axes[0].legend()
        axes[0].set_ylabel("G")

        axes[1].plot(self.df_temp["Timestamp"], self.df_temp["Temperature"], color='green')
        axes[1].set_title("Temperature ({} Hz)".format(round(self.fs_temp, 3)))


"""
x = Axivity(accel_edf_path="C:/Users/ksweber/Desktop/007_Axivity_Test/007_AxTestAcc.edf",
            temp_edf_path="C:/Users/ksweber/Desktop/007_Axivity_Test/007_AxTestTemp.edf", temperature_period=2)
x.plot_data()
"""
