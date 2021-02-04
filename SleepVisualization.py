from ImportEDF import *
import matplotlib.pyplot as plt
import pandas as pd


def import_subject(id):
    accel = GENEActiv(filepath="/Volumes/Kyle's External HD/OND06 NDWrist Data/Accelerometer/OND06_SBH_{}_"
                               "GNAC_ACCELEROMETER_LWrist.edf".format(id),
                      load_raw=True, start_offset=0, end_offset=0)

    light = GENEActivLight("/Volumes/Kyle's External HD/OND06 NDWrist Data/Light/OND06_SBH_{}_"
                           "GNAC_LIGHT_LWrist.edf".format(id))

    temp = GENEActivTemperature("/Volumes/Kyle's External HD/OND06 NDWrist Data/Temperature/"
                                "OND06_SBH_{}_GNAC_TEMPERATURE_LWrist.edf".format(id))

    nw = pd.read_csv("/Users/kyleweber/Desktop/Data/OND06/ReMiNDDNonWearReformatted_GAgoldstandarddataset_15Dec2020.csv")
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

    def onclick(event):
        if event.xdata != None:
            date = mdates.num2date(event.xdata)
            print(date.strftime('%Y-%m-%d %H:%M:%S'))

    cid = fig.canvas.mpl_connect('button_press_event', onclick)

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

    nw = pd.read_csv("/Users/kyleweber/Desktop/Data/OND06/ReMiNDDNonWearReformatted_GAgoldstandarddataset_6Nov2020.csv")
    nw = nw.loc[nw["ID"] == id]
    nw = nw.loc[nw["location"] == "LW"]
    nw["start_time"] = pd.to_datetime(nw["start_time"])
    nw["end_time"] = pd.to_datetime(nw["end_time"])

    for row in nw.itertuples():
        ax1.fill_between(x=[row.start_time, row.end_time], y1=0, y2=max(accel.svm), color='red', alpha=.5)
        ax2.fill_between(x=[row.start_time, row.end_time], y1=min(temp.temperature), y2=max(temp.temperature),
                         color='red', alpha=.5)
        ax3.fill_between(x=[row.start_time, row.end_time], y1=0, y2=max(light.light_avg), color='red', alpha=.5)


a, l, t, nw = import_subject(1039)
# check_log(2364)
