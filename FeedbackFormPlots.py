import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# ============================================ Sample code for feedback form ==========================================


def plot_activity_page_plot(subject_object):
    plt.rcParams.update({'font.size': 14, })
    fig, ax1 = plt.subplots(1)
    ax1.plot(subject_object.wrist.epoch.timestamps, subject_object.wrist.epoch.svm, color='black')
    ax1.fill_between(x=[subject_object.sleep.data.iloc[0]["TIME_SLEEP"],
                        subject_object.sleep.data.iloc[1]["TIME_WAKE"]],
                     y1=0, y2=max(subject_object.wrist.epoch.svm), color='#46b9db', alpha=.75)
    ax1.fill_between(x=[subject_object.sleep.data.iloc[2]["TIME_WAKE"],
                        subject_object.sleep.data.iloc[2]["TIME_SLEEP"]],
                     y1=0, y2=max(subject_object.wrist.epoch.svm), color='#87db46', alpha=.75)
    ax1.fill_between(x=[subject_object.nonwear.nonwear_log.iloc[2]["DEVICEOFF"],
                        subject_object.nonwear.nonwear_log.iloc[2]["DEVICEON"]],
                     y1=0, y2=max(subject_object.wrist.epoch.svm), color='#db8446', alpha=.75)
    xfmt = mdates.DateFormatter("%a, %b %-d \n %I:%M %p")
    ax1.xaxis.set_major_formatter(xfmt)
    plt.xticks(rotation=45)
    ax1.set_yticks([])
    ax1.set_ylabel("Less         Movement          More")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)


def plot_sleep(subject_object):

    from datetime import timedelta
    plt.rcParams.update({'font.size': 14, })

    fig, ax1 = plt.subplots(1)
    ax1.plot(subject_object.wrist.epoch.timestamps, subject_object.wrist.epoch.svm, color='black')
    ax1.fill_between(x=[subject_object.sleep.data.iloc[0]["TIME_SLEEP"], subject_object.sleep.data.iloc[1]["TIME_WAKE"]],
                     y1=0, y2=max(subject_object.wrist.epoch.svm), color='#46b9db', alpha=.75)

    xfmt = mdates.DateFormatter("%a, %b %-d \n %I:%M %p")
    locator = mdates.HourLocator(byhour=np.arange(0, 24, 2), interval=1)

    ax1.xaxis.set_major_formatter(xfmt)
    ax1.xaxis.set_major_locator(locator)

    ax1.set_yticks([])
    plt.xticks(rotation=45)
    ax1.set_ylabel("Less         Movement          More")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.xlim([subject_object.sleep.data.iloc[0]["TIME_SLEEP"]-timedelta(hours=2),
              subject_object.sleep.data.iloc[1]["TIME_WAKE"]+timedelta(hours=2)])

# plot_sleep(x)


def plot_ecg(subject_object):
    plt.rcParams.update({'font.size': 14, })

    fig, ax1 = plt.subplots(1)
    plt.plot(np.arange(0, 250 * 10, 1) / 250, subject_object.ecg.filtered[6307500:6307500 + 250 * 10], color='black')
    ax1.set_yticks(())
    ax1.set_xlabel("Seconds")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

# plot_ecg(x)


def plot_gait_bouts(la_filename=None, ra_filename=None):

    la_filename = pd.read_csv(la_filename)
    ra_filename = pd.read_csv(ra_filename)

    df_lankle = pd.read_csv(la_filename, skiprows=99, sep=",")
    df_lankle.columns = ["Timestamp", "X", "Y", "Z", "Lux", "Button", "Temperature"]
    df_rankle = pd.read_csv(ra_filename, skiprows=99, sep=",")
    df_rankle.columns = ["Timestamp", "X", "Y", "Z", "Lux", "Button", "Temperature"]

    plt.rcParams.update({'font.size': 14, })
    plt.plot(np.arange(0, 164000-162100)/75, df_lankle["X"].iloc[162100:164000], color='black')
    plt.plot(np.arange(0, 164000-162100)/75, df_rankle["X"].iloc[162100:164000], color='red')
    plt.xlabel("Seconds")
    plt.yticks([])
    plt.fill_between(x=[0, 6], y1=-8, y2=8, color='orange', alpha=.7)
    plt.fill_between(x=[9.4, 21.5], y1=-8, y2=8, color='green', alpha=.5)
    plt.fill_between(x=[6, 9.4], y1=-8, y2=8, color='grey', alpha=.7)
    plt.fill_between(x=[21.5, 25], y1=-8, y2=8, color='grey', alpha=.7)


#plot_gait_bouts(la_filename="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab4_LAnkle.csv",
#                ra_filename="/Users/kyleweber/Desktop/Python Scripts/WearablesCourse/Data Files/Lab4_RAnkle.csv")