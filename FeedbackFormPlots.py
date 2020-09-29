import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


# ============================================ Sample code for feedback form ==========================================


def plot_activity_page_plot(subject_object):
    fig, ax1 = plt.subplots(1)
    ax1.plot(subject_object.wrist.epoch.timestamps[:-1], subject_object.wrist.epoch.svm, color='black')
    ax1.fill_between(x=[subject_object.sleep.data.iloc[0]["TIME_SLEEP"],
                        subject_object.sleep.data.iloc[1]["TIME_WAKE"]],
                     y1=0, y2=max(subject_object.wrist.epoch.svm), color='#46b9db', alpha=.75)
    ax1.fill_between(x=[subject_object.sleep.data.iloc[2]["TIME_WAKE"],
                        subject_object.sleep.data.iloc[2]["TIME_SLEEP"]],
                     y1=0, y2=max(subject_object.wrist.epoch.svm), color='#87db46', alpha=.75)
    ax1.fill_between(x=[subject_object.nonwear.nonwear_log.iloc[2]["DEVICE OFF"],
                        subject_object.nonwear.nonwear_log.iloc[2]["DEVICE ON"]],
                     y1=0, y2=max(subject_object.wrist.epoch.svm), color='#db8446', alpha=.75)
    xfmt = mdates.DateFormatter("%a, %b %-d \n %I:%M %p")
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.set_yticks([])
    ax1.set_ylabel("Less             Movement              More")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)


def plot_sleep():

    from datetime import timedelta

    fig, ax1 = plt.subplots(1)
    ax1.plot(subject_object.wrist.epoch.timestamps[:-1], subject_object.wrist.epoch.svm, color='black')
    ax1.fill_between(x=[subject_object.sleep.data.iloc[0]["TIME_SLEEP"], subject_object.sleep.data.iloc[1]["TIME_WAKE"]],
                     y1=0, y2=max(subject_object.wrist.epoch.svm), color='#46b9db', alpha=.75)

    xfmt = mdates.DateFormatter("%a, %b %-d \n %I:%M %p")
    locator = mdates.HourLocator(byhour=np.arange(0, 24, 2), interval=1)

    ax1.xaxis.set_major_formatter(xfmt)
    ax1.xaxis.set_major_locator(locator)

    ax1.set_yticks([])
    ax1.set_ylabel("Less             Movement              More")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    plt.xlim([subject_object.sleep.data.iloc[0]["TIME_SLEEP"]-timedelta(hours=2),
              subject_object.sleep.data.iloc[1]["TIME_WAKE"]+timedelta(hours=2)])


def plot_ecg():
    fig, ax1 = plt.subplots(1)
    plt.plot(np.arange(0, 250 * 10, 1) / 250, subject_object.ecg.filtered[6307500:6307500 + 250 * 10], color='black')
    ax1.set_yticks(())
    ax1.set_xlabel("Seconds")
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
