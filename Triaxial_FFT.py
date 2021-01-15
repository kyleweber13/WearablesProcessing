import datetime
import matplotlib.pyplot as plt
from datetime import timedelta
import pandas as pd
import numpy as np
import scipy.fft


def run_fft(timestamps=None, acc_x=None, acc_y=None, acc_z=None, acc_vm=None,
            start=None, duration_seconds=10, sample_rate=75, plot_raw=False, downsample=3):
    """Runs FFT on given data. Plots output data.

    :argument
    -timestamps: list/array of timestamps in "%Y-%m-%d %H:%M:%S" format
    -acc_x/acc_y/acc_z/acc_vm: list/array of accelerometer axis data. Need to be synced with timestamps.
    -start: timestamp of where to start window of data. If None, uses start of data
    -duration_seconds: window length in seconds
    -sample_rate: Hz
    -plot_raw: boolean to plot accelerometer data
    -downsample: factor by which accelerometer data is downsampled for plotting only
    """

    if start is not None:
        start_index = int((datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S") -
                           pd.to_datetime(timestamps[0])).total_seconds() *
                          sample_rate)
        start = pd.to_datetime(timestamps[0])

    if start is None:
        start_index = 0
        start = pd.to_datetime(timestamps[0])

    stop_index = start_index + sample_rate * duration_seconds
    data_len = int(sample_rate * duration_seconds)

    if acc_x is not None:
        fft_x = scipy.fft.fft(acc_x[start_index:stop_index])
    if acc_x is None:
        fft_x = None
    if acc_y is not None:
        fft_y = scipy.fft.fft(acc_y[start_index:stop_index])
    if acc_y is None:
        fft_y = None
    if acc_z is not None:
        fft_z = scipy.fft.fft(acc_z[start_index:stop_index])
    if acc_z is None:
        fft_z = None
    if acc_vm is not None:
        fft_vm = scipy.fft.fft(acc_vm[start_index:stop_index])
    if acc_vm is None:
        fft_vm = None

    xf = np.linspace(0.0, 1.0 / (2.0 * (1 / sample_rate)), data_len // 2)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(11, 8))
    plt.subplots_adjust(hspace=.3)
    ax1.set_title("FFT from {} to {}".format(start, start + timedelta(seconds=duration_seconds)))

    if plot_raw:
        if acc_x is not None:
            ax1.plot(timestamps[start_index:stop_index:downsample], acc_x[start_index:stop_index:downsample],
                     color='black', label="X")
        if acc_y is not None:
            ax1.plot(timestamps[start_index:stop_index:downsample], acc_y[start_index:stop_index:downsample],
                     color='red', label="Y")
        if acc_z is not None:
            ax1.plot(timestamps[start_index:stop_index:downsample], acc_z[start_index:stop_index:downsample],
                     color='dodgerblue', label="Z")
        ax1.set_ylabel("G")

    if acc_x is not None:
        ax2.plot(xf, 2.0 / data_len / 2 * np.abs(fft_x[0:data_len // 2]), color='black', label="FFT_x")
    if acc_y is not None:
        ax3.plot(xf, 2.0 / data_len / 2 * np.abs(fft_y[0:data_len // 2]), color='red', label="FFT_y")
    if acc_z is not None:
        ax4.plot(xf, 2.0 / data_len / 2 * np.abs(fft_z[0:data_len // 2]), color='dodgerblue', label="FFT_z")
    if acc_vm is not None:
        ax5.plot(xf, 2.0 / data_len / 2 * np.abs(fft_vm[0:data_len // 2]), color='magenta', label="FFT_VM")

    ax2.fill_between(x=[3, 7], y1=0, y2=ax2.get_ylim()[1], color='green', alpha=.5)
    ax3.fill_between(x=[3, 7], y1=0, y2=ax3.get_ylim()[1], color='green', alpha=.5)
    ax4.fill_between(x=[3, 7], y1=0, y2=ax4.get_ylim()[1], color='green', alpha=.5)
    ax5.fill_between(x=[3, 7], y1=0, y2=ax5.get_ylim()[1], color='green', alpha=.5)

    ax2.set_ylabel("Power")
    ax3.set_ylabel("Power")
    ax4.set_ylabel("Power")
    ax5.set_ylabel("Power")
    ax5.set_xlabel("Frequency (Hz)")
