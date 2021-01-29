import pandas as pd
import numpy as np
from datetime import timedelta


def approxentropy(data, m=2, r=3) -> float:
    """Approximate_entropy from Pincus et al., 1991.
       Code from https://en.wikipedia.org/wiki/Approximate_entropy#Python_implementation

    :argument
    -data: time series data
    -m, r: not sure --> see paper

    :return
    -approximate entropy value: 0 = noise; 2 = perfect agreement
    """

    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[data[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [
            len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0)
            for x_i in x
        ]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    N = len(data)

    return abs(_phi(m + 1) - _phi(m))


def interday_stability(timestamps, data):
    """Calculates interday stability measure using accelerometer data. Equation from Merilahti et al. (2016).

    :argument
    -timestamps: list/array of timestamps
    -data: outcome measure data; indexes need to match timestamps

    :return
    -weekly_df: dataframe of hourly time series data over course of collection
    -hourly_df: average of each hour across days
    -IS: interday stability measure (0-1; 0 = noise, 1 = perfect agreement)
    """

    # Gets start dates (days)
    dates = sorted(set([i.date() for i in timestamps]))

    # Crops data to exclude first and last days
    timestamps = timestamps.loc[(timestamps >= pd.to_datetime(str(dates[1]) + " " + "00:00:00")) &
                                (timestamps <= pd.to_datetime(str(dates[-1]) + " " + "00:00:00"))]
    data = data.iloc[timestamps.index[0]:timestamps.index[-1] + 1]

    data = pd.DataFrame(list(zip(timestamps, data)), columns=["Timestamp", "Data"])

    # Boundaries for each epoch (hourly)
    hours = [i for i in
             pd.date_range(start=timestamps.iloc[0], end=timestamps.iloc[-1] + timedelta(hours=1), freq="1H")]

    # Calculates hourly average SVM
    hourly_averages = []
    for start, stop in zip(hours[:], hours[1:]):
        d = data.loc[(start <= data["Timestamp"]) & (data["Timestamp"] <= stop)]
        hourly_averages.append(d["Data"].mean())

    hourly_data = pd.DataFrame(list(zip(hours, hourly_averages, [i.time().hour for i in hours[:-1]])),
                               columns=["Timestamp", "Average", "Hour"])

    # Equation
    n = hourly_data.shape[0]  # total number of one-hour epochs
    p = 24  # number of epochs per day
    xh = [hourly_data["Average"].loc[hourly_data["Hour"] == i].mean() for i in
          range(24)]  # Mean of each hour across days
    xi = hourly_data["Average"]  # Average for each hour (time series)
    x = xi.mean()  # Mean of all epochs

    numerator = n * sum([(h - x) ** 2 for h in xh])
    denominator = p * sum([(i - x) ** 2 for i in xi])

    IS = numerator / denominator

    weekly_df = pd.DataFrame(list(zip(hours, xi)), columns=["Timestamp", "Average"])
    hourly_df = pd.DataFrame(list(zip([i for i in range(24)], xh)), columns=["Hour", "Average"])

    return weekly_df, hourly_df, round(IS, 5)


# file = pd.read_csv("O:/Data/ReMiNDD/Processed Data/Activity/Epoched Wrist Data/OND06_SBH_1027_EpochedAccelerometer.csv")

# weekly, hourly, IS = interday_stability(timestamps=pd.to_datetime(file["Timestamp"]), data=file["WristSVM"])
