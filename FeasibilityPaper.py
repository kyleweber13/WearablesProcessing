import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import scipy.stats as stats
from scikit_posthocs import posthoc_nemenyi_friedman as nemenyi
from scikit_posthocs import posthoc_conover_friedman as conover
import math

"""
===================================================== DATA IMPORT =====================================================
"""

filename = "/Users/kyleweber/Desktop/Data Tables From Adam_11Feb2021.xlsx"

# Average nonwear rate per participant across 4 days of wear
df1 = pd.read_excel(filename, sheet_name="Figure 1", skiprows=3, usecols=[0, 1, 2, 3, 4, 5])

df1_long = pd.melt(df1, id_vars="Participant", var_name="Location", value_name="Values")
df1_desc = df1_long.groupby("Location")["Values"].describe()

# Nonwear rate (at least 2 sensors) by time of day
df2a = pd.read_excel(filename, sheet_name="Figure 2A", skiprows=4, usecols=[0, 1, 2])
df2a_desc = df2a[["Day", "Night"]].describe().transpose()

# Nonwear rates by day, >= 2 devices removed
df2b = pd.read_excel(filename, sheet_name="Figure 2b", skiprows=4, usecols=[0, 1, 2, 3, 4, 5, 6])
df2b_long = pd.melt(df2b, id_vars="Participant", var_name="Day", value_name="Values")
df2b_desc = df2b_long.groupby("Day")["Values"].describe()

# Daily average nonwear % by cohort
df_intext = pd.read_excel(filename, sheet_name="Kyle's Tab")
df_intext_desc = df_intext.groupby("Cohort")["Values"].describe()

"""
===================================================== STATISTICS ======================================================
"""

def check_normality(dataframe, transform=None):

    print("Transformation = {}".format(transform))

    for col in dataframe.columns[1:]:
        if transform is None:
            data = stats.shapiro(dataframe[col])
        if transform == "log":
            if 0 in dataframe[col]:
                d = [i for i in dataframe[col] + 1]
            if 0 not in dataframe[col]:
                d = [i for i in dataframe[col]]
            data = stats.shapiro([math.log(i) for i in d])
        if transform == "root":
            data = stats.shapiro([math.sqrt(i) for i in dataframe[col]])

        sig = "***" if data[1] < .05 else ""
        print("-{}: W = {}, p = {} {}".format(col, round(data[0], 3), data[1], sig))


def generate_histograms(dataframe):
    n_plots = dataframe.shape[1] - 1
    if n_plots > 3:
        plot_shape = (2, int(np.ceil(n_plots / 2)))
    if n_plots <= 3:
        plot_shape = (1, n_plots)

    fig, axes = plt.subplots(plot_shape[0], plot_shape[1], figsize=(10, 6))

    col_ind = 1
    for ax in axes:
        try:
            for subax in range(len(ax)):
                try:
                    ax[subax].hist(dataframe[dataframe.columns[col_ind]], edgecolor='black', color='dodgerblue')
                    ax[subax].set_title(dataframe.columns[col_ind])
                    col_ind += 1
                except IndexError:
                    pass
        except TypeError:
            try:
                ax.hist(dataframe[dataframe.columns[col_ind]], edgecolor='black', color='dodgerblue')
                ax.set_title(dataframe.columns[col_ind])
                col_ind += 1
            except IndexError:
                pass


def calculate_friedman(dataframe, usetest="friedman"):
    stats1_nemenyi = None
    stats1_conover = None

    if usetest == "friedman":
        print("Running Friedman test.")
        stats_main = pg.friedman(data=dataframe, dv="Values", within=dataframe.columns[1],
                                 subject="Participant", method='f')
    if usetest == "chisq":
        print("Running Chi squared test.")
        stats_main = pg.friedman(data=dataframe, dv="Values", within=dataframe.columns[1],
                                 subject="Participant", method='chisq')

    stats_main["Sig."] = ["*" if stats_main["p-unc"].iloc[0] < .05 else ""]

    if stats_main["Sig."].iloc[0] == "*":
        print("-{} result was significant".format(usetest.capitalize()))
        print("    -Running Nemenyi and Conover post-hoc tests.")
        print("        -Nemenyi accounts for multiple comparisons while Conover does not.")

        stats_nemenyi = nemenyi(a=dataframe, y_col="Values", block_col="Participant",
                                group_col=dataframe.columns[1], melted=True)

        stats_conover = conover(a=dataframe, y_col="Values", block_col="Participant",
                                group_col=dataframe.columns[1], melted=True)

    return stats_main, stats_nemenyi, stats_conover


def calculate_kruskalwallish(dataframe):

    stats_pairwise_t = None

    stats_main = pg.kruskal(data=dataframe, dv="Values", between=dataframe.columns[1], detailed=True)
    stats_main["Sig."] = ["*" if stats_main["p-unc"].iloc[0] < .05 else " "]

    # if stats_main["Sig."].iloc[0] == "*":

    stats_pairwise_t = pg.pairwise_ttests(data=dataframe, dv="Values", between=dataframe.columns[1],
                                          within=None, subject="Participant", parametric=False,
                                          marginal=True, alpha=.05, tail="two-sided",
                                          padjust='none', effsize="cohen", correction='auto')

    stats_pairwise_t["Sig."] = ["*" if row[[i for i in stats_pairwise_t.columns].index("p-unc") + 1] < .05
                                else " " for row in stats_pairwise_t.itertuples()]

    return stats_main, stats_pairwise_t


def generate_mean_plot(dataframe, error_bars="sd"):

    if error_bars == "sd" or error_bars == "std":
        error = dataframe["std"]
    if error_bars == "sem":
        error = [row.std / math.sqrt(row.count) for row in dataframe.itertuples()]
    if "ci" in error_bars or "CI" in error_bars:
        error = []
        for row in dataframe.itertuples():
            t_crit = stats.t.ppf(float("{}".format(float(error_bars.split("%")[0])/100)), row.count-1)
            error.append(row.std / math.sqrt(row.count) * t_crit)

    plt.title("Group means with {} error bars".format(error_bars))
    plt.bar(dataframe.index, dataframe["mean"], color='dodgerblue', edgecolor='black',
            yerr=error, capsize=4)


"""

# Checks normality for all columns in given wide-format DF
check_normality(dataframe=df1, transform=None)

# Generates histogram of all non-participant columns in wide-format DF
generate_histograms(df2a)

# Generates boxplot from long-format DF sorted by "by" column name
df1_long.boxplot(by="Location", column="Values")
df2a.boxplot(column=["Day", "Night"], cumulative=True)
df2b_long.boxplot(by="Day", column="Values")
df_intext.boxplot(by="Cohort", column='Values')

# Generates group means for descriptive format DF. Able to have "sd", "sem", or "x%ci" error bars
generate_mean_plot(df1_desc, "std")

# Friedman test + posthocs on Figure1 data
stats1_main, stats1_nemenyi, stats1_conover = calculate_friedman(dataframe=df1_long, usetest="friedman")

# Wilcoxon test for Figure2a data
stats2a = pg.wilcoxon(x=df2a["Day"], y=df2a["Night"], tail='two-sided')

# Friedman test + posthocs on Figure2b data
stats2b_main, stats2b_nemenyi, stats2b_conover = calculate_friedman(dataframe=df2b_long, usetest="friedman")

# Friedman test + pairwise unpaired 'non-parametric T-test' on In Text data
stats_intext_main, stats_intext_pairwise = calculate_kruskalwallish(dataframe=df_intext)

"""


"""
====================================================== GRAPHING =======================================================
"""


def plot_figure1(show_boxplot=False, max_y=None):
    plt.subplots(1, figsize=(10, 6))

    colors = ['red', 'orange', 'yellow', 'green', 'dodgerblue']
    line_x = -.25

    if show_boxplot:
        medianprops = dict(linewidth=2, color='red')

        plt.boxplot(df1[["Left Ankle", "Right Ankle", "Left Wrist", "Right Wrist", "Chest"]],
                    # labels=["Left Ankle", "Right Ankle", "Left Wrist", "Right Wrist", "Chest"],
                    positions=[0, 1, 2, 3, 4], medianprops=medianprops,
                    showfliers=False, manage_ticks=False, zorder=1)

    for i, colname, color in zip(np.arange(0, 5), df1.columns[1:], colors):
        plt.scatter([i for j in range(df1.shape[0])], df1[colname], color='grey', edgecolor='black', zorder=0)
        if not show_boxplot:
            plt.plot([line_x, line_x + .5], [df1[colname].median(), df1[colname].median()], 'k-', lw=2)

        line_x += 1
    plt.xlabel("Wear Location")
    plt.ylabel("Percent Non-Wear")
    if max_y is not None:
        plt.ylim(-5, max_y)
    plt.xticks([0, 1, 2, 3, 4], ["Left Ankle", "Right Ankle", "Left Wrist", "Right Wrist", "Chest"])


def plot_figure2a(label_subjs=False, show_boxplot=True, y_max=None):

    plt.subplots(1, figsize=(10, 6))

    if show_boxplot:
        medianprops = dict(linewidth=2, color='red')

        plt.boxplot(df2a[["Day", "Night"]], positions=[0, 1], medianprops=medianprops,
                    showfliers=False, manage_ticks=False, zorder=1)

    if not label_subjs:
        line_x = -.25
        for time, color in zip(["Day", "Night"], ['white', 'grey']):

            plt.scatter([time for i in range(df2a.shape[0])], df2a[time], color=color, edgecolor='black')

            if not show_boxplot:
                plt.plot([line_x, line_x + .5], [df2a[time].median(), df2a[time].median()], 'k-', lw=2)

            line_x += 1

    if label_subjs:
        for time, color in zip([0, 1], ['white', 'grey']):
            for i, val in enumerate(df2a.iloc[:, time + 1]):
                plt.text(time, val, s=int(df2a.loc[i]["Participant"]))

    plt.xticks([0, 1], ["Day", "Night"])
    plt.xlim(-1, 2)
    plt.xlabel("Time of Day")
    plt.ylabel("Percent Non-Wear")

    if y_max is not None:
        plt.ylim(-2, y_max)


def plot_figure2b(label_subjs=False, show_boxplot=True, y_max=None, greyscale=True):

    plt.subplots(1, figsize=(10, 6))

    line_x = -.25

    if show_boxplot:
        medianprops = dict(linewidth=2, color='red')

        plt.boxplot(df2b[["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6"]],
                    positions=[0, 1, 2, 3, 4, 5], medianprops=medianprops,
                    showfliers=False, manage_ticks=False, zorder=1)

    if not label_subjs:

        if not greyscale:
            colours = ['red', 'orange', 'yellow', 'green', 'dodgerblue', 'purple']
        if greyscale:
            colours = ['grey' for i in range(6)]

        for day, color in zip(df2b.columns[1:], colours):
            plt.scatter([day for i in range(df2b.shape[0])], df2b[day], color=color, edgecolor='black')

            if not show_boxplot:
                plt.plot([line_x, line_x + .5], [df2b[day].median(), df2b[day].median()], 'k-', lw=2)

            line_x += 1

    if label_subjs:
        for day, color in zip(np.arange(0, 7), ['red', 'orange', 'yellow', 'green', 'dodgerblue', 'purple']):
            for i, val in enumerate(df2b.iloc[:, day + 1]):
                plt.text(day, val, s=int(df2b.loc[i]["Participant"]))

    plt.xticks(np.arange(0, df2b.shape[1]-1), df2b.columns[1:])
    plt.xlabel("Day")
    plt.ylabel("Percent Non-Wear")

    if y_max is not None:
        plt.ylim(-3, y_max)
    plt.xlim(-.5, 6)


# plot_figure1(show_boxplot=False, max_y=None)
# plot_figure2a(label_subjs=False, show_boxplot=True, y_max=50)
# plot_figure2b(label_subjs=False, show_boxplot=True, y_max=25, greyscale=True)
