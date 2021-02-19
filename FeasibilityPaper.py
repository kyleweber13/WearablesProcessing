import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
import scipy.stats as stats
from scikit_posthocs import posthoc_nemenyi_friedman as nemenyi
from scikit_posthocs import posthoc_conover_friedman as conover
import math

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
