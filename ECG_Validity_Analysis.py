import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np


class EcgAnalysis:

    def __init__(self, counts_filename=None, intensity_filename=None):

        self.counts_filename = counts_filename
        self.intensity_filename = intensity_filename
        self.df_counts = None
        self.df_intensity = None

        self.wrist_data = None
        self.ankle_data = None
        self.wrist_anova = None
        self.ankle_anova = None

        # RUNS METHODS
        self.import_files()

    def import_files(self):

        self.df_counts = pd.read_excel(self.counts_filename)
        self.df_intensity = pd.read_excel(self.intensity_filename)

    def analyze_data(self, data_type, show_plot=True):

        if data_type == "counts":
            df = self.df_counts
            iv = "Counts"
            color_list = ["black", "dimgray", "gray", "darkgray", "slategray",
                          "silver", "lightgrey", "gainsboro", "whitesmoke", "white"]
            bar_width = self.df_counts.iloc[1][iv]-self.df_counts.iloc[0][iv]
            alignment="edge"

        if data_type == "intensity":
            df = self.df_intensity
            iv = "Intensity"
            color_list = ["lightgreen", "green", "darkred", "orange", "grey", "red"]
            bar_width = 1
            alignment = "center"

        wrist_anova = pg.rm_anova(data=df, dv="Wrist_Invalid", within=iv, subject="Subject",
                                  correction=True, detailed=True)
        ankle_anova = pg.rm_anova(data=df, dv="Ankle_Invalid", within=iv, subject="Subject",
                                  correction=True, detailed=True)

        wrist_data = df.groupby(by=[iv])["Wrist_Invalid"].describe()
        wrist_data["95%CI"] = wrist_data["std"] / np.sqrt(wrist_data["count"]) * \
                              scipy.stats.t.ppf(.95, wrist_data["count"]-1)
        ankle_data = df.groupby(by=[iv])["Ankle_Invalid"].describe()
        ankle_data["95%CI"] = ankle_data["std"] / np.sqrt(ankle_data["count"]) * \
                              scipy.stats.t.ppf(.95, ankle_data["count"]-1)

        if show_plot:
            plt.subplots(1, 2, figsize=(10, 6))

            plt.subplot(1, 2, 1)
            plt.subplots_adjust(bottom=.2)
            plt.bar([i for i in wrist_data.index], wrist_data["mean"],
                    color=color_list, edgecolor='black', width=bar_width, align=alignment,
                    yerr=wrist_data["95%CI"], capsize=8, ecolor='black')
            plt.ylabel("% Invalid")
            plt.title("Wrist (n={})".format(len(set(df["Subject"]))))
            plt.ylim(0, 100)
            plt.xlabel(iv)
            plt.xticks(rotation=45)

            plt.subplot(1, 2, 2)
            plt.subplots_adjust(bottom=.2)
            plt.title("Ankle (n={})".format(len(set(df["Subject"]))))
            plt.bar([i for i in ankle_data.index], ankle_data["mean"], width=bar_width, align=alignment,
                    color=color_list, edgecolor='black',
                    yerr=ankle_data["95%CI"], capsize=8, ecolor='black')
            plt.ylim(0, 100)
            plt.xlabel(iv)
            plt.xticks(rotation=45)

        self.wrist_data = wrist_data
        self.ankle_data = ankle_data
        self.wrist_anova = wrist_anova
        self.ankle_anova = ankle_anova

    def analyze_subjects(self, plot_means=False):

        icon_dict = {"Sedentary": "o", "Light": "^", "Moderate": "s", "Vigorous": "P"}
        color_dict = {"Sedentary": "green", "Light": "yellow", "Moderate": "orange", "Vigorous": "red"}
        subjs = sorted(set(self.df_intensity["Subject"]))

        plt.subplots(1, 2, figsize=(10, 6))
        plt.suptitle("Invalid Data by Participant")

        plt.subplot(1, 2, 1)
        plt.ylabel("% Invalid")
        plt.xlabel("ID")
        plt.title("Wrist Data")

        plt.subplot(1, 2, 2)
        plt.xlabel("ID")
        plt.title("Ankle Data")

        for subj in subjs:
            for intensity in ["Sedentary", "Light", "Moderate", "Vigorous"]:
                df = self.df_intensity.loc[(self.df_intensity["Intensity"] == intensity) &
                                           (self.df_intensity["Subject"] == subj)]

                plt.subplot(1, 2, 1)
                plt.scatter(x=str(subj), y=df["Wrist_Invalid"],
                            marker=icon_dict[intensity], edgecolor="black", color=color_dict[intensity])

                if subj == subjs[-1]:
                    plt.legend(labels=["Sedentary", "Light", "Moderate", "Vigorous"])

                plt.subplot(1, 2, 2)
                plt.scatter(x=str(subj), y=df["Ankle_Invalid"],
                            marker=icon_dict[intensity], edgecolor='black', color=color_dict[intensity])
                if subj == subjs[-1]:
                    plt.legend(labels=["Sedentary", "Light", "Moderate", "Vigorous"])

        if plot_means:

            for intensity in ["Sedentary", "Light", "Moderate", "Vigorous"]:
                if self.wrist_data is not None:
                    plt.subplot(1, 2, 1)
                    plt.axhline(y=self.wrist_data["mean"].loc[intensity],
                                color=color_dict[intensity], linestyle='dashed')

                if self.ankle_data is not None:
                    plt.subplot(1, 2, 2)
                    plt.axhline(y=self.ankle_data["mean"].loc[intensity],
                                color=color_dict[intensity], linestyle='dashed')

    def plot_activity_validity(self, intensity="All Activity"):

        color_dict = {"All Activity": "lightgreen", "Light": "green", "MVPA": "darkred",
                      "Moderate": "orange", "Sedentary": "grey", "Vigorous": "red"}

        plt.close('all')

        plt.subplots(1, 2, figsize=(10, 6))
        plt.suptitle("{} vs. Percent Invalid ECG".format(intensity))

        plt.subplot(1, 2, 1)
        plt.scatter(self.df_intensity.loc[self.df_intensity["Intensity"] == intensity]["Wrist_Percent_Epochs"],
                    self.df_intensity.loc[self.df_intensity["Intensity"] == intensity]["Wrist_Invalid"],
                    edgecolor='black', color=color_dict[intensity])
        plt.ylim(0, 100)
        plt.xlim(0, )
        plt.ylabel("% Invalid")
        plt.xlabel("% {}".format(intensity))

        wrist_r = scipy.stats.pearsonr(self.df_intensity.loc[self.df_intensity["Intensity"]
                                                             == intensity]["Wrist_Percent_Epochs"],
                                       self.df_intensity.loc[self.df_intensity["Intensity"] ==
                                                             intensity]["Wrist_Invalid"])
        plt.title("Wrist Data (r={})".format(round(wrist_r[0], 3)))

        plt.subplot(1, 2, 2)
        plt.scatter(self.df_intensity.loc[self.df_intensity["Intensity"] == intensity]["Ankle_Percent_Epochs"],
                    self.df_intensity.loc[self.df_intensity["Intensity"] == intensity]["Ankle_Invalid"],
                    edgecolor='black', color=color_dict[intensity])
        plt.ylim(0, 100)
        plt.xlim(0, )
        plt.xlabel("% {}".format(intensity))

        ankle_r = scipy.stats.pearsonr(self.df_intensity.loc[self.df_intensity["Intensity"]
                                                             == intensity]["Ankle_Percent_Epochs"],
                                       self.df_intensity.loc[self.df_intensity["Intensity"] ==
                                                             intensity]["Ankle_Invalid"])
        plt.title("Ankle Data (r={})".format(round(ankle_r[0], 3)))


x = EcgAnalysis(counts_filename="/Users/kyleweber/Desktop/Counts ECG Data/All.xlsx",
                intensity_filename="/Users/kyleweber/Desktop/Intensity ECG Data/All.xlsx")
x.analyze_data(data_type="intensity", show_plot=True)
# x.analyze_subjects(plot_means=True)
x.plot_activity_validity("MVPA")
