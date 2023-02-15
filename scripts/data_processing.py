import os
import re
import sys

import pandas as pd
import signal_processing as spr
import numpy as np
import machine_learning as ml
import fiiireflyyy.firelearn as fl
import fiiireflyyy.firefiles as ff
from pathlib import Path
import matplotlib.pyplot as plt
import PATHS as P




def make_highest_features_dataset_from_complete_dataset(foi, df, percentage=0.05, save=False):
    """
    make_highest_features_dataset_from_complete_dataset(foi, df, percentage=0.05, save=False):

        Considering a DataFrame dataset, with for each column a feature, returns
        a dataframe with only a selection of columns.

        Parameters
        ----------
        foi: list of int
            features of interest.
        df: DataFrame
            the dataset to extract columns from.
        percentage: float, optional, default: 0.05
            for the name of the dataset. The percentage of features that are kept.
        save: bool, optional, default: False
            Whether to save the resulting dataframe.

        Returns
        -------
        out: DataFrame
            reduced dataframe columns-wise.
    """
    df_foi = df[[f for f in foi]]
    df_foi["status"] = df["status"]
    if save:
        df_foi.to_csv(os.path.join(os.path.dirname(df), f"highest {percentage * 100}% features - "
                                                        f"{os.path.basename(df)}"),
                      index=False)
    return df_foi


def make_raw_frequency_plots_from_pr_files(parent_dir, to_include, to_exclude=(), save=False, show=False,
                                           verbose=False):
    """
    make_raw_frequency_plots_from_pr_files(parent_dir, to_include, to_exclude=(), save=False, show=False,
                                           verbose=False):

        plot the amplitude depending on the frequencies from pre-processed
        files.

        Parameters
        ----------
        parent_dir: 'str'
            the oldest parent from which we will parse files to extract data.
        to_include : tuple of str
            Allow to select children paths of parent_dir. All the
            elements in to_include must be present in the path for it to be
            selected.
        to_exclude : tuple of str, optional, default: ()
            Allow to select children paths of parent_dir. A path
            will not be selected if any element of to_exclude is present
            in the path.
        save: bool, optional, default: False
            Whether to save or not the resulting figure.
        show: bool, optional, default: False
            Whether to show or not the resulting figure.
        verbose: bool, optional, default: False
            Whether to display or not more processing information in the console.

        Returns
        -------
        out: plt.Axes
            the resulting figure
    """
    all_files = ff.get_all_files(parent_dir)
    files = []
    organoids = []
    for f in all_files:
        if all(i in f for i in to_include) and (not any(e in f for e in to_exclude)):
            files.append(f)
            organoid_key = os.path.basename(Path(f).parent.parent.parent.parent) + "_" + \
                           os.path.basename(Path(f).parent.parent) + "_" + os.path.basename(Path(f).parent)
            if organoid_key not in organoids:
                organoids.append(organoid_key)  # for parent: P.NOSTACHEL ==> - StachelINF2
            if verbose:
                print("added: ", f)
    number_of_organoids = len(organoids)
    print(number_of_organoids, organoids)

    infected_organoids = []
    non_infected_organoids = []
    for f in files:
        print(f)
        organoid_key = os.path.basename(Path(f).parent.parent.parent.parent) + "_" + \
                       os.path.basename(Path(f).parent.parent) + "_" + os.path.basename(Path(f).parent)

        df = pd.read_csv(f)
        df_top = top_n_electrodes(df, 35, "TimeStamp [µs]")
        channels = df_top.columns
        fft_all_channels = pd.DataFrame()

        for ch in channels[1:]:
            filtered = spr.butter_filter(df_top[ch], order=3, lowcut=50)
            clean_fft, clean_freqs = spr.fast_fourier(filtered, 10000)
            fft_all_channels[ch] = clean_fft
            fft_all_channels["Frequency [Hz]"] = clean_freqs

        df_mean = merge_all_columns_to_mean(fft_all_channels, "Frequency [Hz]").round(3)
        downsampled_df = smoothing(df_mean["mean"], 300, 'mean')
        if "TAHV" in organoid_key:
            infected_organoids.append(downsampled_df)
            print("added infected: ", organoid_key, len(downsampled_df))
        if "NI" in organoid_key:
            non_infected_organoids.append(downsampled_df)
            print("added not infected: ", organoid_key, len(downsampled_df))

    fig, ax = plt.subplots()
    non_infected_arrays = [np.array(x) for x in non_infected_organoids]
    mean_non_infected = [np.mean(k) for k in zip(*non_infected_arrays)]
    std_non_infected = [np.std(k) for k in zip(*non_infected_arrays)]
    low_std_non_infected = [mean_non_infected[x] - std_non_infected[x] for x in range(len(mean_non_infected))]
    high_std_non_infected = [mean_non_infected[x] + std_non_infected[x] for x in range(len(mean_non_infected))]
    ax.plot([int(x * 5000 / 300) for x in range(0, 300)], mean_non_infected, color='blue', label='not infected')
    ax.fill_between(x=[int(x * 5000 / 300) for x in range(0, 300)], y1=low_std_non_infected, y2=high_std_non_infected,
                    color='blue', alpha=.5)

    infected_arrays = [np.array(x) for x in infected_organoids]
    mean_infected = [np.mean(k) for k in zip(*infected_arrays)]
    std_infected = [np.std(k) for k in zip(*infected_arrays)]
    low_std_infected = [mean_infected[x] - std_infected[x] for x in range(len(mean_infected))]
    high_std_infected = [mean_infected[x] + std_infected[x] for x in range(len(mean_infected))]
    ax.plot([int(x * 5000 / 300) for x in range(0, 300)], mean_infected, color='red', label='infected')
    ax.fill_between(x=[int(x * 5000 / 300) for x in range(0, 300)], y1=low_std_infected, y2=high_std_infected,
                    color='red', alpha=.5)

    recording_time = ""
    for t in to_include:
        if "T=" in t:
            recording_time = t
    rg27 = "-RG27"
    if "+RG27" in parent_dir:
        rg27 = "+RG27"

    title = f"Smoothened frequencies for all organoids {rg27} at {recording_time} "
    plt.title(title)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [pV]")
    plt.legend()
    if save:
        plt.savefig(os.path.join(P.RESULTS, title + ".png"), dpi=1000)
    if show:
        plt.show()
    return ax



def make_filtered_sampled_freq_files(f):
    """
    make_filtered_sampled_freq_files():

        make frequency files of format two columns (one column 'Frequencies [Hz]'
        and one column 'mean') from raw files. Save every sample as a different
        file. Without parameters, this function is a process by itself.

        Parameters
        ----------
        f: str
            The path of the Dataframe to transform.
        Returns
        -------
    """
    print(f)
    df = pd.read_csv(f)
    df_top = top_n_electrodes(df, 35, "TimeStamp")
    samples = equal_samples(df_top, 30)
    channels = df_top.columns
    n_sample = 0
    for df_s in samples:
        fft_all_channels = pd.DataFrame(columns=["Frequency [Hz]", "mean"])

        # fft of the signal
        for ch in channels[1:]:
            filtered = spr.butter_filter(df_s[ch], order=3, lowcut=50)
            clean_fft, clean_freqs = spr.fast_fourier(filtered, 10000)
            fft_all_channels["Frequency [Hz]"] = clean_freqs
            fft_all_channels[ch] = clean_fft

        # mean between the topped channels
        df_mean = merge_all_columns_to_mean(fft_all_channels, "Frequency [Hz]").round(3)

        id = os.path.basename(f).split("_")[1]
        df_mean.to_csv(os.path.join(os.path.dirname(f), f"freq_50hz_sample{n_sample}_{id}.csv"),
                       index=False)
        n_sample += 1






def dataframe_to_frequencies(df, file_path="", sf=10000):
    """
    dataframe_to_frequencies(df, file_path="", sf=10000):

        Applies a fast fourier transform to all the columns of a dataframe
        except the first one.

        Parameters
        ----------
        df: DataFrame
            the data in temporal domain. The first column will be
            considered as the time axis.
        file_path: str, optional, default: ""
            If file_path is not empty, then it is the name under
            which the frequency dataframe will be saved.
        sf: int, optional, default: 10000
            sampling frequency, in Hertz.

        Returns
        -------
        out: DataFrame
            the dataframe in frequencies domain.

    """
    freq_df = pd.DataFrame()
    channels = df.columns
    for channel in channels[1:]:
        clean_fft, clean_freq = spr.fast_fourier(df[channel], sf)
        freq_df["Frequency [Hz]"] = clean_freq
        freq_df[channel] = clean_fft

    folder_path = os.path.dirname(file_path)
    isExist = os.path.exists(folder_path)
    if file_path:
        if not isExist:
            os.makedirs(folder_path)
            freq_df.to_csv(os.path.join(P.DATASETS, file_path), index=False)
        else:
            freq_df.to_csv(os.path.join(P.DATASETS, file_path), index=False)
    return freq_df


def merge_all_columns_to_mean(df: pd.DataFrame, except_column=""):
    """
    merge_all_columns_to_mean(df: pd.DataFrame, except_column=""):

        average all the columns, except an optional specified one,
        in a dataframe into one. The average is done row-wise.

        Parameters
        ----------
        df: DataFrame
            the dataframe to average
        except_column: str, optional, default: ""
            the name of the column to exclude from the average.
            Will be included in the resulting dataset.

        Returns
        --------
        out: DataFrame
            Dataframe containing on column labeled 'mean', and
            an optional second column based on the
            except_column parameter
    """

    excepted_column = pd.DataFrame()
    if except_column:
        for col in df.columns:
            if except_column in col:
                except_column = col
        excepted_column = df[except_column]
        df.drop(except_column, axis=1, inplace=True)

    df_mean = pd.DataFrame(columns=["mean", ])
    df_mean['mean'] = df.mean(axis=1)

    if except_column != "":
        for col in df.columns:
            if except_column in col:
                except_column = col
        df_mean[except_column] = excepted_column

    return df_mean


def top_n_electrodes(df, n, except_column="TimeStamp [µs]"):
    """
        top_N_electrodes(df, n, except_column):

            Select only the n electrodes with the highest standard
            deviation, symbolizing the highest activity.

            Parameters
            ----------
            df: Dataframe
                Contains the data. If using MEA it has the following
                formatting: the first column contains the time dimension,
                names 'TimeStamp [µs]', while each other represent the
                different electrodes of the MEA, with names going as
                '48 (ID=1) [pV]' or similar.
            n: int
                the number of channels to keep, sorted by the highest
                standard deviation.
            except_column: str, optional, default: 'TimeStamp [µs]'
                The name of a column to exclude of the selection.
                This column will be included in the resulting
                dataframe.

            Returns
            -------
            out: pandas Dataframe
                Dataframe containing only n columns, corresponding
                to the n channels with the highest standard deviation.
                If except_column exists, then this very column is added
                untouched from the original dataframe to the resulting
                one.

            Notes
            -----
            This function only use the standard deviation as metric to
            use to sort the channels. Any modification on this metric
            should be done on the line indicated below.

            Examples
            --------
            >> df = pd.read_csv(file)
            >> df_top = dpr.top_N_electrodes(df=df, n=35, except_column='TimaStamp [µs]")
            Returns a dataframe containing the top 35 channels based on the std of the signal

    """
    # getting the complete name of the column to exclude, in case of slight fluctuation in name
    if except_column:
        for col in df.columns:
            if except_column in col:
                except_column = col

    # managing 'except_column'
    dfc = df.drop(except_column, axis=1)
    df_filtered = pd.DataFrame()
    df_filtered[except_column] = df[except_column]

    # getting the top channels by metric. Metric changes should be done here.
    all_metric = []
    for c in dfc.columns:
        all_metric.append(np.std(dfc[c]))
    top_indices = sorted(range(len(all_metric)), key=lambda i: all_metric[i], reverse=True)[:n]

    # creating resulting dataframe
    for c in dfc.columns:
        id = c.split(")")[0].split("=")[1]
        if int(id) in top_indices:
            df_filtered[c] = dfc[c]

    return df_filtered
