import os
import re
import sys

import pandas as pd
import signal_processing as spr
import numpy as np
import machine_learning as ml
from firelib.firelib import firefiles as ff
import PATHS as P
from pathlib import Path
import matplotlib.pyplot as plt


def concatenate_datasets():
    print()


def make_highest_features_dataset_from_complete_dataset(foi, complete_dataset, percentage=0.05, save=False):
    """
    Extract columns corresponding to features of interest from a complete dataset and saves/returns it.

    :param foi: the columns names of the features of interests
    :param complete_dataset: the complete dataset to extract the features from.
    :param percentage: for the title. Corresponding percentage for the highest features of interest.
    :return: dataframe of interest
    """
    df_foi = complete_dataset[[f for f in foi]]
    df_foi["status"] = complete_dataset["status"]
    if save:
        df_foi.to_csv(os.path.join(os.path.dirname(complete_dataset), f"highest {percentage * 100}% features - "
                                                                      f"{os.path.basename(complete_dataset)}"),
                      index=False)
    return df_foi


def make_raw_frequency_plots_from_pr_files(parent_dir, to_include=(), to_exclude=(), save=False, show=False,
                                           verbose=False):
    all_files = ff.get_all_files(os.path.join(parent_dir))
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
    columns = list(range(0, 300))
    dataset = pd.DataFrame(columns=columns)
    target = pd.DataFrame(columns=["status", ])

    n_processed_files = 0
    infected_organoids = []
    non_infected_organoids = []
    for f in files:
        print(f)
        organoid_key = os.path.basename(Path(f).parent.parent.parent.parent) + "_" + \
                       os.path.basename(Path(f).parent.parent) + "_" + os.path.basename(Path(f).parent)

        df = pd.read_csv(f)

        df_top = top_N_electrodes(df, 35, "TimeStamp [µs]")

        channels = df_top.columns

        fft_all_channels = pd.DataFrame()

        # fft of the signal
        for ch in channels[1:]:
            filtered = spr.butter_filter(df_top[ch], order=3, lowcut=50)
            clean_fft, clean_freqs = spr.fast_fourier(filtered, 10000)
            fft_all_channels[ch] = clean_fft
            fft_all_channels["Frequency [Hz]"] = clean_freqs
        # mean between the topped channels
        df_mean = merge_all_columns_to_mean(fft_all_channels, "Frequency [Hz]").round(3)
        downsampled_df = down_sample(df_mean["mean"], 300, 'mean')
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

    #
    #
    #
    #     # construct the dataset with n features
    #     dataset.loc[len(dataset)] = downsampled_df
    #
    #     path = Path(f)
    #     if "NI" in os.path.basename(path.parent.parent):
    #         target.loc[len(target)] = 0
    #     elif "INF" in os.path.basename(path.parent.parent):
    #         target.loc[len(target)] = 1

    # if verbose:
    #     progress = int(np.ceil(n_processed_files / len(files) * 100))
    #     sys.stdout.write(f"\rProgression of processing all_files: {progress}%")
    #     sys.stdout.flush()
    #     n_processed_files += 1


def make_dataset_from_freq_files(parent_directories, commentary="", targets=(), to_include=(), to_exclude=(), save=False, verbose=False):
    """
    Use frequency files of format two columns (one column 'Frequencies [Hz]' and one column 'mean') to generate a
    dataset used for classification.

    :param to_exclude:
    :param to_include:
    :param timepoint: The time point to study.
    :param title: name of the resulting dataset.
    :param parent_directories: list of the names of the parent directory that contains all files to make the dataset from.
    :return:

    Args:
        y_vector:
    """
    # todo : documentation

    freq_files = []
    for parent_dir in parent_directories:
        files = ff.get_all_files(os.path.join(parent_dir))

        for f in files:
            if all(i in f for i in to_include) and (not any(e in f for e in to_exclude)):
                if os.path.basename(Path(f).parent.parent) in targets:
                    freq_files.append(f)
    columns = list(range(0, 300))
    dataset = pd.DataFrame(columns=columns)
    target = pd.DataFrame(columns=["status", ])
    title = commentary + ""

    n_processed_files = 0
    target_id = 0
    targets_correspondence = {}

    for f in freq_files:
        df = pd.read_csv(f)
        # Downsampling by n
        downsampled_df = down_sample(df["mean"], 300, 'mean')

        # construct the dataset with n features
        dataset.loc[len(dataset)] = downsampled_df

        path = Path(f)
        target.loc[len(target)] = os.path.basename(path.parent.parent)

        if verbose:
            progress = int(np.ceil(n_processed_files / len(freq_files) * 100))
            sys.stdout.write(f"\rProgression of processing files: {progress}%")
            sys.stdout.flush()
            n_processed_files += 1
    dataset["status"] = target["status"]
    if verbose:
        print("\n")
    if save:
        dataset.to_csv(os.path.join(P.DATASETS, title+".csv"), index=False)
    return dataset


def make_filtered_sampled_freq_files():
    """
    make frequency files of format two columns (one column 'Frequencies [Hz]' and one column 'mean') from raw files.

    :return:
    """
    for timepoint in ("T=0MIN", "T=30MIN", "T=48H", "T=96H"):
        for rg27 in (P.RG27, P.NORG27):
            files = ff.get_all_files(os.path.join(rg27, timepoint))
            raw_files = []
            for f in files:
                if "pr_" in f and "TTX" not in f:
                    raw_files.append(f)

            for f in raw_files:
                print(f)
                df = pd.read_csv(f)
                df_top = top_N_electrodes(df, 35, "TimeStamp")
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


def make_filtered_numbered_freq_files(mono_time, top_n=35, truncate=30, n_features=300, lowcut=10):
    files = ff.get_all_files("E:\\Organoids\\four organoids per label\\")
    paths_pr = []
    columns = list(range(0, n_features))

    dataset = pd.DataFrame(columns=columns)
    identities = pd.DataFrame(columns=["organoid number", ])
    target = pd.DataFrame(columns=["status", ])
    for f in files:
        if "pr_" in f:
            paths_pr.append(f)
    print(paths_pr)
    for p in paths_pr:
        if p.split("\\")[3] == mono_time:
            print("path = ", p)
            df = pd.read_csv(p)
            # selecting top channels by their std

            df_top = top_N_electrodes(df, top_n, "TimeStamp")

            samples = equal_samples(df_top, truncate)
            channels = df_top.columns
            for df_s in samples:
                fft_all_channels = pd.DataFrame()

                # fft of the signal
                for ch in channels[1:]:
                    filtered = spr.butter_filter(df_s[ch], order=3, lowcut=lowcut)
                    clean_fft, clean_freqs = spr.fast_fourier(filtered, 10000)
                    fft_all_channels[ch] = clean_fft
                    fft_all_channels["frequency"] = clean_freqs
                # mean between the topped channels
                df_mean = merge_all_columns_to_mean(fft_all_channels, "frequency").round(3)

                # Downsampling by n
                downsampled_df = down_sample(df_mean["mean"], n_features, 'mean')

                # construct the dataset with n features
                dataset.loc[len(dataset)] = downsampled_df
                identities.loc[len(identities)] = p.split("\\")[5]
                if p.split("\\")[4] == "NI":
                    target.loc[len(target)] = 0
                elif p.split("\\")[4] == "INF":
                    target.loc[len(target)] = 1

    dataset.insert(loc=0, column="organoid number", value=identities["organoid number"])
    dataset["status"] = target["status"]
    folder = "Four organoids\\datasets\\"
    ff.verify_dir(folder)
    title = f"{folder}filtered_{lowcut}_numbered_frequency_top{str(top_n)}_nfeatures_{n_features}_{mono_time}.csv"
    dataset.to_csv(title, index=False)


def down_sample(data, n: int, mode: str):
    if len(data.index) > n:
        step = int(len(data.index) / n)
        lower_limit = 0
        upper_limit = step
        ds_data = []
        if mode == 'mean':
            while upper_limit <= len(data):
                ds_data.append(np.mean(data[lower_limit:upper_limit]).round(3))
                lower_limit = upper_limit
                upper_limit += step
        excedent = len(ds_data) - n
        ds_data = ds_data[:-excedent or None]
        return ds_data
    else:
        raise Exception("downsampling: length of data " + str(len(data.index)) + "< n " + str(n))


def equal_samples(df, n):
    step = int(len(df) / n)
    lower_limit = 0
    upper_limit = step
    samples = []
    while upper_limit <= len(df):
        samples.append(df[lower_limit:upper_limit])
        lower_limit = upper_limit
        upper_limit += step
    return samples


def make_freq_file(path, channels, file_path):
    """
    Make the fft of a temporal signal file and save it
    :param path: path of the temporal signal file
    :param channels: all the column to apply the fft to.
    :param file_path: path of the new file
    :return:
    """
    df = pd.read_csv(path)
    freq_df = pd.DataFrame()
    for channel in channels[1:]:
        clean_fft, clean_freq = spr.fast_fourier(df[channel], 10000)
        freq_df["frequency"] = clean_freq
        freq_df[channel] = clean_fft

    folder_path = os.path.dirname(file_path)
    isExist = os.path.exists(folder_path)
    if not isExist:
        os.makedirs(folder_path)
        freq_df.to_csv(file_path, index=False)
    else:
        freq_df.to_csv(file_path, index=False)


def clean_std_threshold(df, threshold):
    """
    generate a dataframe where some channels are omitted because of their too low standard deviation based on the
    threshold.

    :param df: dataframe in the frequencies domain.
    :param threshold: acceptable standard deviation compared to max and min std of all channels. Between 0 and 1.
    :return:
    """
    chans = []  # all the headers
    for col in df.columns:
        chans.append(col)

    # getting the std
    standards = {}
    for ch in chans:
        standards[ch] = np.std(df[ch])

    # keeping the channel or not
    min_key = min(standards, key=standards.get)
    min_value = standards.get(min_key)
    max_key = max(standards, key=standards.get)
    max_value = standards.get(max_key)
    limit = min_value + threshold * (max_value - min_value)
    clean_channels = []
    for key in standards:
        if standards.get(key) > limit:
            clean_channels.append(key)

    dfc = pd.DataFrame()  # cleaned dataframe
    dfc["frequency"] = df["frequency"]
    dfc[clean_channels] = df[clean_channels]

    return dfc
    # write new file
    # folder_path = "exp_may2021freq_std_clean_" + str(threshold) + "/" + path.split("/")[1]
    # isExist = os.path.exists(folder_path)
    # if not isExist:
    #     os.makedirs(folder_path)
    #     dfc.to_csv(folder_path + "/std_cleaned_freq.csv", index=False)
    # else:
    #     dfc.to_csv(folder_path + "/std_cleaned_freq.csv", index=False)


def merge_all_columns_to_mean(df: pd.DataFrame, except_column=""):
    excepted_column = pd.DataFrame()
    if except_column != "":
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


def top_N_electrodes(df, n, except_column):
    """
    keep the n electrodes with the highest std
    :param dfc: dataframe to filter
    :param n: number of electrodes to keep
    :return: filtered dataframe
    """
    for col in df.columns:
        if except_column in col:
            except_column = col
    dfc = df.drop(except_column, axis=1)
    df_filtered = pd.DataFrame()
    df_filtered[except_column] = df[except_column]

    all_std = []
    for c in dfc.columns:
        all_std.append(np.std(dfc[c]))
    top_indices = sorted(range(len(all_std)), key=lambda i: all_std[i], reverse=True)[:n]

    for c in dfc.columns:
        id = c.split(")")[0].split("=")[1]
        if int(id) in top_indices:
            df_filtered[c] = dfc[c]

    return df_filtered
