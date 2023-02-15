import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import PATHS as P
import data_processing as dpr
import signal_processing as spr
import fiiireflyyy.firelearn as fl
import fiiireflyyy.firefiles as ff
import fiiireflyyy.fireprocess as fp


def fig_smoothened_frequencies_0_5000Hz_Mock_Tahv_rg27(batch):
    show = False
    percentiles = 0.1
    n_components = 2
    batches = {"batch 1": ["1", "2", "3"], "batch 2": ["4", "5", "6", ], "batch 3": ["7", "8", "9", ],
               "batch 12": ["1", "2", "3", "4", "5", "6", ],
               "all slices": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]}
    min_freq = 0
    max_freq = 5000
    tahvni = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                             to_include=("freq_50hz_sample", "T=48H"),
                                             to_exclude=("TTX", "RG27",),
                                             target_keys={"NI": "NI", "TAHV": "TAHV"},
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment="")

    discarded_tahvni = fp.discard_outliers_by_iqr(tahvni, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    tahv_rg27 = fp.make_dataset_from_freq_files(parent_dir=P.RG27,
                                                to_include=("freq_50hz_sample", "T=48H", "RG27"),
                                                to_exclude=("TTX", "NI"),
                                                target_keys={"NI": "NI", "TAHV": "TAHV"},
                                                verbose=False,
                                                save=False,
                                                freq_range=(min_freq, max_freq),
                                                select_samples=batches[batch],
                                                separate_samples=False,
                                                label_comment="")

    discarded_tahv_rg27 = fp.discard_outliers_by_iqr(tahv_rg27, low_percentile=percentiles,
                                                     high_percentile=1 - percentiles,
                                                     mode='capping')

    discarded_tahvni.replace("NI", "Mock", inplace=True)
    discarded_tahvni.replace("TAHV", "Tahv", inplace=True)
    discarded_tahv_rg27.replace("TAHV", "RG27-treated Tahv", inplace=True)

    global_df = pd.concat([discarded_tahvni, discarded_tahv_rg27], ignore_index=True)

    plt.figure(figsize=(8, 8))
    plt.plot(global_df.loc[global_df["label"] == "Mock"].mean(axis=0), label="Mock", linewidth=1, color='g')
    plt.fill_between([x for x in range(0, 300)],
                     global_df.loc[global_df["label"] == "Mock"].mean(axis=0).subtract(
                         global_df.loc[global_df["label"] == "Mock"].std(axis=0)),
                     global_df.loc[global_df["label"] == "Mock"].mean(axis=0).add(
                         global_df.loc[global_df["label"] == "Mock"].std(axis=0)),
                     color='g', alpha=.5)
    plt.plot(global_df.loc[global_df["label"] == "Tahv"].mean(axis=0), label="Tahv", linewidth=1, color='b')
    plt.fill_between([x for x in range(0, 300)],
                     global_df.loc[global_df["label"] == "Tahv"].mean(axis=0).subtract(
                         global_df.loc[global_df["label"] == "Tahv"].std(axis=0)),
                     global_df.loc[global_df["label"] == "Tahv"].mean(axis=0).add(
                         global_df.loc[global_df["label"] == "Tahv"].std(axis=0)),
                     color='b', alpha=.5)
    plt.plot(global_df.loc[global_df["label"] == "RG27-treated Tahv"].mean(axis=0),
             label="RG27-treated Tahv", linewidth=1, color='r')
    plt.fill_between([x for x in range(0, 300)],
                     global_df.loc[global_df["label"] == "RG27-treated Tahv"].mean(axis=0).subtract(
                         global_df.loc[global_df["label"] == "RG27-treated Tahv"].std(axis=0)),
                     global_df.loc[global_df["label"] == "RG27-treated Tahv"].mean(axis=0).add(
                         global_df.loc[global_df["label"] == "RG27-treated Tahv"].std(axis=0)),
                     color='r', alpha=.5)
    ratio = (max_freq - min_freq) / 300
    x_ds = [x for x in range(0, 301, 15)]
    x_freq = [int(x * ratio + min_freq) for x in x_ds]
    plt.xticks(x_ds, x_freq, rotation=45)
    plt.xlabel("Smoothened frequencies [Hz]", fontsize=25)
    plt.ylabel("Amplitude [pV]", fontsize=25)
    plt.legend(prop={'size': 20})
    plt.savefig(os.path.join(P.RESULTS,
                             f"Fig Smoothened frequencies Mock-Tahv-RG27 on {min_freq}-{max_freq}Hz {batch}.png"),
                dpi=1200)
    if show:
        plt.show()


def fig_PCA_0_5000Hz_on_batch_for_Mock_Tahv_applied_on_rg27(batch):
    percentiles = 0.1
    n_components = 2
    batches = {"batch 1": ["1", "2", "3"], "batch 2": ["4", "5", "6", ], "batch 3": ["7", "8", "9", ],
               "batch 12": ["1", "2", "3", "4", "5", "6", ],
               "all slices": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]}
    min_freq = 0
    max_freq = 5000
    tahvni = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                             to_include=("freq_50hz_sample", "T=48H"),
                                             to_exclude=("TTX", "RG27",),
                                             target_keys={"NI": "NI", "TAHV": "TAHV"},
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment="")

    discarded_tahvni = fp.discard_outliers_by_iqr(tahvni, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    tahv_rg27 = fp.make_dataset_from_freq_files(parent_dir=P.RG27,
                                                to_include=("freq_50hz_sample", "T=48H", "RG27"),
                                                to_exclude=("TTX", "NI"),
                                                target_keys={"NI": "NI", "TAHV": "TAHV"},
                                                verbose=False,
                                                save=False,
                                                freq_range=(min_freq, max_freq),
                                                select_samples=batches[batch],
                                                separate_samples=False,
                                                label_comment="")

    discarded_tahv_rg27 = fp.discard_outliers_by_iqr(tahv_rg27, low_percentile=percentiles,
                                                     high_percentile=1 - percentiles,
                                                     mode='capping')

    discarded_tahvni.replace("NI", "Mock", inplace=True)
    discarded_tahvni.replace("TAHV", "Tahv", inplace=True)
    discarded_tahv_rg27.replace("TAHV", "RG27-treated Tahv", inplace=True)

    pca, pcdf, ratios = fl.fit_pca(discarded_tahvni, n_components=n_components)
    rg27_pcdf = fl.apply_pca(pca, discarded_tahv_rg27)
    global_df = pd.concat([pcdf, rg27_pcdf], ignore_index=True)
    rounded_ratio = [round(r * 100, 1) for r in ratios]
    fl.plot_pca(global_df, n_components=n_components, show=False,
                title=f"Fig PCA on {min_freq}-{max_freq}Hz {batch} for Mock,Tahv applied on rg27-treated Tahv",
                points=True, metrics=True, savedir=P.RESULTS, ratios=rounded_ratio)
