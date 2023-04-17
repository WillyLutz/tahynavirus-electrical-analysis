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
import numpy as np


def Confusion_matrix(min_freq=0, max_freq=5000, batch="batch 2", timepoint="T=48H"):
    show = False
    percentiles = 0.1
    batches = {"batch 1": ["1", "2", "3"], "batch 2": ["4", "5", "6", ], "batch 3": ["7", "8", "9", ],
               "batch 4": ["10", "11", "12"],
               "batch 1_2": ["1", "2", "3", "4", "5", "6", ], "batch 1_4": ["1", "2", "3", "10", "11", "12", ],
               "batch 2_4": ["4", "5", "6", "10", "11", "12"],
               "all slices": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]}

    class1 = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                             to_include=("freq_50hz_sample", timepoint),
                                             to_exclude=("TTX", "RG27", "TAHV"),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment=f"",
                                             target_keys={'NI': 'Mock', 'TAHV': 'Tahv'})

    discarded_class1 = fp.discard_outliers_by_iqr(class1, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    class2 = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                             to_include=("freq_50hz_sample", timepoint),
                                             to_exclude=("TTX", "RG27", "NI"),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment=f"",
                                             target_keys={'NI': 'Mock', 'TAHV': 'Tahv'})

    discarded_class2 = fp.discard_outliers_by_iqr(class2, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    class3 = fp.make_dataset_from_freq_files(parent_dir=P.RG27,
                                             to_include=("freq_50hz_sample", timepoint),
                                             to_exclude=("TTX", "NI",),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment=f"",
                                             target_keys={'NI': 'NI', 'TAHV': 'RG27-treated\nTahv'})

    discarded_class3 = fp.discard_outliers_by_iqr(class3, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    rfc, _ = fl.train_RFC_from_dataset(pd.concat([discarded_class1, discarded_class2], ignore_index=True))

    global_df = pd.concat([discarded_class1, discarded_class2, discarded_class3, ], ignore_index=True)
    savepath = os.path.join(P.RESULTS, batch)
    ff.verify_dir(savepath)
    fl.test_model_by_confusion(rfc, global_df, training_targets=(f'Mock', f'Tahv'),
                               testing_targets=tuple(set(list((
                                   f'Mock', f'Tahv', f'RG27-treated\nTahv',)))),
                               show=show, verbose=False, savepath=savepath,
                               title=f"Confusion matrix train on {timepoint} Tahv,Mock, test on Tahv,Mock,RG27 for {batch} "
                                     f"{min_freq}-{max_freq}Hz",
                               iterations=5, )


def Smoothened_frequencies(min_freq=0, max_freq=5000, batch="all slices", timepoint="T=48H"):
    batches = {"batch 1": ["1", "2", "3"], "batch 2": ["4", "5", "6", ], "batch 3": ["7", "8", "9", ],
               "batch 4": ["10", "11", "12"],
               "batch 1_2": ["1", "2", "3", "4", "5", "6", ], "batch 1_4": ["1", "2", "3", "10", "11", "12", ],
               "batch 2_4": ["4", "5", "6", "10", "11", "12"],
               "all slices": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]}

    percentiles = 0.1
    show = False
    class1 = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                             to_include=("freq_50hz_sample", timepoint),
                                             to_exclude=("TTX", "RG27", "NI"),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment="",
                                             target_keys={'NI': 'Mock', 'TAHV': 'Tahv'})

    discarded_class1 = fp.discard_outliers_by_iqr(class1, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')
    class2 = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                             to_include=("freq_50hz_sample", timepoint),
                                             to_exclude=("TTX", "RG27", "TAHV"),
                                             verbose=False,
                                             freq_range=(min_freq, max_freq),
                                             save=False,
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment="",
                                             target_keys={'NI': 'Mock', 'TAHV': 'Tahv'})

    discarded_class2 = fp.discard_outliers_by_iqr(class2, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    class3 = fp.make_dataset_from_freq_files(parent_dir=P.RG27,
                                             to_include=("freq_50hz_sample", timepoint,),
                                             to_exclude=("TTX", "NI"),
                                             verbose=False,
                                             freq_range=(min_freq, max_freq),
                                             save=False,
                                             separate_samples=False,
                                             select_sample=batches[batch],
                                             label_comment="",
                                             target_keys={'NI': 'RG27-treated Mock', 'TAHV': 'RG27-treated Tahv'}
                                             )
    discarded_class3 = fp.discard_outliers_by_iqr(class3, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    global_df = pd.concat([discarded_class1, discarded_class3, discarded_class2], ignore_index=True)

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
    savepath = os.path.join(P.RESULTS, batch)
    ff.verify_dir(savepath)
    plt.savefig(os.path.join(savepath,
                             f"Smoothened frequencies Mock-Tahv-RG27 on {min_freq}-{max_freq}Hz {batch} at {timepoint}.png"),
                dpi=1200)
    if show:
        plt.show()


def PCA_fit_on_two_test_on_one(min_freq=0, max_freq=5000, batch="all slices",
                               timepoint="T=48H"):
    show = False
    percentiles = 0.1
    n_components = 2
    batches = {"batch 1": ["1", "2", "3"], "batch 2": ["4", "5", "6", ], "batch 3": ["7", "8", "9", ],
               "batch 4": ["10", "11", "12"],
               "batch 1_2": ["1", "2", "3", "4", "5", "6", ], "batch 1_4": ["1", "2", "3", "10", "11", "12", ],
               "batch 2_4": ["4", "5", "6", "10", "11", "12"],
               "all slices": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]}

    class1 = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                             to_include=("freq_50hz_sample", timepoint),
                                             to_exclude=("TTX", "RG27",),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment="",
                                             target_keys={'NI': 'Mock', 'TAHV': 'Tahv'})
    discarded_class1 = fp.discard_outliers_by_iqr(class1, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    class2 = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                             to_include=("freq_50hz_sample", timepoint),
                                             to_exclude=("TTX", "NI",),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment="",
                                             target_keys={'NI': 'Mock', 'TAHV': 'Tahv'})
    discarded_class2 = fp.discard_outliers_by_iqr(class2, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')
    class3 = fp.make_dataset_from_freq_files(parent_dir=P.RG27,
                                             to_include=("freq_50hz_sample", timepoint),
                                             to_exclude=("TTX", "NI",),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment="",
                                             target_keys={'NI': 'RG27-treated Mock', 'TAHV': 'RG27-treated Tahv'})
    discarded_class3 = fp.discard_outliers_by_iqr(class3, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    pca, pcdf, ratios = fl.fit_pca(pd.concat([discarded_class1, discarded_class2], ignore_index=True), n_components=n_components)
    class3_pcdf = fl.apply_pca(pca, discarded_class3)
    global_df = pd.concat([pcdf, class3_pcdf], ignore_index=True)

    rounded_ratio = [round(r * 100, 1) for r in ratios]
    fl.plot_pca(global_df, n_components=n_components, show=show,
                title=f"PCA on {min_freq}-{max_freq}Hz {batch} for Mock,Tahv, applied on RG27 at {timepoint}",
                points=True, metrics=True, savedir=os.path.join(P.RESULTS, batch), ratios=rounded_ratio)


def Amplitude_for_Mock_CoV_in_region_Hz_at_T_24H_for_all_organoids(min_freq=0, max_freq=5000, batch="all slices",
                                                                   timepoint="T=48H"):
    show = False
    percentiles = 0.1
    batches = {"batch 1": ["1", "2", "3"], "batch 2": ["4", "5", "6", ], "batch 3": ["7", "8", "9", ],
               "batch 4": ["10", "11", "12"],
               "batch 1_2": ["1", "2", "3", "4", "5", "6", ], "batch 1_4": ["1", "2", "3", "10", "11", "12", ],
               "batch 2_4": ["4", "5", "6", "10", "11", "12"],
               "all slices": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]}

    cov = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                          to_include=("freq_50hz_sample", timepoint),
                                          to_exclude=("TTX", "RG27", "NI"),
                                          verbose=False,
                                          save=False,
                                          select_samples=batches[batch],
                                          separate_samples=False,
                                          freq_range=(min_freq, max_freq),
                                          label_comment="",
                                          target_keys={'NI': 'NI', 'TAHV': 'TAHV'})

    discarded_cov = fp.discard_outliers_by_iqr(cov, low_percentile=percentiles,
                                               high_percentile=1 - percentiles,
                                               mode='capping')

    ni = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                         to_include=("freq_50hz_sample", timepoint),
                                         to_exclude=("TTX", "RG27", "TAHV"),
                                         verbose=False,
                                         save=False,
                                         select_samples=batches[batch],
                                         freq_range=(min_freq, max_freq),
                                         separate_samples=False,
                                         label_comment="",
                                         target_keys={'NI': 'NI', 'TAHV': 'TAHV'})

    discarded_ni = fp.discard_outliers_by_iqr(ni, low_percentile=percentiles,
                                              high_percentile=1 - percentiles,
                                              mode='capping')

    discarded_ni.replace("NI", "Mock", inplace=True)
    discarded_cov.replace("TAHV", "Tahv", inplace=True)
    global_df = pd.concat([discarded_ni, discarded_cov], ignore_index=True)

    data = pd.DataFrame(columns=["label", "mean amplitude [pV]", "std amplitude [pV]"])
    plt.figure(figsize=(9, 8))
    mock_df = global_df.loc[global_df["label"] == "Mock"]
    mock_region = mock_df.loc[:, mock_df.columns != "label"]
    plt.bar(0, np.mean(np.array(mock_region)), color='dimgray',
            yerr=np.std(np.array(mock_region)))
    data.loc[len(data)] = ["Mock", np.mean(np.array(mock_region)), np.std(np.array(mock_region))]

    cov_df = global_df.loc[global_df["label"] == "Tahv"]
    cov_region = cov_df.loc[:, cov_df.columns != "label"]
    plt.bar(1, np.mean(np.array(cov_region)), color='darkgray',
            yerr=np.std(np.array(cov_region)))
    data.loc[len(data)] = ["Tahv", np.mean(np.array(cov_region)), np.std(np.array(cov_region))]

    plt.xticks([0, 1, ], ["Mock", "Tahv"], rotation=0, fontsize=20)
    plt.ylabel("Mean amplitude [pV]", fontsize=25)

    plt.savefig(
        os.path.join(P.RESULTS,
                     f"Amplitude for Mock,Tahv in {min_freq}-{max_freq} Hz at {timepoint} for {batch}.png"),
        dpi=1200)
    data.to_csv(
        os.path.join(P.RESULTS,
                     f"Amplitude for Mock,Tahv in {min_freq}-{max_freq} Hz at {timepoint} for {batch}.csv"),
        index=False)
    if show:
        plt.show()


def Feature_importance_for_regionHz_at_T_24H_batch_for_Mock_CoV(min_freq=0, max_freq=5000, batch="all slices",
                                                                timepoint="T=48H"):
    percentiles = 0.1
    show = False
    batches = {"batch 1": ["1", "2", "3"], "batch 2": ["4", "5", "6", ], "batch 3": ["7", "8", "9", ],
               "batch 4": ["10", "11", "12"],
               "batch 1_2": ["1", "2", "3", "4", "5", "6", ], "batch 1_4": ["1", "2", "3", "10", "11", "12", ],
               "batch 2_4": ["4", "5", "6", "10", "11", "12"],
               "all slices": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]}

    class1 = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                             to_include=("freq_50hz_sample", timepoint),
                                             to_exclude=("TTX", "RG27", "NI"),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment=f" {batch}",
                                             target_keys={'NI': 'NI', 'TAHV': 'TAHV'})

    discarded_class1 = fp.discard_outliers_by_iqr(class1, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    class2 = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                             to_include=("freq_50hz_sample", timepoint),
                                             to_exclude=("TTX", "RG27", "TAHV"),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment=f" {batch}",
                                             target_keys={'NI': 'NI', 'TAHV': 'TAHV'})

    discarded_class2 = fp.discard_outliers_by_iqr(class2, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    discarded_class2.replace(f"NI {batch}", "Mock", inplace=True)
    discarded_class1.replace(f"TAHV {batch}", "Tahv", inplace=True)
    train_df = pd.concat([discarded_class1, discarded_class2], ignore_index=True)

    rfc, _ = fl.train_RFC_from_dataset(train_df)

    _, mean_importance = fl.get_top_features_from_trained_RFC(rfc, percentage=1, show=show, save=False, title='',
                                                              savepath='')
    plt.figure(figsize=(9, 8))
    plt.plot(mean_importance, color='b', linewidth=1)
    plt.fill_between([x for x in range(0, 300)], mean_importance, color='b', alpha=.5)

    hertz = []
    factor = 5000 / 300
    for i in range(300):
        hertz.append(int(i * factor))

    xticks = [x for x in range(0, 300, 50)]
    new_ticks = [hertz[x] for x in xticks]
    xticks.append(300)
    new_ticks.append(5000)
    plt.xticks(xticks, new_ticks, rotation=15, fontsize=15)
    plt.xlabel("Frequency-like features [Hz]", fontsize=25)
    plt.ylabel("Feature importance [AU]", fontsize=25)
    plt.savefig(
        os.path.join(P.RESULTS,
                     f"Feature importance for {min_freq}-{max_freq}Hz at {timepoint} {batch} for Mock,Tahv.png"),
        dpi=1200)
    if show:
        plt.show()


def Smoothened_frequencies_regionHz_Mock_CoV_on_batch(min_freq=0, max_freq=5000, batch="all slices", timepoint="T=48H"):
    percentiles = 0.1
    batches = {"batch 1": ["1", "2", "3"], "batch 2": ["4", "5", "6", ], "batch 3": ["7", "8", "9", ],
               "batch 4": ["10", "11", "12"],
               "batch 1_2": ["1", "2", "3", "4", "5", "6", ], "batch 1_4": ["1", "2", "3", "10", "11", "12", ],
               "batch 2_4": ["4", "5", "6", "10", "11", "12"],
               "all slices": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]}

    show = False
    tahvni = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                             to_include=("freq_50hz_sample", timepoint),
                                             to_exclude=("TTX", "RG27",),
                                             verbose=False,
                                             save=False,
                                             freq_range=(min_freq, max_freq),
                                             select_samples=batches[batch],
                                             separate_samples=False,
                                             label_comment="",
                                             target_keys={'NI': 'NI', 'TAHV': 'TAHV'})

    discarded_tahvni = fp.discard_outliers_by_iqr(tahvni, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    discarded_tahvni.replace("NI", "Mock", inplace=True)
    discarded_tahvni.replace("TAHV", "Tahv", inplace=True)

    plt.figure(figsize=(8, 8))
    plt.plot(discarded_tahvni.loc[discarded_tahvni["label"] == "Mock"].mean(axis=0), label="Mock", linewidth=1,
             color='g')
    plt.fill_between([x for x in range(0, 300)],
                     discarded_tahvni.loc[discarded_tahvni["label"] == "Mock"].mean(axis=0).subtract(
                         discarded_tahvni.loc[discarded_tahvni["label"] == "Mock"].std(axis=0)),
                     discarded_tahvni.loc[discarded_tahvni["label"] == "Mock"].mean(axis=0).add(
                         discarded_tahvni.loc[discarded_tahvni["label"] == "Mock"].std(axis=0)),
                     color='g', alpha=.5)
    plt.plot(discarded_tahvni.loc[discarded_tahvni["label"] == "Tahv"].mean(axis=0), label="Tahv",
             linewidth=1, color='b')
    plt.fill_between([x for x in range(0, 300)],
                     discarded_tahvni.loc[discarded_tahvni["label"] == "Tahv"].mean(axis=0).subtract(
                         discarded_tahvni.loc[discarded_tahvni["label"] == "Tahv"].std(axis=0)),
                     discarded_tahvni.loc[discarded_tahvni["label"] == "Tahv"].mean(axis=0).add(
                         discarded_tahvni.loc[discarded_tahvni["label"] == "Tahv"].std(axis=0)),
                     color='b', alpha=.5)

    ratio = 5000 / 300
    x_ds = [x for x in range(0, 301, 15)]
    x_freq = [int(x * ratio) for x in x_ds]
    plt.xticks(x_ds, x_freq, rotation=45)
    plt.xlabel("Smoothened frequencies [Hz]", fontsize=25)
    plt.ylabel("Amplitude [pV]", fontsize=25)
    plt.legend(prop={'size': 20})
    plt.savefig(os.path.join(P.RESULTS,
                             f"Smoothened frequencies {min_freq}-{max_freq}Hz Mock-Tahv on {batch} {timepoint}.png"),
                dpi=1200)
    if show:
        plt.show()
