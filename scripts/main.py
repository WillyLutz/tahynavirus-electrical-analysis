import datetime
import shutil
import time

import pandas as pd
import os
import fileinput
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import signal_processing as spr
import data_processing as dpr
import machine_learning as ml
import seaborn as sns
import PATHS as P
import data_processing as dp
import pickle
import forestci as fci
from pathlib import Path
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import fiiireflyyy.firelearn as fl
import fiiireflyyy.firefiles as ff
import fiiireflyyy.fireprocess as fp

pd.set_option('display.max_columns', None)
import complete_procedures as cp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def main():
    min_freq = 0
    max_freq = 5000
    for batch in ["batch 1", ]:
        timepoint = "T=48H"
        show = False
        percentiles = 0.1
        batches = {"batch 1": ["1", "2", "3"], "batch 2": ["4", "5", "6", ], "batch 3": ["7", "8", "9", ],
                   "batch 4": ["10", "11", "12"],
                   "batch 1_2": ["1", "2", "3", "4", "5", "6", ], "batch 1_4": ["1", "2", "3", "10", "11", "12", ],
                   "batch 2_4": ["4", "5", "6", "10", "11", "12"],
                   "all slices": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]}

        class1_2 = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                                   to_include=("freq_50hz_sample", timepoint),
                                                   to_exclude=("TTX", "RG27",),
                                                   verbose=False,
                                                   save=False,
                                                   freq_range=(min_freq, max_freq),
                                                   select_samples=batches[batch],
                                                   separate_samples=False,
                                                   label_comment="",
                                                   target_keys={'NI': 'Mock', 'TAHV': 'Tahv'})
        discarded_class1_2 = fp.discard_outliers_by_iqr(class1_2, low_percentile=percentiles,
                                                        high_percentile=1 - percentiles,
                                                        mode='capping')
        class3 = fp.make_dataset_from_freq_files(parent_dir=P.RG27,
                                                 to_include=("freq_50hz_sample", timepoint),
                                                 to_exclude=("TTX", "NI"),
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
        mock = discarded_class1_2.loc[discarded_class1_2["label"] == "Mock"].mean(axis=0)
        tahv = discarded_class1_2.loc[discarded_class1_2["label"] == "Tahv"].mean(axis=0)
        rg27 = discarded_class3.loc[discarded_class3["label"] == "RG27-treated Tahv"].mean(axis=0)
        plt.plot(mock, color="green", alpha=.5, label="mock")
        plt.plot(tahv, color="blue", alpha=.5, label="tahv")
        plt.plot(rg27, color='red', alpha=.5, label="rg27-treated Tahv")
        plt.legend()
        plt.show()


# fig_confusion_matrix_train_on_batch_Mock_tahv_0_5000Hz_test_on_rg27("batch 3")


def fig_confusion_matrix_train_on_batch_Mock_tahv_0_5000Hz_test_on_rg27(batch):
    show = True
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

    pca, pcdf, ratios = fl.fit_pca(discarded_tahvni, n_components=3)
    fl.plot_pca(pcdf, ratios=())


now = datetime.datetime.now()
print(now)
main()
print("RUN:", datetime.datetime.now() - now)
