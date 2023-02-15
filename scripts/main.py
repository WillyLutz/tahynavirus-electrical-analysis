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
    for batch in ["batch 1", "batch 2", "batch 12", ]:
        fig_confusion_matrix_train_on_batch_Mock_tahv_0_5000Hz_test_on_rg27(batch)


def fig_confusion_matrix_train_on_batch_Mock_tahv_0_5000Hz_test_on_rg27(batch):
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
    rfc, _ = fl.train_RFC_from_dataset(discarded_tahvni)

    global_df = pd.concat([discarded_tahvni, discarded_tahv_rg27], ignore_index=True)
    fl.test_model_by_confusion(rfc, global_df, training_targets=(f'Mock', f'Tahv'),
                               testing_targets=(
                                   f'Mock', f'Tahv', f'RG27-treated Tahv',),
                               show=show, verbose=False, savepath=P.RESULTS,
                               title=f"Fig Confusion matrix train on T=48H Mock,Tahv test on Mock,Tahv,Stachel for {batch} "
                                     f"{min_freq}-{max_freq}Hz",
                               iterations=5, )


now = datetime.datetime.now()
print(now)
main()
print("RUN:", datetime.datetime.now() - now)
