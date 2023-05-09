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
from scipy.stats.stats import pearsonr

pd.set_option('display.max_columns', None)
import complete_procedures as cp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.signal import find_peaks


def main():
    timepoint = "T=48H"
    min_freq = 500
    max_freq = 3000
    percentiles = 0.1
    batches = {"batch 1": ["1", "2", "3"], "batch 2": ["4", "5", "6", ], "batch 3": ["7", "8", "9", ],
               "batch 4": ["10", "11", "12"],
               "batch 1_2": ["1", "2", "3", "4", "5", "6", ], "batch 1_4": ["1", "2", "3", "10", "11", "12", ],
               "batch 2_4": ["4", "5", "6", "10", "11", "12"],
               "all slices": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]}

    class1_df = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                                to_include=("odd_harmonics", timepoint),
                                                to_exclude=("TTX", "RG27", "TAHV"),
                                                verbose=False,
                                                save=False,
                                                freq_range=(min_freq, max_freq),
                                                select_samples=batches["batch 1"],
                                                separate_samples=False,
                                                label_comment=f"",
                                                target_keys={'NI': 'Mock b1',
                                                             'TAHV': 'Tahv b1'})  # IMPORTANT, else it may result in an empty dataframe
    discarded_class1 = fp.discard_outliers_by_iqr(class1_df, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')
    class2_df = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                                to_include=("odd_harmonics", timepoint),
                                                to_exclude=("TTX", "RG27", "TAHV"),
                                                verbose=False,
                                                save=False,
                                                freq_range=(min_freq, max_freq),
                                                select_samples=batches["batch 2"],
                                                separate_samples=False,
                                                label_comment=f"",
                                                target_keys={'NI': 'Mock b2',
                                                             'TAHV': 'Tahv b2'})  # IMPORTANT, else it may result in an empty dataframe
    discarded_class2 = fp.discard_outliers_by_iqr(class2_df, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')
    class3_df = fp.make_dataset_from_freq_files(parent_dir=P.NODRUG,
                                                to_include=("odd_harmonics", timepoint),
                                                to_exclude=("TTX", "RG27", "TAHV"),
                                                verbose=False,
                                                save=False,
                                                freq_range=(min_freq, max_freq),
                                                select_samples=batches["batch 4"],
                                                separate_samples=False,
                                                label_comment=f"",
                                                target_keys={'NI': 'Mock b4',
                                                             'TAHV': 'Tahv b4'})  # IMPORTANT, else it may result in an empty dataframe
    discarded_class3 = fp.discard_outliers_by_iqr(class3_df, low_percentile=percentiles,
                                                  high_percentile=1 - percentiles,
                                                  mode='capping')

    global_df = pd.concat([discarded_class1, discarded_class2, discarded_class3], ignore_index=True)
    rfc, _ = fl.train_RFC_from_dataset(global_df)


    fl.test_rfc_by_confusion(rfc, global_df, training_targets=(f'Mock b1', f'Mock b2', f'Mock b4'),
                             testing_targets=tuple(set(list((
                                 f'Mock b1', f'Mock b2', f'Mock b4',)))),
                             show=True, verbose=False, #savepath=os.path.join(P.RESULTS, batch),
                             title=f"Confusion matrix train on {timepoint} Mock between batches 1, 2 and 4 "
                                   f"50-3000 Hz odd harmonics",
                             iterations=5, )
now = datetime.datetime.now()
print(now)
main()
print("RUN:", datetime.datetime.now() - now)
